from copy import deepcopy
from typing import List, Union

import numpy as np
import pandas as pd
from sklearn import preprocessing
from tqdm import tqdm

from rcpp.performativity_simulator import PerformativitySimulator
from rcpp.main import run_trajectory
from rcpp.width_calculator import WidthCalculator
from rcpp.risk_measure import RiskMeasure
from rcpp.loss_simulator import LossSimulator
from rcpp.main import plot_lambda_vs_iteration, plot_loss_vs_iteration, plot_final_loss_vs_iteration


def create_balanced_split(X_all, Y_all, num_balanced_samples, class1_proportion=0.5):
    """
    Splits a dataset into a subsample with a specified class balance and the remaining data.

    The target subsample (often called the 'balanced' part, though it can be skewed
    by `class1_proportion`) will have a specified total number of samples.
    Class labels in Y_all are assumed to be binary (0 and 1).

    Args:
        X_all (array-like): Feature data, typically a NumPy array or list of lists,
                            with shape (N, D), where N is the total number of
                            samples and D is the number of features.
        Y_all (array-like): Binary class labels (0 or 1), typically a NumPy array
                            or list, with shape (N,).
        num_balanced_samples (int): The total number of samples desired in the
                                   target subsample.
        class1_proportion (float, optional): The desired proportion of class 1
            in the target subsample. Defaults to 0.5 (aiming for a 50/50 balance).
            This value must be between 0 and 1, inclusive.

    Returns:
        tuple: A tuple containing four NumPy arrays:
               (X_balanced, Y_balanced, X_remaining, Y_remaining)
               - X_balanced (np.ndarray): Features of the target subsample.
               - Y_balanced (np.ndarray): Labels of the target subsample.
               - X_remaining (np.ndarray): Features of the data not included in the subsample.
               - Y_remaining (np.ndarray): Labels of the data not included in the subsample.

    Raises:
        ValueError: If `num_balanced_samples` is negative,
                    if `class1_proportion` is not in the range [0, 1],
                    or if the requested number of samples for either class
                    in the target subsample exceeds the number of available
                    samples of that class in the original dataset.
    """
    X_all_np = np.asarray(X_all)
    Y_all_np = np.asarray(Y_all)

    if num_balanced_samples < 0:
        raise ValueError("num_balanced_samples must be non-negative.")
    if not (0 <= class1_proportion <= 1):
        raise ValueError("class1_proportion must be between 0 and 1 (inclusive).")

    # Identify indices for each class in the original dataset
    class0_indices = np.where(Y_all_np == 0)[0]
    class1_indices = np.where(Y_all_np == 1)[0]

    num_class0_total = len(class0_indices)
    num_class1_total = len(class1_indices)

    # Calculate the number of samples to select from each class for the target subsample
    # np.round handles cases like 2.5 by rounding to the nearest even integer (e.g., 2.0).
    # This ensures that for class1_proportion=0.5 and an odd num_balanced_samples,
    # the counts are as close as possible (e.g., for 5 samples, 2 of one class, 3 of other).
    num_class1_balanced = int(np.round(num_balanced_samples * class1_proportion))
    num_class0_balanced = num_balanced_samples - num_class1_balanced
    
    # Ensure calculated numbers for the subsample are not negative.
    # Given the prior checks on num_balanced_samples and class1_proportion,
    # num_class1_balanced will be in [0, num_balanced_samples],
    # and num_class0_balanced will also be in [0, num_balanced_samples].
    # So, explicit checks for negativity here are redundant but harmless if extreme float issues were a concern.

    # Validate if enough samples are available in each class
    if num_class0_balanced > num_class0_total:
        raise ValueError(
            f"Not enough samples in class 0 for the target subsample. "
            f"Requested: {num_class0_balanced}, Available: {num_class0_total}"
        )
    if num_class1_balanced > num_class1_total:
        raise ValueError(
            f"Not enough samples in class 1 for the target subsample. "
            f"Requested: {num_class1_balanced}, Available: {num_class1_total}"
        )

    # Randomly select indices for the target subsample (without replacement)
    # np.random.choice handles size=0 correctly, returning an empty array.
    selected_class0_indices = np.random.choice(
        class0_indices, size=num_class0_balanced, replace=False
    )
    selected_class1_indices = np.random.choice(
        class1_indices, size=num_class1_balanced, replace=False
    )

    # Combine the selected indices from both classes
    balanced_indices = np.concatenate([selected_class0_indices, selected_class1_indices])
    
    # Shuffle the combined indices to ensure the samples in the subsample are mixed,
    # rather than grouped by class from the concatenation. This is an in-place shuffle.
    np.random.shuffle(balanced_indices)

    # Create the target subsample using the selected indices
    X_balanced = X_all_np[balanced_indices]
    Y_balanced = Y_all_np[balanced_indices]

    # Identify the indices for the remaining data
    # np.setdiff1d finds elements in the first array not in the second.
    # Both arrays of indices must be 1D. `assume_unique=True` can offer a performance gain
    # if it's known that balanced_indices contains unique elements (which it does).
    all_original_indices = np.arange(X_all_np.shape[0])
    remaining_indices = np.setdiff1d(all_original_indices, balanced_indices, assume_unique=True)

    # Create the remaining dataset
    X_remaining = X_all_np[remaining_indices]
    Y_remaining = Y_all_np[remaining_indices]

    return X_balanced, Y_balanced, X_remaining, Y_remaining


class CreditScoringSimulator(PerformativitySimulator):
    def __init__(self, M, beta = 0.0):
        """
        Arguments:
        - M: bound on the base distribution PDF of the predictions Y_hat (not the loss)
        - beta: the beta in beta-CVaR, e.g., 0.9 for 90%
        """
        self.M = M
        self.beta = beta

    def simulate_shift(self,
                       Z_base: Union[List[np.ndarray], None],
                       Z_prev: Union[List[np.ndarray], None],
                       lambda_: float,
                       gamma: float) -> List[np.ndarray]:
        shift_size = gamma / self.M * (1. - self.beta)
        assert len(Z_base) == 2
        Y_hat, Y = Z_base
        condition = (1 - lambda_) < (Y_hat - shift_size)
        Y_hat_new = Y_hat.copy()
        Y_hat_new[~condition] -= shift_size
        Y_hat_new = np.clip(Y_hat_new, 0, 1)
        return [Y_hat_new, Y]


def load_data(file_loc):
    """Load data from csv file.

    Parameters
    ----------
        file_loc: string
            path to the '.csv' training data file
    Returns
    -------
        X_full: np.array
            class-balanced data matrix     
        Y_full: np.array
            corresponding labels (0/1) 
        data: DataFrame
            raw data     
    """

    data = pd.read_csv(file_loc, index_col=0)
    data.dropna(inplace=True)

    # full data set
    X_all = data.drop('SeriousDlqin2yrs', axis=1)

    # zero mean, unit variance
    X_all = preprocessing.scale(X_all)

    # add bias term
    X_all = np.append(X_all, np.ones((X_all.shape[0], 1)), axis=1)

    # outcomes
    Y_all = np.array(data['SeriousDlqin2yrs'])

    return X_all, Y_all


def run_credit_experiment(
    Z,
    width_calculator: WidthCalculator,
    risk_measure: RiskMeasure,
    performativity_simulator: PerformativitySimulator,
    loss_simulator: LossSimulator,
    args,
    gammas: List[float] = [0.1, 0.2, 0.5, 1, 1.2, 1.5],
    save_dir: str = "./figures",
    num_iters: int = 1000,
):
    """
    Run the RCPP algorithm and plot the results.
    """

    def cut(data):
        tot = len(data[0])
        assert tot > args.N
        idx = np.random.choice(tot, size=args.N, replace=False)
        data_cal = [d[idx] for d in data]
        data_test = [d[~idx] for d in data]
        return data_cal, data_test

    # plot 1: change gamma, i.e. magnitude of distribution shift
    gamma_trajectories = []
    for gamma in gammas:
        args_copy = deepcopy(args)
        args_copy.gamma = gamma
        Z_cal, Z_test = cut(Z)
        trajectory = run_trajectory(
            Z_cal,
            Z_test,
            width_calculator,
            risk_measure,
            performativity_simulator,
            loss_simulator,
            args_copy
        )
        gamma_trajectories.append(trajectory)

    # Plot 1
    plot_lambda_vs_iteration(gammas, gamma_trajectories, args, save_dir=save_dir)

    trajectories = []
    for _ in tqdm(range(num_iters), desc="Running trials"):
        Z_cal, Z_test = cut(Z)
        trajectory = run_trajectory(
            Z_cal,
            Z_test,
            width_calculator,
            risk_measure,
            performativity_simulator,
            loss_simulator,
            args
        )
        trajectories.append(trajectory)

    # Plot 2/3: plot risk level v. iteration
    plot_loss_vs_iteration(trajectories, args, save_dir=save_dir)

    # Plot 4: plot final risk level v. iteration
    plot_final_loss_vs_iteration(trajectories, args, save_dir=save_dir)