from typing import List

import numpy as np
import pandas as pd
from sklearn import preprocessing

from rcpp.width_calculator import WidthCalculator, CLTWidth, CLTQuantileVarianceWidth
from rcpp.risk_measure import RiskMeasure, MeanRiskMeasure, CVaRRiskMeasure
from rcpp.performativity_simulator import PerformativitySimulator
from rcpp.loss_simulator import LossSimulator, ZeroOneLossSimulator, ZeroUniformLossSimulator
from rcpp.main import run_experiment


class CreditScoringSimulator(PerformativitySimulator):
    def __init__(self, gamma, M, beta = 0.0):
        """
        Arguments:
        - gamma: paraeter in the Wasserstein distance formulation
        - M: bound on the base distribution PDF of the predictions Y_hat (not the loss)
        - beta: the beta in beta-CVaR, e.g., 0.9 for 90%
        """
        self.shift_size = gamma / M * (1 - beta)

    def simulate_shift(self, Z: List[np.ndarray], lambda_: float) -> List[np.ndarray]:
        assert len(Z) == 2
        Y_hat, Y = Z
        condition = (1 - lambda_) < (Y_hat - self.shift_size)
        Y_hat_new = Y_hat.copy()
        Y_hat_new[~condition] -= self.shift_size
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

    # balance classes
    default_indices = np.where(Y_all == 1)[0]
    other_indices = np.where(Y_all == 0)[0][:10000]
    indices = np.concatenate((default_indices, other_indices))

    X_balanced = X_all[indices]
    Y_balanced = Y_all[indices]

    # shuffle arrays
    p = np.random.permutation(len(indices))
    X_full = X_balanced[p]
    Y_full = Y_balanced[p]
    return X_full, Y_full, data



class args:
    alpha = 0.3         # risk control level
    tightness = 0.08    # tightness parameter, may throw error if too low
    delta = 0.1         # failure probability or confidence parameter
    tau = 1.0           # safety parameter
    N = 2000            # number of samples in cohort
    lambda_max = 1.0    # maximum value for lambda
    ell_max = 1.0
    gamma = 0.1         # parameter in the Wasserstein distance formulation, for simulating distribution shift



if __name__ == "__main__":
    # Example usage
    Z_cal = [np.random.rand(100), np.random.rand(100)]
    Z_test = [np.random.rand(100), np.random.rand(100)]

    M = 3.0
    width_calculator = CLTWidth(args.alpha, args.ell_max, tol=1e-5)
    risk_measure = MeanRiskMeasure()
    performativity_simulator = CreditScoringSimulator(gamma=args.gamma, M=M)
    loss_simulator = ZeroOneLossSimulator()

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression
    from collections import Counter

    # Set random seed for reproducibility
    np.random.seed(42)

    # Load data
    path_to_csv_file = './cs-training.csv'
    X_all, Y_all, data = load_data(path_to_csv_file)

    # Report full feature dimension
    d = X_all.shape[1] - 1
    print('d =', d)

    # Show class distribution in the full dataset
    print("Class distribution in full dataset:", Counter(Y_all))

    # Identify indices for each class
    idx_0 = np.where(Y_all == 0)[0]
    idx_1 = np.where(Y_all == 1)[0]

    # Choose equal number of samples from each class
    n_samples_per_class = min(len(idx_0), len(idx_1), 1500)  # capped at 1000 per class
    idx_balanced = np.concatenate([
        np.random.choice(idx_0, size=n_samples_per_class, replace=False),
        np.random.choice(idx_1, size=n_samples_per_class, replace=False),
    ])

    # Shuffle balanced indices
    np.random.shuffle(idx_balanced)

    # Create balanced training set
    X_train = X_all[idx_balanced]
    Y_train = Y_all[idx_balanced]

    model = LogisticRegression(fit_intercept=False)
    model.fit(X_train, Y_train)
    Y_hat = model.predict_proba(X_train)[:,1]
    counts, bins, _ = plt.hist(Y_hat, bins=30, density=True, color='skyblue', edgecolor='black', alpha=0.75, label='Histogram of y_hat')
    print(f"Verify that M={M} >= PDF upper bound = {counts.max()}.")

    # Remaining indices (not used in training) go to calibration/test pool
    idx_remaining = np.setdiff1d(np.arange(len(Y_all)), idx_balanced)
    X_temp = X_all[idx_remaining]
    Y_temp = Y_all[idx_remaining]
 
    # Create calibrate / test set  # TODO: fix
    X_cal = X_temp[:len(X_temp)//2]
    Y_cal_hat = model.predict_proba(X_cal)[:,1]
    Y_cal = Y_temp[:len(Y_temp)//2]

    X_test = X_temp[len(X_temp)//2:]
    Y_test_hat = model.predict_proba(X_test)[:,1]
    Y_test = Y_temp[len(Y_temp)//2:]

    # Show class distributions
    print("Class distribution in training set:", Counter(Y_train))
    print("Class distribution in calibration/test set:", Counter(Y_temp))

    run_experiment(
        Z_cal=[Y_cal_hat, Y_cal],
        Z_test=[Y_test_hat, Y_test],
        width_calculator=width_calculator,
        risk_measure=risk_measure,
        performativity_simulator=performativity_simulator,
        loss_simulator=loss_simulator,
        args=args,
        gammas=[0.1, 0.2, 0.5, 1., 1.2, 1.5],
        save_dir="./applications/credit_scoring/figures",
        num_iters=1000
    )
