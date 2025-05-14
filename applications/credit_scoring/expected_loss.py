import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

from rcpp.width_calculator import CLTWidth
from rcpp.risk_measure import MeanRiskMeasure
from rcpp.loss_simulator import ZeroOneLossSimulator

from applications.credit_scoring.utils import (
    load_data,
    create_balanced_split,
    CreditScoringSimulator,
    run_credit_experiment
)


class Args:
    def __init__(self):
        self.alpha = 0.3         # risk control level
        self.tightness = 0.08    # tightness parameter, may throw error if too low
        self.delta = 0.1        # failure probability or confidence parameter
        self.tau = 1.0           # safety parameter
        self.N = 2000            # number of samples in cohort
        self.lambda_max = 1.0    # maximum value for lambda
        self.ell_max = 1.0
        self.gamma = 1.0         # parameter in the Wasserstein distance formulation, for simulating distribution shift


if __name__ == "__main__":

    np.random.seed(42)

    # Load data
    path_to_csv_file = './applications/credit_scoring/data/cs-training.csv'
    X_all, Y_all = load_data(path_to_csv_file)  # Shape (N, 11), (N,)

    # Train model
    X_train, Y_train, X_temp, Y_temp = create_balanced_split(X_all, Y_all, num_balanced_samples=1500, class1_proportion=0.5)
    model = LogisticRegression(fit_intercept=False)
    model.fit(X_train, Y_train)
    Y_hat = model.predict_proba(X_train)[:,1]
    Y_hat = Y_hat[Y_train == 1]
    counts, bins, _ = plt.hist(Y_hat, bins=30, density=True, color='skyblue', edgecolor='black', alpha=0.75, label='Histogram of y_hat')
    M = counts.max() # PDF upper bound
    print(f"Rate of positive labels in training set = {Y_train.mean():.4f}.")
    print(f"PDF upper bound = {M}.")
    Cp = Y_train.mean() * M
    print(f"Cp = {Cp:.4f}.")
    del X_train, Y_train

    # Generate PRC data
    args = Args()
    idx_1 = np.where(Y_temp == 1)[0]
    idx_0 = np.where(Y_temp == 0)[0][:10000]
    indices = np.concatenate((idx_1, idx_0))
    X = X_temp[indices]
    Y_hat = model.predict_proba(X)[:,1]
    Y = Y_temp[indices]
    del idx_1, idx_0, indices, X, X_temp, Y_temp

    # Run experiment
    width_calculator = CLTWidth(args.alpha, args.ell_max, tol=1e-5)
    risk_measure = MeanRiskMeasure()
    performativity_simulator = CreditScoringSimulator(M=Cp)
    loss_simulator = ZeroOneLossSimulator()
    run_credit_experiment(
        Z=[Y_hat, Y],
        width_calculator=width_calculator,
        risk_measure=risk_measure,
        performativity_simulator=performativity_simulator,
        loss_simulator=loss_simulator,
        args=args,
        gammas=[0.1, 0.2, 0.5, 1, 2.0, 5.0, 10.0],
        save_dir="./applications/credit_scoring/figures/expected_loss/",
        num_iters=1000
    )
