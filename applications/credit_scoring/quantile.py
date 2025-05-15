import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

from rcpp.width_calculator import CLTQuantileVarianceWidth
from rcpp.risk_measure import CVaRRiskMeasure
from rcpp.loss_simulator import ZeroUniformLossSimulator

from applications.credit_scoring.utils import (
    load_data,
    create_balanced_split,
    CreditScoringSimulator,
    run_credit_experiment
)


class Args:
    def __init__(self):
        self.alpha = 0.25         # risk control level
        self.tightness = 0.12    # tightness parameter, may throw error if too low
        self.delta = 0.1        # failure probability or confidence parameter
        self.tau = 1.0           # safety parameter
        self.N = 10000            # number of samples in cohort
        self.lambda_min = 0.0
        self.lambda_safe = 1.0    # maximum value for lambda
        self.ell_max = 1.0
        self.gamma = 1.0         # parameter in the Wasserstein distance formulation, for simulating distribution shift
        self.shift_size = 0.3

        self.beta = 0.9


if __name__ == "__main__":

    args = Args()
    np.random.seed(123)

    save_dir = "./applications/credit_scoring/figures/quantile_loss/"

    # Load data
    path_to_csv_file = './applications/credit_scoring/data/cs-training.csv'
    X_all, Y_all = load_data(path_to_csv_file)  # Shape (N, 11), (N,)

    # Train model
    X_train, Y_train, X_temp, Y_temp = create_balanced_split(X_all, Y_all, num_balanced_samples=1500, class1_proportion=0.5)
    model = LogisticRegression(fit_intercept=False)
    model.fit(X_train, Y_train)
    del X_train, Y_train

    # Generate PRC data
    X, Y = X_temp, Y_temp
    p = Y.mean()
    print(f"Base rate: p = {p}.")
    Y_hat = model.predict_proba(X)[:,1]
    del X_temp, Y_temp

    # Data stats
    Y_hat = model.predict_proba(X)[:,1]
    counts, bins, _ = plt.hist(Y_hat[Y==1], bins=20, density=True, color='skyblue', edgecolor='black', alpha=0.75, label='Histogram of y_hat')
    plt.xlabel(r'f(x)', fontsize=20)
    plt.ylabel('density', fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_dir + "density.pdf", dpi=300)
    M = counts.max() # PDF upper bound
    print(f"Rate of positive labels in set = {Y.mean():.4f}.")
    print(f"PDF upper bound = {M}.")
    Cp = Y.mean() * M
    print(f"Cp = {Cp:.4f}.")

    # Run experiment
    width_calculator = CLTQuantileVarianceWidth(args.beta, p)
    risk_measure = CVaRRiskMeasure(args.beta)
    performativity_simulator = CreditScoringSimulator(args.shift_size)
    loss_simulator = ZeroUniformLossSimulator()
    run_credit_experiment(
        Z=[Y_hat, Y],
        width_calculator=width_calculator,
        risk_measure=risk_measure,
        performativity_simulator=performativity_simulator,
        loss_simulator=loss_simulator,
        args=args,
        taus=[1e-3, 2e-1, 4e-1, 1, 1.6, 2, 4],
        save_dir=save_dir,
        num_iters=1000
    )
