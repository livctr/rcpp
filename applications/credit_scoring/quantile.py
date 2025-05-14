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
        self.delta = 0.1         # failure probability or confidence parameter
        self.tau = 1.0           # safety parameter
        self.N = 10000            # number of samples in cohort
        self.lambda_max = 1.0    # maximum value for lambda
        self.ell_max = 1.0
        self.gamma = 1.0         # parameter in the Wasserstein distance formulation, for simulating distribution shift

        self.beta = 0.9


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
    print(f"PDF upper bound = {M}.")
    del X_train, Y_train

    # Generate PRC data
    args = Args()
    X, Y = X_temp, Y_temp
    p = Y.mean()
    print(f"Base rate: p = {p}.")
    Y_hat = model.predict_proba(X)[:,1]
    del X, X_temp, Y_temp

    # Run experiment
    width_calculator = CLTQuantileVarianceWidth(args.beta, p)
    risk_measure = CVaRRiskMeasure(args.beta)
    performativity_simulator = CreditScoringSimulator(M=M, beta=args.beta)
    loss_simulator = ZeroUniformLossSimulator()
    run_credit_experiment(
        Z=[Y_hat, Y],
        width_calculator=width_calculator,
        risk_measure=risk_measure,
        performativity_simulator=performativity_simulator,
        loss_simulator=loss_simulator,
        args=args,
        gammas=[0.1, 1, 3, 10, 30],
        save_dir="./applications/credit_scoring/figures/quantile_loss/",
        num_iters=1000
    )
