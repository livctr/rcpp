"""
Main file for RCPP algorithm. 
"""
from rcpp.width_calculator import WidthCalculator
from rcpp.risk_measures import RiskMeasure
from rcpp.performativity_simulator import PerformativitySimulator
from rcpp.loss_simulator import LossSimulator

import numpy as np
from tqdm import tqdm


def binary_search_solver(objective, lambda_low, lambda_high, tol=1e-6, max_iter=100):
    if objective(lambda_low) <= 0:
        return lambda_low

    if objective(lambda_high) > 0:
        raise ValueError("No crossing found: objective > 0 for all λ in [lambda_low, lambda_high].")

    for _ in range(max_iter):
        mid = (lambda_low + lambda_high) / 2.0
        if objective(mid) <= 0:
            lambda_high = mid
        else:
            lambda_low = mid
        if lambda_high - lambda_low < tol:
            break
    return lambda_high


def find_guaranteed_T(
    width_calculator: WidthCalculator,
    delta: float,
    N: int,
    tightness: float,
    tau: float,
    lambda_max: float,
    max_T: int = 1000):
    """
    Searches 1 through T for the minimum number of iterations for guaranteed convergence .
    """
    best_T = None
    best_result = None

    for T in range(1, max_T + 1):
        width = width_calculator.get_width(delta / T, N)
        delta_lambda = (tightness - 2 * width) / (2 * tau)

        if delta_lambda <= 0 or delta_lambda >= 1:
            continue

        if T >= lambda_max / delta_lambda:
            best_T = T
            break

    if best_T is None:
        raise ValueError(f"Tightness {tightness} cannot be achieved in {max_T} iterations.")

    return best_T, delta_lambda


def run_trajectory(
    Z_cal,
    Z_test,
    width_calculator: WidthCalculator,
    risk_measure: RiskMeasure,
    performativity_simulator: PerformativitySimulator,
    loss_simulator: LossSimulator,
    args):
  
    # Note: the numbers correspond to the Algorithm in paper
    # 1. Initialize lambda^{(0)}
    lambda_ = args.lambda_max  # initialize the deployment threshold
    lambda_hat = lambda_       # deployment threshold to return
    lambdas = [lambda_]
    losses_tt = []
    losses_tm1_t = []

    # Jointly solve for T, delta_prime, and delta_lambda
    T, delta_lambda = find_guaranteed_T(width_calculator,
                                        args.delta,
                                        args.N,
                                        args.tightness,
                                        args.tau)
    bound = width_calculator.get_width(args.delta / T, args.N)

    while True:
        # Apply previous threshold lambda^{(t-1)}
        Z_cal_tm1 = performativity_simulator.simulate_shift(Z_cal, lambda_)
        Z_test_tm1 = performativity_simulator.simulate_shift(Z_test, lambda_)

        # [tracking] Calculate the realized loss L(lambda^{(t-1)}, lambda^{(t-1)})
        losses_tt.append(
            loss_simulator.calc_loss(Z_test_tm1, lambda_, do_new_sample=True)
        )

        # 4. Receive samples from previous threshold lambda^{(t-1)}
        idx = np.random.choice(len(Z_cal_tm1[0]), size=args.N, replace=True) 
        Z_sample = [Z[idx] for Z in Z_cal_tm1]

        # 5. Find lambda^{(t)}_mid
        def loss_at_new_lambda(lambda_new):
            losses = loss_simulator.calc_loss(Z_sample, lambda_new, do_new_sample=False)
            emp_risk = risk_measure.calculate(losses)
            return emp_risk + bound + args.tau * (lambda_ - lambda_new) - args.alpha
        loss_simulator.calc_loss(Z_sample, lambda_, do_new_sample=True)  # Set the randomness
        lambda_mid = binary_search_solver(loss_at_new_lambda, 0, 1)

        # 6. Set new lambda^{(t)}
        lambda_new = min(lambda_, lambda_mid)
        lambdas.append(lambda_)

        # [tracking] Calculate the realized loss L(lambda^{(t-1)}, lambda^{(t)})
        losses_tm1_t.append(
            loss_simulator.calc_loss(Z_test_tm1, lambda_new, do_new_sample=True)
        )

        # 7-8. Stopping condition 
        if lambda_new >= lambda_ - delta_lambda:
            lambda_hat = lambda_new
            break

        lambda_ = lambda_new

    # Calculate the L(lambda_hat, lambda_hat)
    Z_shifted = performativity_simulator.simulate_shift(Z_test, lambda_hat)
    losses_tt.append(
        loss_simulator.calc_loss(Z_shifted, lambda_hat, do_new_sample=True)
    )

    return {
        "lambda_hat": lambda_hat,
        "losses_tt": losses_tt,
        "losses_tm1_t": losses_tm1_t,
        "lambdas": lambdas,
    }
