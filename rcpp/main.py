"""
Main file for RCPP algorithm. 
"""
from typing import List, Tuple

from rcpp.width_calculator import WidthCalculator
from rcpp.risk_measure import RiskMeasure
from rcpp.performativity_simulator import PerformativitySimulator
from rcpp.loss_simulator import LossSimulator

import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
import os

from copy import deepcopy



class Trajectory:
    def __init__(self, lambda_hat, risks_tt, risks_tm1_t, lambdas, guaranteed_T, delta_lambda):
        self.lambda_hat = np.array(lambda_hat)
        self.risks_tt = np.array(risks_tt)
        self.risks_tm1_t = np.array(risks_tm1_t)
        self.lambdas = np.array(lambdas)
        self.guaranteed_T = guaranteed_T
        self.delta_lambda = delta_lambda


def binary_search_solver(objective, lambda_low, lambda_high, tol=1e-6, max_iter=100) -> float:
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
    max_T: int = 1000) -> Tuple[int, float]:
    """
    Searches 1 through T for the minimum number of iterations for guaranteed convergence .
    """
    best_T = None

    for T in range(1, max_T + 1):
        width = width_calculator.get_width(delta / T, N)
        delta_lambda = (tightness - 2 * width) / (2 * tau)

        if delta_lambda <= 0 or delta_lambda >= 1:
            continue

        if T >= lambda_max / delta_lambda:
            best_T = T
            break

    if best_T is None:
        raise ValueError(f"Tightness {tightness} cannot be achieved in {max_T} iterations. width@{max_T} = {width}")

    return best_T, delta_lambda


def run_trajectory(
    Z_cal,
    Z_test,
    width_calculator: WidthCalculator,
    risk_measure: RiskMeasure,
    performativity_simulator: PerformativitySimulator,
    loss_simulator: LossSimulator,
    args,
    control_risk: bool = True,
    num_iters: int = 10) -> Trajectory:
  
    performativity_simulator.reset()
    # Note: the numbers correspond to the Algorithm in paper
    # 1. Initialize lambda^{(0)}
    if control_risk:
        lambda_ = args.lambda_max  # initialize the deployment threshold
        lambda_hat = lambda_       # deployment threshold to return
        # Jointly solve for T, delta_prime, and delta_lambda
        T, delta_lambda = find_guaranteed_T(
            width_calculator,
            args.delta, args.N, args.tightness, args.tau, args.lambda_max
        )
        bound = width_calculator.get_width(args.delta / T, args.N)
    else:
        lambda_ = 0.0
        lambda_hat = 0.0
        T, delta_lambda = None, None

    lambdas = [lambda_]
    risks_tt = []
    risks_tm1_t = []
    Z_cal_tm1 = Z_cal
    Z_test_tm1 = Z_test
    iter = 0

    while True:
        # Apply previous threshold lambda^{(t-1)}
        cal_len, test_len = len(Z_cal[0]), len(Z_test[0])
        Z = [np.concatenate([Z_cal[i], Z_test[i]]) for i in range(len(Z_cal))]
        Z_tm1 = [np.concatenate([Z_cal_tm1[i], Z_test_tm1[i]]) for i in range(len(Z_cal_tm1))]
        Z_tm1 = performativity_simulator.simulate_shift(Z, Z_tm1, lambda_, args.gamma)
        Z_cal_tm1 = [Z_tm1[i][:cal_len] for i in range(len(Z_tm1))]
        Z_test_tm1 = [Z_tm1[i][cal_len:] for i in range(len(Z_tm1))]

        # [tracking] Calculate the realized loss L(lambda^{(t-1)}, lambda^{(t-1)})
        risks_tt.append(
            risk_measure.calculate(
                loss_simulator.calc_loss(Z_test_tm1, lambda_, do_new_sample=True)
            )
        )

        # 4. Receive samples from previous threshold lambda^{(t-1)}
        idx = np.random.choice(len(Z_cal_tm1[0]), size=args.N, replace=True) 
        Z_sample = [Z[idx] for Z in Z_cal_tm1]

        if control_risk:
            # 5. Find lambda^{(t)}_mid
            def loss_at_new_lambda(lambda_new):
                losses = loss_simulator.calc_loss(Z_sample, lambda_new, do_new_sample=False)
                emp_risk = risk_measure.calculate(losses)
                return emp_risk + bound + args.tau * (lambda_ - lambda_new) - args.alpha
            loss_simulator.calc_loss(Z_sample, lambda_, do_new_sample=True)  # Set the randomness
            lambda_mid = binary_search_solver(loss_at_new_lambda, 0, 1)

            # 6. Set new lambda^{(t)}
            lambda_new = min(lambda_, lambda_mid)
        else:
            lambda_new = 0.0
        lambdas.append(lambda_new)

        # [tracking] Calculate the realized loss L(lambda^{(t-1)}, lambda^{(t)})
        risks_tm1_t.append(
            risk_measure.calculate(
                loss_simulator.calc_loss(Z_test_tm1, lambda_new, do_new_sample=True)
            )
        )

        # 7-8. Stopping condition 
        if control_risk and lambda_new >= lambda_ - delta_lambda:
            lambda_hat = lambda_new
            break

        lambda_ = lambda_new

        iter += 1
        if not control_risk and iter >= num_iters:
            lambda_hat = 0.0
            break

    # Calculate the L(lambda_hat, lambda_hat)
    Z_shifted = performativity_simulator.simulate_shift(Z_test, Z_test_tm1, lambda_hat, args.gamma)
    risks_tt.append(
        risk_measure.calculate(
            loss_simulator.calc_loss(Z_shifted, lambda_hat, do_new_sample=True)
        )
    )

    return Trajectory(
        lambda_hat=lambda_hat,
        risks_tt=risks_tt,
        risks_tm1_t=risks_tm1_t,
        lambdas=lambdas,
        guaranteed_T=T,
        delta_lambda=delta_lambda,
    )


def plot_lambda_vs_iteration(
    gammas: List[float],
    trajectories: List[Trajectory],
    args,
    save_dir: str = "./figures"):
    """
    Plot the trajectory of lambda over iterations for different gamma values.
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(7, 4))
    plt.ylim(0, args.lambda_max)
    colors = ['blue', 'green', 'red', 'orange', 'black', 'purple', 'brown']
    for i in range(len(gammas)):
        plt.plot(trajectories[i].lambdas, label=rf"$\gamma$ = {gammas[i]}" if len(gammas) > 1 else None, color=colors[i], alpha=0.7, linewidth=2)
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Threshold', fontsize=14)
    plt.ylabel(r'$\lambda_t$', fontsize=14)
    plt.ylim(0, 1.1)
    print(f"Guaranteed convergence by T={trajectories[i].guaranteed_T}")
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    if save_dir:
        save_path = os.path.join(save_dir, "lambda_vs_iteration.pdf")
        plt.savefig(save_path, dpi=300)
        print(f"Saved lambda plot to {save_path}")

    plt.figure(figsize=(7, 4))
    max_iters = 0
    for i in range(len(gammas)):
        trajectory = trajectories[i]
        diffs = np.abs(trajectory.lambdas[1:] - trajectory.lambdas[:-1])
        max_iters = max(max_iters, len(diffs))
        plt.plot(range(1, len(diffs) + 1), diffs, marker='o', markersize=2,
                 label=rf"$\gamma$ = {gammas[i]}" if len(gammas) > 1 else None, color=colors[i], alpha=0.5)

    # `delta_lambda` is the same for all gammas
    plt.axhline(y=trajectory.delta_lambda, color='gray', linestyle='--', linewidth=1.5, label=rf"$\Delta\lambda$")
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel(r'$\lambda_{t-1} - \lambda_t$', fontsize=14)
    plt.xticks(ticks=range(max_iters + 1), labels=[str(i) for i in range(max_iters + 1)], fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_dir:
        save_path = os.path.join(save_dir, "lambda_diff_vs_iteration.pdf")
        plt.savefig(save_path, dpi=300)
        print(f"Saved lambda plot to {save_path}")


def plot_loss_vs_iteration(
    trajectories: List[Trajectory],
    args,
    save_dir: str = "./figures"
):
    num_trajectories = len(trajectories)
    lower_q, upper_q = args.delta / 2, 1 - args.delta / 2

    # Each of these arrays is of length 2 * max_t
    # It contains max_t pairs of L(lambda_t, lambda_t) and L(lambda_{t-1}, lambda_t) in that order
    risk_mean = []
    trajectories_remaining = []

    while True:
        keep_going = False
        iter = len(risk_mean) // 2

        risks_tm1_ts = []
        risks_tts = []
        remaining = 0
        for i in range(num_trajectories):
            trajectory = trajectories[i]
            if iter < len(trajectory.risks_tm1_t):
                risks_tts.append(trajectory.risks_tt[iter])
                risks_tm1_ts.append(trajectory.risks_tm1_t[iter])
                remaining += 1
                keep_going = True

        if not keep_going:
            break
    
        risk_mean.append(np.mean(risks_tts))
        risk_mean.append(np.mean(risks_tm1_ts))
        trajectories_remaining.append(remaining)
    
    # Get x-axis values
    offset = 0.05
    num_iters = len(risk_mean) // 2
    ts = np.empty(2 * num_iters)
    ts[0::2] = np.arange(num_iters) - offset
    ts[1::2] = np.arange(num_iters) + offset

    plt.figure(figsize=(7, 4))

    # plt.plot(ts, risk_mean, color='blue', label="Risk", alpha=1.0, linewidth=5)
    for trajectory in trajectories:
        trajectory_risk = np.empty(len(trajectory.risks_tt[:-1]) + len(trajectory.risks_tm1_t))
        trajectory_risk[0::2] = trajectory.risks_tt[:-1]
        trajectory_risk[1::2] = trajectory.risks_tm1_t
        plt.plot(ts[:len(trajectory_risk)], trajectory_risk, color='blue', alpha=min(1, 10. / num_trajectories), linewidth=1)

    # plt.fill_between(ts, risk_lower_bnd, risk_upper_bnd, color='orange', alpha=0.4)

    plt.axhline(args.alpha, linestyle='--', color='red', label=r'Upper bound $\alpha$')
    plt.axhline(args.alpha - args.tightness, linestyle='--', color='green', label=r'Lower Bound $\alpha - \Delta\alpha$')

    plt.xlabel("Iteration", fontsize=14)
    plt.ylabel("Risk", fontsize=14)
    # Set x-axis to show 1-based iteration numbers
    plt.xticks(ticks=range(num_iters), labels=[str(i) for i in range(num_iters)], fontsize=12)
    plt.ylim(0, 0.5)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    if save_dir:
        save_path = os.path.join(save_dir, "loss_vs_iteration.pdf")
        plt.savefig(save_path, dpi=300)
        print(f"Saved loss plot to {save_path}")



def plot_final_loss_vs_iteration(
    trajectories: List[Trajectory],
    args,
    save_dir: str = "./figures"
):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    points = [
        (len(trajectory.risks_tt) - 1, trajectory.risks_tt[-1]) for trajectory in trajectories
    ]
    points = np.array(points)

    tot = len(trajectories)
    below_lower_bnd = np.sum(points[:, 1] < args.alpha - args.tightness)
    above_upper_bnd = np.sum(points[:, 1] > args.alpha)
    between_bnds = tot - below_lower_bnd - above_upper_bnd
    print(f"Final loss: {tot} total, {below_lower_bnd} below lower bound, {between_bnds} between bounds, {above_upper_bnd} above upper bound")

    plt.figure(figsize=(7, 4))
    plt.scatter(points[:, 0], points[:, 1], color='blue', alpha=0.6, marker='x')

    plt.axhline(args.alpha, linestyle='--', color='#d62728', label=r'Upper Bound $\alpha$', linewidth=1.5)  # red
    plt.axhline(args.alpha - args.tightness, linestyle='--', color='#2ca02c', label=r'Lower Bound $\alpha - \Delta\alpha$', linewidth=1.5)  # green

    plt.xlabel("Stopping Iteration", fontsize=14)
    plt.ylabel("Risk", fontsize=14)

    xlim_lower = np.min(points[:, 0]) - 1
    xlim_upper = np.max(points[:, 0]) + 1
    plt.xlim(xlim_lower, xlim_upper)

    ylim_lower = min(args.alpha - args.tightness * 1.5, points[:, 1].min())
    ylim_upper = max(args.alpha + args.tightness * 1.5, points[:, 1].max())

    plt.ylim(ylim_lower, ylim_upper)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()

    if save_dir:
        save_path = os.path.join(save_dir, "final_loss_vs_iteration.pdf")
        plt.savefig(save_path, dpi=300)
        print(f"Saved final loss plot to {save_path}")

