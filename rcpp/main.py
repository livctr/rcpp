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
    lambda_range: float,
    max_T: int = 10000) -> Tuple[int, float]:
    """
    Searches 1 through T for the minimum number of iterations for guaranteed convergence .
    """
    best_T = None

    for T in range(1, max_T + 1):
        width = width_calculator.get_width(delta / T, N)
        delta_lambda = (tightness - 2. * width) / (2. * tau)

        if delta_lambda < float(lambda_range) / T:
            continue

        best_T = T
        break

    if best_T is None:
        raise ValueError(f"Tightness {tightness} cannot be achieved in {max_T} iterations. width@{max_T} = {width}. {tightness - 2. * width}")

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
        lambda_ = args.lambda_safe  # initialize the deployment threshold
        lambda_hat = lambda_       # deployment threshold to return
        # Jointly solve for T, delta_prime, and delta_lambda
        T, delta_lambda = find_guaranteed_T(
            width_calculator,
            args.delta, args.N, args.tightness, args.tau, args.lambda_safe - args.lambda_min
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
    taus: List[float],
    trajectories: List[List[Trajectory]],
    colors: List[str],
    args,
    save_dir: str = "./figures"):
    """
    Plot the trajectory of lambda over iterations for different tau values.
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(7, 4))
    plt.ylim(0, args.lambda_safe)
    max_iters = 0
    for i in range(len(trajectories)):
        label = rf"$\tau$ = {taus[i]}" if len(taus) > 1 else None
        plt.plot(trajectories[i][0].lambdas, label=label, color=colors[i])
        plt.scatter([len(trajectories[i][0].lambdas)-1], [trajectories[i][0].lambdas[-1]], color=colors[i], marker='x', s=50)
        max_iters = max(max_iters, len(trajectories[i][0].lambdas))
    plt.xlabel('Iteration', fontsize=20)
    plt.ylabel(r'$\lambda_t$', fontsize=20)
    plt.xticks(ticks=range(0, max_iters + 1, 2), labels=[str(i) for i in range(0, max_iters + 1, 2)], fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(0, 1.1)
    leg = plt.legend(loc='upper right')
    for lh in leg.legend_handles:
        lh.set_alpha(1)
    plt.grid(True)
    plt.tight_layout()
    if save_dir:
        save_path = os.path.join(save_dir, "lambda_vs_iteration.pdf")
        plt.savefig(save_path, dpi=300)
        print(f"Saved lambda plot to {save_path}")

    plt.figure(figsize=(7, 4))
    max_iters = 0
    for i in range(len(trajectories)):
        last_diffs = []  # for scatter plot
        label = rf"$\tau$ = {taus[i]}" if len(taus) > 1 else None
        diffs = np.abs(trajectories[i][0].lambdas[1:] - trajectories[i][0].lambdas[:-1])

        max_iters = max(max_iters, len(diffs))
        plt.plot(range(1, len(diffs) + 1), diffs, markersize=2,
                label=label, color=colors[i])
        if len(diffs) > 0:
            last_diffs.append((len(diffs), diffs[-1]))
        
        if last_diffs:
            plt.scatter(*zip(*last_diffs), color=colors[i], marker='x', s=50)

        plt.axhline(y=trajectories[i][0].delta_lambda, color=colors[i], linestyle='--', linewidth=1.5, alpha=0.7)

    # `delta_lambda` is the same for all taus
    plt.xlabel('Iteration', fontsize=20)
    plt.xticks(ticks=range(0, max_iters + 1, 2), labels=[str(i) for i in range(0, max_iters + 1, 2)], fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel(r'$\lambda_{t-1} - \lambda_t$', fontsize=20)
    plt.yscale('symlog', linthresh=1e-5)
    plt.ylim(bottom=-5e-6)
    plt.grid(True)
    plt.tight_layout()
    if save_dir:
        save_path = os.path.join(save_dir, "lambda_diff_vs_iteration.pdf")
        plt.savefig(save_path, dpi=300)
        print(f"Saved lambda plot to {save_path}")

def plot_loss_vs_iteration(
    tau: float,
    trajectories: List[Trajectory],
    color: str,
    args,
    save_dir: str = "./figures"
):
    plt.figure(figsize=(7, 4))

    tau_str = str(tau).replace('.', '_')

    offset = 0.05
    endpoints = []
    num_trajectories = len(trajectories)
    for i in range(num_trajectories):
        xs = []
        # Each of these arrays is of length 2 * max_t + 1
        # It contains max_t pairs of L(lambda_t, lambda_t) and L(lambda_{t-1}, lambda_t) in that order
        risks = []
        trajectory = trajectories[i]
        assert len(trajectory.risks_tt) == len(trajectory.risks_tm1_t) + 1
        for j in range(len(trajectory.risks_tt)):
            xs.append(j - offset)
            risks.append(trajectory.risks_tt[j])
            if j < len(trajectory.risks_tm1_t):
                xs.append(j + offset)
                risks.append(trajectory.risks_tm1_t[j])
            else:
                endpoints.append((j - offset, trajectory.risks_tt[j]))
        label = rf"$\tau$ = {tau}" if i == 0 else None
        plt.plot(xs, risks, color=color, alpha=min(1, 10. / num_trajectories), linewidth=0.5, label=label)
    plt.scatter(*zip(*endpoints), color=color, alpha=min(1, 10. / num_trajectories), marker='x', s=50)

    plt.axhline(args.alpha, linestyle='--', color='red', label=r'Upper bound $\alpha$')
    plt.axhline(args.alpha - args.tightness, linestyle='--', color='green', label=r'Lower Bound $\alpha - \Delta\alpha$')

    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("Risk", fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(0, 0.5)
    plt.legend(loc='lower right')
    for lh in plt.gca().get_legend().legend_handles:
        lh.set_alpha(1)
    plt.grid(True)
    plt.tight_layout()
    if save_dir:
        save_path = os.path.join(save_dir, f"loss_vs_iteration_{tau_str}.pdf")
        plt.savefig(save_path, dpi=300)
        print(f"Saved loss plot to {save_path}")


def plot_final_loss_vs_iteration(
    taus: List[float],
    tau_trajectories: List[List[Trajectory]],
    colors: List[str],
    args,
    save_dir: str = "./figures"
):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    end_risks = []
    for trajectories in tau_trajectories:
        end_risks_tau = []
        for trajectory in trajectories:
            end_risks_tau.append(trajectory.risks_tt[-1])
        end_risks.append(end_risks_tau)

    x_indices = np.arange(len(taus))

    plt.figure(figsize=(7, 4))

    for i, index in enumerate(x_indices):
        y_values = end_risks[i]
        x_values = [index] * len(y_values)  # Repeat index for each end_risk value
        plt.scatter(x_values, y_values, marker='x', color=colors[i], alpha=min(1, 10. / len(trajectories)), s=50)
    plt.xticks(x_indices, taus, fontsize=14)
    plt.yticks(fontsize=14)
    plt.axhline(args.alpha, linestyle='--', color='#d62728', label=r'Upper Bound $\alpha$', linewidth=1.5)  # red
    plt.axhline(args.alpha - args.tightness, linestyle='--', color='#2ca02c', label=r'Lower Bound $\alpha - \Delta\alpha$', linewidth=1.5)  # green

    plt.xlabel(rf"$\tau$", fontsize=20)
    plt.ylabel("Risk", fontsize=20)

    ylim_lower = min(0.05 + min([min(risks) for risks in end_risks]), args.alpha - args.tightness * 1.5)
    ylim_upper = max(0.05 + max([max(risks) for risks in end_risks]), args.alpha + args.tightness * 0.5)
    plt.ylim(ylim_lower, ylim_upper)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_dir:
        save_path = os.path.join(save_dir, "final_loss_vs_tau.pdf")
        plt.savefig(save_path, dpi=300)
        print(f"Saved final loss plot to {save_path}")

    in_bounds = []
    out_of_bounds = []
    for end_risks_tau in end_risks:
        in_bounds.append(0)
        out_of_bounds.append(0)
        for end_risk in end_risks_tau:
            if args.alpha - args.tightness <= end_risk <= args.alpha:
                in_bounds[-1] += 1
            else:
                out_of_bounds[-1] += 1

    total = [i + o for i, o in zip(in_bounds, out_of_bounds)]
    in_bounds_frac = [i / t if t > 0 else 0 for i, t in zip(in_bounds, total)]
    out_of_bounds_frac = [o / t if t > 0 else 0 for o, t in zip(out_of_bounds, total)]

    n_groups = len(taus)

    fig, ax = plt.subplots(figsize=(7, 4))

    bar_width = 0.35
    index = np.arange(n_groups)

    # Plot bars
    rects1 = ax.bar(index, in_bounds_frac, bar_width,
                    label='In Bounds', color='#2ca02c')

    rects2 = ax.bar(index + bar_width, out_of_bounds_frac, bar_width,
                    label='Out of Bounds', color='#d62728')

    ax.axhline(args.delta, linestyle='--', color='#1f77b4', label=r'Failure Probability $\delta$', linewidth=1.5)

    # Add labels, title, and ticks
    ax.set_xlabel(rf"$\tau$", fontsize=20)
    ax.set_ylabel('Relative Frequency', fontsize=20)
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(taus)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.legend()

    plt.tight_layout()
    if save_dir:
        save_path = os.path.join(save_dir, "failure_prob_vs_tau.pdf")
        plt.savefig(save_path, dpi=300)
        print(f"Saved final loss plot to {save_path}")
