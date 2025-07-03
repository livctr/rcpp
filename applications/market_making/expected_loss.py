import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

from rcpp.width_calculator import CLTWidth, WidthCalculator
from rcpp.risk_measure import MeanRiskMeasure, RiskMeasure
from rcpp.performativity_simulator import PerformativitySimulator
from rcpp.loss_simulator import ZeroOneLossSimulator, LossSimulator

import pandas as pd
from scipy.stats import norm

from rcpp.main import find_guaranteed_T, binary_search_solver, Trajectory


def compute_loss(df: pd.DataFrame, lambda_: float, RHO: float, SIGMA: float, filter_nans: bool = True) -> np.ndarray:
    """Given a DataFrame with 'Close' prices, compute the profit."""

    # Compute target: normalize current price to 100k, then scale next_close accordingly
    target = (df["Close"].shift(-1) / df["Close"]) * 100000

    # Compute prediction: average of current and previous 4 closes. Assume simple mean model.
    pred = df["Close"].rolling(window=5).mean() / df["Close"] * 100000

    bid = pred - RHO * lambda_ / 2
    ask = pred + RHO * lambda_ / 2

    # Assume investor belief is a normal distribution with mean df['target'] and std SIGMA
    volume_on_bid = norm.cdf(bid, loc=target, scale=SIGMA)
    volume_on_ask = 1 - norm.cdf(ask, loc=target, scale=SIGMA)

    profit = volume_on_bid * (target - bid) + volume_on_ask * (ask - target)
    loss = - profit

    if filter_nans:
        loss = loss[~np.isnan(loss)]

    return np.array(loss)


class CLTMarketMakingWidth(WidthCalculator):
    def __init__(self, clt_var: float):
        self.clt_var = clt_var

    def get_width(self, delta_prime, N):
        return norm.ppf(1 - delta_prime / 2) * np.sqrt(self.clt_var) / np.sqrt(N)
    

class MarketMakingShiftSimulator:

    def __init__(self, rho: float = 100, sigma: float = 200):
        """
        rho: spread, assuming bitcoin is normalized to 100k
        sigma: std of price beliefs, assuming bitcoin is normalized
        """
        self.rho = rho
        self.sigma = sigma

    def reset(self):
        return

    def simulate_shift(self, df, lambda_,):
        target = (df["Close"].shift(-1) / df["Close"]) * 100000
        pred = df["Close"].rolling(window=5).mean() / df["Close"] * 100000

        if df.empty:
            raise ValueError("DataFrame is empty, cannot simulate shift.")
    
        bid = pred - self.rho * lambda_ / 2
        ask = pred + self.rho * lambda_ / 2

        # Assume investor belief is a normal distribution with mean target and std sigma
        volume_on_bid = norm.cdf(bid, loc=target, scale=self.sigma)
        volume_on_ask = 1 - norm.cdf(ask, loc=target, scale=self.sigma)

        return target, pred, volume_on_bid, volume_on_ask


class MarketMakingLossSimulator:

    def __init__(self, rho: float = 100):
        self.rho = rho

        self._volumes = None, None

    def calc_loss(self, Z, lambda_: float, do_new_sample: bool = True) -> np.ndarray:
        """
        Given a deployment threshold `lambda_`, calculates the loss of the data `Z`
        and returns the loss.

        Arguments:
        - Z: a list of np.ndarray's, representing the data.
        - lambda_: the deployment threshold
        - do_new_sample: since the loss is calculated at different thresholds during each round, 
            and we need to simulate loss stochasticity from round to round, this flag
            indicates between the two. For non-stochastic losses, this flag does nothing.

        Returns:
        - a np.ndarray of shape (N,), representing the loss.
        """
        assert len(Z) == 4
        target, pred, volume_on_bid, volume_on_ask = Z

        # [TODO]: Can make volume stochastic
        if do_new_sample or self._volumes is None:
            # Sample uniformly
            u = np.random.uniform(0, 1, len(target))
            stoch_volume_on_bid = (u < volume_on_bid).astype(float)
            stoch_volume_on_ask = (u < volume_on_ask).astype(float)
            self._volumes = stoch_volume_on_bid, stoch_volume_on_ask
        else:
            stoch_volume_on_bid, stoch_volume_on_ask = self._volumes

        counterfactual_bid = pred - self.rho * lambda_ / 2
        counterfactual_ask = pred + self.rho * lambda_ / 2

        profit = stoch_volume_on_bid * (target - counterfactual_bid) + stoch_volume_on_ask * (counterfactual_ask - target)
        loss = - profit
        return np.array(loss)


from datetime import datetime

def run_trajectory(
    df: pd.DataFrame,
    start_week: datetime,
    width_calculator: WidthCalculator,
    risk_measure: RiskMeasure,
    performativity_simulator: MarketMakingShiftSimulator,
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
        T, delta_lambda = -1, -1

    lambdas = [lambda_]
    risks_tt = []
    risks_tm1_t = []

    df_tm1 = df[(df['Timestamp'] >= start_week) & (df['Timestamp'] < start_week + pd.Timedelta(weeks=1))]
    week = start_week + pd.Timedelta(weeks=1)
    iter = 0

    # Week 1: deploy lambda. Z sim D(lambda) l(z, lambda)

    while True:

        # Get past week's data
        # print(f"Week {week.strftime('%Y-%m-%d')}, iter {iter}, lambda_ = {lambda_:.4f}")
        df_t = df[(df['Timestamp'] >= week) & (df['Timestamp'] < week + pd.Timedelta(weeks=1))]
        if df_t.empty:
            raise ValueError(f"No data available for week starting {week.strftime('%Y-%m-%d')}.")
    
        # Apply previous threshold lambda^{(t-1)} (on previous week's data)
        Z_tm1 = performativity_simulator.simulate_shift(df_tm1, lambda_)

        # [tracking] Calculate the realized loss L(lambda^{(t-1)}, lambda^{(t-1)})
        risks_tt.append(
            risk_measure.calculate(
                loss_simulator.calc_loss(Z_tm1, lambda_, do_new_sample=True)
            )
        )

        # 4. Use previous week's samples for calibration
        if control_risk:
            # 5. Find lambda^{(t)}_mid
            def loss_at_new_lambda(lambda_new):
                losses = loss_simulator.calc_loss(Z_tm1, lambda_new, do_new_sample=False)
                emp_risk = risk_measure.calculate(losses)
                return emp_risk + bound + args.tau * (lambda_ - lambda_new) - args.alpha
            loss_simulator.calc_loss(Z_tm1, lambda_, do_new_sample=True)  # Set the randomness
            lambda_mid = binary_search_solver(loss_at_new_lambda, 0, 1)

            # 6. Set new lambda^{(t)}
            lambda_new = min(lambda_, lambda_mid)
        else:
            lambda_new = 0.0
        lambdas.append(lambda_new)


        # [tracking] Calculate the "realized" loss L(lambda^{(t-1)}, lambda^{(t)})
        # Loss from the new market width w/o its effects on the volume
        risks_tm1_t.append(
            risk_measure.calculate(
                loss_simulator.calc_loss(Z_tm1, lambda_new, do_new_sample=True)
            )
        )

        # Update the data for the next iteration
        df_tm1 = df_t

        # 7-8. Stopping condition 
        if control_risk and lambda_new >= lambda_ - delta_lambda:
            lambda_hat = lambda_new
            # import pdb ; pdb.set_trace()
            break

        lambda_ = lambda_new

        iter += 1
        if not control_risk and iter >= num_iters:
            # import pdb ; pdb.set_trace()
            lambda_hat = 0.0
            break

    # Calculate the L(lambda_hat, lambda_hat)
    Z_shifted = performativity_simulator.simulate_shift(df_t, lambda_hat)
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


class Args:
    def __init__(self):
        self.alpha = 0.0         # risk control level
        self.tightness = 10.0    # tightness parameter, may throw error if too low
        self.delta = 0.1        # failure probability or confidence parameter
        self.tau = 1.0           # safety parameter
        self.N = 60 * 24 * 7 - 5            # number of samples in cohort
        self.lambda_min = 0.0
        self.lambda_safe = 1.0    # maximum value for lambda

        self.RHO = 100  # Spread, assuming bitcoin is normalized to 100k
        self.SIGMA = 200  # Std of price beliefs, assuming bitcoin is normalized

from copy import deepcopy
from tqdm import tqdm
from rcpp.main import plot_loss_vs_iteration, plot_final_loss_vs_iteration
from typing import List
import os


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


def run_mm_experiment(
    df,
    width_calculator,
    risk_measure,
    performativity_simulator,
    loss_simulator,
    args,
    taus = [1, 5, 10, 50, 1e2, 200],
    save_dir: str = "./figures",
    num_iters: int = 1000,
):
    """
    Run the RCPP algorithm and plot the results.
    """
    tau_trajectories = []
    for tau in taus:
        args_copy = deepcopy(args)
        args_copy.tau = tau
        trajectories = []
        for i in tqdm(range(num_iters), desc=f"Running trials for tau={tau}"):
            np.random.seed(i)
            trajectory = run_trajectory(
                df,
                start_week=datetime(2023, 2, 1),  # Start from the first week of February
                width_calculator=width_calculator,
                risk_measure=risk_measure,
                performativity_simulator=performativity_simulator,
                loss_simulator=loss_simulator,
                args=args,
                control_risk=True,  # Set to False if you want to run without risk control
                # num_iters=10  # Number of iterations to run
            )
            trajectories.append(trajectory)
        tau_trajectories.append(trajectories)
    
    colors = ['blue', 'green', 'red', 'orange', 'black', 'purple', 'brown']
    
    # Plot 1/2
    plot_lambda_vs_iteration(taus, tau_trajectories, colors, args, save_dir=save_dir)

    # Plot 3
    for i, tau in enumerate(taus):
        plot_loss_vs_iteration(tau, tau_trajectories[i], colors[i], args, save_dir=save_dir)
    
    # Plot 4
    plot_final_loss_vs_iteration(taus, tau_trajectories, colors, args, save_dir=save_dir)

    return tau_trajectories


if __name__ == "__main__":

    # We model Z = (target, pred, volume_on_bid, volume_on_ask)
    args = Args()
    np.random.seed(42)  # 123 0.47582531, 1234 0.48204899, 12345 0.47727013

    save_dir = "./applications/market_making/figures/expected_loss/"

    path_to_csv_file = './applications/market_making/data/btcusd_1-min_data_filtered.csv'
    df = pd.read_csv(path_to_csv_file)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')

    # Use simple average pricing model to predict next price
    target = (df["Close"].shift(-1) / df["Close"]) * 100000
    pred = df["Close"].rolling(window=5).mean() / df["Close"] * 100000

    # Find profit variance 

    # In month 1, we calibrate the CLT width by finding the maximum variance of the loss.
    # This occurs when the market is one-sided, i.e., when everyone hits the bid or lifts
    # the ask.
    clt_cal_start, clt_cal_end = "2023-01-01", "2023-01-31"
    df_clt_cal_index = df[(df['Timestamp'] >= clt_cal_start) & (df['Timestamp'] <= clt_cal_end)].index
    target_clt_cal = target[df_clt_cal_index]
    pred_clt_cal = pred[df_clt_cal_index]
    # import pdb ; pdb.set_trace()
    max_clt_var = np.var(target_clt_cal - pred_clt_cal, ddof=1)

    # Run experiment
    width_calculator = CLTMarketMakingWidth(clt_var=max_clt_var)
    risk_measure = MeanRiskMeasure()
    mms_simulator = MarketMakingShiftSimulator(rho=args.RHO, sigma=args.SIGMA)
    loss_simulator = MarketMakingLossSimulator(rho=args.RHO)

    run_mm_experiment(
        df,
        width_calculator=width_calculator,
        risk_measure=risk_measure,
        performativity_simulator=mms_simulator,
        loss_simulator=loss_simulator,
        args=args,
        taus=[1e-2, 5e-2, 1e-1, 5e-1, 1],
        save_dir=save_dir,
        num_iters=10
    )
