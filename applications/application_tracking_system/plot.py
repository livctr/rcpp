import numpy as np
from copy import deepcopy
from tqdm import tqdm

import rcpp
from rcpp.main import plot_lambda_vs_iteration, plot_loss_vs_iteration, plot_final_loss_vs_iteration
import pickle

def plot_ats_experiment(
    args,
    save_dir: str = "./figures",
):
    """
    Run the RCPP algorithm and plot the results.
    """

    def cut(data):
        tot = len(data[0])
        assert tot > args.N
        idx = np.random.choice(tot, size=args.N, replace=False)
        return idx

    with open("credit_data.pkl", "rb") as f:
        data = pickle.load(f)
    

    colors = ['blue', 'green', 'red', 'orange', 'black', 'purple', 'brown']
    taus = [args.tau]
    with open("./applications/application_tracking_system/figures/expected_loss/trajectories.pkl", "rb") as f:
        tau_trajectories = [pickle.load(f)]

    # Plot 1/2
    plot_lambda_vs_iteration(taus, tau_trajectories, colors, args, save_dir=save_dir)

    # Plot 3
    for i, tau in enumerate(taus):
        plot_loss_vs_iteration(tau, tau_trajectories[i], colors[i], args, save_dir=save_dir)
    
    # Plot 4
    plot_final_loss_vs_iteration(taus, tau_trajectories, colors, args, save_dir=save_dir)

    return tau_trajectories


# DANGER DANGER: COPIED DIRECTLY FROM main.py
class Args:
    def __init__(self):
        self.alpha = 0.3         # risk control level
        self.tightness = 0.082    # tightness parameter, may throw error if too low
        self.delta = 0.1        # failure probability or confidence parameter
        self.tau = 1.0           # safety parameter
        self.N = 2000            # number of samples in cohort
        self.lambda_min = 0.0
        self.lambda_safe = 1.0 + 1e-4   # maximum value for lambda
        self.ell_max = 1.0
        self.gamma = 0.0


if __name__ == "__main__":
    plot_ats_experiment(Args(), save_dir="./applications/application_tracking_system/figures")
