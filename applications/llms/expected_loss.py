from typing import List, Union

import numpy as np

from rcpp.width_calculator import CLTWidth
from rcpp.risk_measure import MeanRiskMeasure
from rcpp.performativity_simulator import PerformativitySimulator
from rcpp.loss_simulator import ZeroOneLossSimulator
from rcpp.main import run_experiment


# TODO: rename
class LLMsSimulator(PerformativitySimulator):

    def __init__(self, model):
        """
        Arguments:
        - M: bound on the base distribution PDF of the predictions Y_hat (not the loss)
        - beta: the beta in beta-CVaR, e.g., 0.9 for 90%
        """
        # TODO set params
        self.model = model
        pass

    def simulate_shift(self,
                       Z_base: Union[List[np.ndarray], None],
                       Z_prev: Union[List[np.ndarray], None],
                       lambda_: float,
                       gamma: float) -> List[np.ndarray]:
        # TODO
        pass


class Args:
    def __init__(self):
        self.alpha = 0.3         # risk control level
        self.tightness = 0.08    # tightness parameter, may throw error if too low
        self.delta = 0.1         # failure probability or confidence parameter
        self.tau = 1.0           # safety parameter
        self.N = 2000            # number of samples in cohort
        self.lambda_max = 1.0    # maximum value for lambda
        self.ell_max = 1.0
        self.gamma = 0.1         # parameter in the Wasserstein distance formulation, for simulating distribution shift


if __name__ == "__main__":

    np.random.seed(42)

    # TODO: Load data
    X_all, Y_all = None, None

    # TODO: Define model
    model = None

    # Run experiment
    args = Args()
    width_calculator = CLTWidth(args.alpha, args.ell_max, tol=1e-5)
    risk_measure = MeanRiskMeasure()
    performativity_simulator = LLMsSimulator(model=model)
    loss_simulator = ZeroOneLossSimulator()
    run_experiment(
        Z=[X_all, Y_all],
        width_calculator=width_calculator,
        risk_measure=risk_measure,
        performativity_simulator=performativity_simulator,
        loss_simulator=loss_simulator,
        args=args,
        gammas=[0.1, 0.2, 0.5, 1, 1.2, 1.5],
        # TODO: rename save directory
        save_dir=f"./applications/{TODO}/figures/expected_loss/",
        num_iters=1000  # TODO: set to 1000 for real experiments, 10 for testing/debugging
    )
