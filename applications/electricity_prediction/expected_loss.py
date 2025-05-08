from typing import List, Union

import numpy as np

from rcpp.width_calculator import CLTWidth
from rcpp.risk_measure import MeanRiskMeasure
from rcpp.performativity_simulator import PerformativitySimulator
from rcpp.loss_simulator import LossSimulator
from rcpp.main import run_experiment


class ElectricityShiftSimulator(PerformativitySimulator):

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
        # TODO ignore Z_base, ignore lambda_, ignore gamma; shift `Z_prev`
        assert len(Z_prev) == 2
        X_prev, Y_prev = Z_prev
        # X_shifted, Y_shifted = shift(Z_prev)
        # return [X_shifted, Y_shifted]
        pass


class ElectricityLossSimulator(LossSimulator):

    def __init__(self, classifier_model):
        self.classifier_model = classifier_model

    def calc_loss(self, Z: List[np.ndarray], lambda_: float, do_new_sample: bool = True) -> np.ndarray:
        assert len(Z) == 2
        X, Y = Z
        Y_hat = self.classifier_model(X)[:, 1]

        loss = np.where(
            (0.5 - lambda_ <= Y_hat) & (Y_hat <= 0.5 + lambda_),  # abstain from prediction
            0,
            np.where(
                Y_hat < 0.5 - lambda_,
                np.where(Y == 1, 1, 0),  # Predict 0, loss if Y == 1
                np.where(Y == 0, 1, 0)  # Predict 1, loss if Y == 0
            )
        )
        return loss


class Args:
    def __init__(self):
        self.alpha = 0.3         # risk control level
        self.tightness = 0.08    # tightness parameter, may throw error if too low
        self.delta = 0.1         # failure probability or confidence parameter
        self.tau = 1.0           # safety parameter
        self.N = 2000            # number of samples in cohort
        self.lambda_max = 1.0    # maximum value for lambda
        self.ell_max = 0.5
        self.gamma = 0.1         # parameter in the Wasserstein distance formulation, for simulating distribution shift


if __name__ == "__main__":

    np.random.seed(42)

    # TODO: Load data
    X_all, Y_all = None, None

    # TODO: Define model
    llm_model = None  # model that does shifting
    classifier_model = None   # model that does classification

    # Run experiment
    args = Args()
    width_calculator = CLTWidth(args.alpha, args.ell_max, tol=1e-5)
    risk_measure = MeanRiskMeasure()
    performativity_simulator = ElectricityShiftSimulator(model=llm_model)
    loss_simulator = ElectricityLossSimulator(classifier_model)
    run_experiment(
        Z=[X_all, Y_all],
        width_calculator=width_calculator,
        risk_measure=risk_measure,
        performativity_simulator=performativity_simulator,
        loss_simulator=loss_simulator,
        args=args,
        gammas=[1.0],
        save_dir=f"./applications/electricity_prediction/figures/expected_loss/",
        num_iters=10  # TODO: set to 1000 for real experiments, 10 for testing/debugging
    )
