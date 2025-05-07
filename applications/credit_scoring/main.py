from typing import List

import numpy as np

from rcpp.performativity_simulator import PerformativitySimulator


class CreditScoringSimulator(PerformativitySimulator):
    def __init__(self, gamma, M, beta = 0.0):
        """
        Arguments:
        - gamma: paraeter in the Wasserstein distance formulation
        - M: bound on the base distribution PDF of the predictions Y_hat (not the loss)
        - beta: the beta in beta-CVaR, e.g., 0.9 for 90%
        """
        self.shift_size = gamma / M * (1 - beta)

    def simulate_shift(self, Z: List[np.ndarray], lambda_: float) -> List[np.ndarray]:
        assert len(Z) == 2
        Y_hat, Y = Z
        condition = (1 - lambda_) < (Y_hat - self.shift_size)
        Y_hat_new = Y_hat.copy()
        Y_hat_new[~condition] -= self.shift_size
        Y_hat_new = np.clip(Y_hat_new, 0, 1)
        return [Y_hat_new, Y]


