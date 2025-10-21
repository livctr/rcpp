from abc import ABC
from typing import List

import numpy as np


class LossSimulator(ABC):

    def calc_loss(self, Z: List[np.ndarray], lambda_: float, do_new_sample: bool = True) -> np.ndarray:
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
        pass

class ZeroOneLossSimulator(LossSimulator):

    def calc_loss(self, Z: List[np.ndarray], lambda_: float, do_new_sample: bool = True, epsilon=1e-4) -> np.ndarray:
        assert len(Z) == 2
        Y_hat, Y = Z
        cont_indicator = np.clip( (1 - lambda_ + epsilon - Y_hat) / (2 * epsilon), 0, 1)
        return cont_indicator * (Y == 1)
  

class ZeroUniformLossSimulator(LossSimulator):

    def __init__(self):
        self.random_losses = None

    def _get_random_losses(self, N, do_new_sample: bool = True):
        if do_new_sample or self.random_losses is None:
            self.random_losses = np.random.uniform(0, 1, N)
        return self.random_losses

    def calc_loss(self, Z: List[np.ndarray], lambda_: float, do_new_sample: bool = True, epsilon=1e-4) -> np.ndarray:
        assert len(Z) == 2
        Y_hat, Y = Z
        random_losses = self._get_random_losses(len(Y), do_new_sample)
        cont_indicator = np.clip( (1 - lambda_ + epsilon - Y_hat) / (2 * epsilon), 0, 1)
        return random_losses * (Y == 1) * cont_indicator
