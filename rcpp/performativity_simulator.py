from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np


class PerformativitySimulator(ABC):
    @abstractmethod
    def simulate_shift(self,
                       Z_base: Union[List[np.ndarray], None],
                       Z_prev: Union[List[np.ndarray], None],
                       lambda_: float,
                       gamma: float) -> List[np.ndarray]:
        """
        Given a deployment threshold `lambda_` and the shift magnitude `gamma`, shifts
        the data (either from `Z_base` or `Z_prev`) to produce a shifted `Z`.

        Arguments:
        - Z_base: a list of np.ndarray's, representing the data from the base distribution.
        - Z_prev: a list of np.ndarray's, representing the data from the previous deployment
        - lambda_: the deployment threshold
        - gamma: the parameter for the Wasserstein distance formulation, for simulating distribution shift.
            The larger the gamma, the more the distribution shift.

        Returns:
        - a list of np.ndarray's, representing the shifted data.
        """
        pass
