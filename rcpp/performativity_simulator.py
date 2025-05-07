from abc import ABC, abstractmethod
from typing import List

import numpy as np


class PerformativitySimulator(ABC):
    @abstractmethod
    def simulate_shift(self, Z: List[np.ndarray], lambda_: float) -> List[np.ndarray]:
        """
        Given a deployment threshold `lambda_`, shifts the data `Z`and returns 
        the shifted data in the same order.

        Arguments:
        - Z: a list of np.ndarray's, representing the data to be shifted.
        - lambda_: the deployment threshold

        Returns:
        - a list of np.ndarray's, representing the shifted data.
        """
        pass

