from abc import ABC, abstractmethod
from typing import List, Union


class PerformativitySimulator(ABC):

    def reset(self) -> None:
        """
        Resets the simulator to its initial state.
        """
        return

    @abstractmethod
    def simulate_shift(self,
                       Z_base: Union[List, None],
                       lambda_: float) -> List:
        """
        Given a deployment threshold `lambda_` and the shift magnitude `gamma`, shifts
        the data from `Z_base` to produce a shifted `Z`.

        Arguments:
        - Z_base: a list of np.ndarray's, representing the data from the base distribution.
        - lambda_: the deployment threshold

        Returns:
        - a list of np.ndarray's, representing the shifted data.
        """
        pass
