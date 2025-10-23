from abc import ABC, abstractmethod
import numpy as np


class RiskMeasure(ABC):
    @abstractmethod
    def calculate(self, losses: np.ndarray) -> float:
        """
        Given a numpy array of shape (N,), calculates the risk measure associated
        with that loss.
        """
        pass


class MeanRiskMeasure(RiskMeasure):
    def calculate(self, losses: np.ndarray) -> float:
        """Calculates the mean losses."""
        return np.nanmean(losses)


class CVaRRiskMeasure(RiskMeasure):

    def __init__(self, beta: float):
        self.beta = beta

    def calculate(self, losses: np.ndarray) -> float:
        """Returns the beta-CVaR risk given an array of losses."""
        sorted_losses = np.sort(losses)

        # Calculate \int_{i/N}^{(i+1)/N} \psi(p) dp
        integral_bounds = np.linspace(0, 1, len(losses) + 1)
        lower_bound = integral_bounds[:-1]
        upper_bound = integral_bounds[1:]
        integral_bounds = np.column_stack((lower_bound, upper_bound))

        is_left = integral_bounds[:, 1] <= self.beta
        is_right = integral_bounds[:, 0] > self.beta

        weights = np.where(
            is_left,
            0.0,
            np.where(
                is_right,
                1.0 / (1.0 - self.beta) * (1. / len(losses)),  # height x width
                1.0 / (1.0 - self.beta) * (integral_bounds[:, 1] - self.beta),
            ),
        )

        # Returns \int_0^1 \psi(p) \hat{F}^{-1}(p) dp; here, F is the loss CDF
        return np.sum(sorted_losses * weights)
