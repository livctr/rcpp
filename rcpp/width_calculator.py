from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import norm


class WidthCalculator(ABC):
    @abstractmethod
    def get_width(self, delta_prime: float, N: int):
        """Returns the width depending on the number of iterations required."""
        pass


class CLTQuantileVarianceWidth(WidthCalculator):
    """
    Computes the variance used in the quantile-based CLT confidence interval.
    Assumes knowledge of the base rate `p` of non-creditworthy individuals.
    """
    def __init__(self, beta, p):
        """
        Arguments:
        - beta: the beta in beta-CVaR, e.g., 0.9 for 90%
        - p: base rate of non-creditworthy individuals
        """
        self.beta = beta
        self.p = p

        self.clt_variance = None

    def _quantile_clt_variance(self):
        """
        Returns the variance used in the quantile-based CLT, see appendix
        for derivation.
        """
        if self.beta <= 1 - self.p:
            return 1 / ((1 - self.beta)**2) * (1. / 12) * (4 - 3*self.p) * self.p
        else:
            return (1 - self.beta) * (3 * self.beta + 1) / (12 * self.p**2)

    def quantile_clt_variance(self):
        if self.clt_variance is None:
            self.clt_variance = self._quantile_clt_variance()
        return self.clt_variance

    def get_width(self, delta_prime: float, N: int):
        """
        Returns the confidence width of the quantile-based CLT confidence interval. 
        """
        sigma_sq = self.quantile_clt_variance()
        return norm.ppf(1 - delta_prime / 2) * np.sqrt(sigma_sq / N)
