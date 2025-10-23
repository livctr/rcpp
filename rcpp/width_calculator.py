from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import norm



class WidthCalculator(ABC):
    @abstractmethod
    def get_width(self, delta_prime: float, N: int):
        """Returns the width depending on the number of iterations required."""
        pass


class CLTWidth(WidthCalculator):
    """
    Computes the width of the confidence interval using the Central
    Limit Theorem (CLT) for a given empirical risk. Does not assume
    anything about the loss distribution except for boundedness.
    """
    def __init__(self, alpha: float, loss_max: float = 1.0, tol=1e-5):
        self.alpha = alpha
        self.loss_max = loss_max
        self.tol = tol

    def _calc_clt_bound(self, delta_prime, N, emp_risk, loss_max = 1):
        """
        Calculate confidence width based on the maximum variance possible if the
        risk level is `emp_risk`.
        """
        p = emp_risk  / loss_max
        max_var = 1 / (N - 1) * (p * N * (loss_max - emp_risk) ** 2 + (1 - p) * N * emp_risk ** 2)
        return norm.ppf(1 - delta_prime / 2) * np.sqrt(max_var) / np.sqrt(N)

    def get_width(self, delta_prime, N):
        """
        Given `alpha`, calculates the confidence width necessary to ensure risk
        control throughout all deployments of lambda.
        """
        # Aggregate `_clt_bound` over possible emp_risks.
        # Since _clt_bound is monotonically increasing in `emp_risk` until p=0.5
        # Then it decreases
        max_clt_bound = self._calc_clt_bound(delta_prime, N, 0.5 * self.loss_max, self.loss_max)
        # The `max_clt_bound` is already sufficient
        if 0.5 * self.loss_max + max_clt_bound < self.alpha:
            return max_clt_bound

        low_emp_risk, high_emp_risk = 0, self.loss_max * 0.5

        while low_emp_risk < high_emp_risk - self.tol:
            mid_emp_risk = (low_emp_risk + high_emp_risk) / 2
            confidence_width = self._calc_clt_bound(delta_prime, N, mid_emp_risk, self.loss_max)
            ucb = mid_emp_risk + confidence_width
            if ucb > self.alpha:
                high_emp_risk = mid_emp_risk
            else:
                low_emp_risk = mid_emp_risk
        return confidence_width


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
