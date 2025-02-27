

# f(theta^T x)
# Utility: u(x, lambda) = softmax(gamma * (1 - lambda  - f(theta^T x)  ))
from typing import Union

import numpy as np
from sklearn.linear_model import LogisticRegression

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def type_II_error(Y, Y_proba, threshold):
    """Y=actual, Y_proba=predicted probability, threshold=threshold"""
    return np.mean((Y == 1) * (Y_proba < 1 - threshold))

def type_II_error_approx(Y, Y_proba, threshold, softness):
    return np.mean((Y == 1) * (sigmoid(softness * (1 - threshold - Y_proba))))


class UCBBound:

    def __init__(self, alpha, tau, eps, L, n, delta, t):
        self.alpha = alpha
        self.tau = tau
        self.eps = eps
        self.L = L
        self.n = n
        self.delta = delta
        self.t = t

    def bound(self):
        pass


class HoeffdingBound(UCBBound):

    def __init__(self, bounds=(0, 1), **kwargs,):
        super().__init__(**kwargs)
        self.lower, self.upper = bounds

    def bound(self):
        return self.alpha - 16 / (self.tau - self.eps * self.L) * (
            (self.upper - self.lower) * np.sqrt(
                1. / 2 / self.n * np.log(1. / self.delta)
            )
        )

# class WSRBound(Bound):

#     def __init__(self, n, delta, bounds=(0, 1)):
#         super().__init__(n, delta)
#         self.lower, self.upper = bounds

#     def bound(self):
#         return np.sqrt(1 / 2 / self.n * np.log(1 / self.delta))
    

class ThreshUpdater:
    """Abstract class for threshold update."""
    def get_thresh_predeploy(self, Y_cal, Y_cal_proba):
        raise NotImplementedError

    def update_postdeploy(self, loss_violation: bool):
        return
    


class SoftLossThreshUpdater(ThreshUpdater):
    """Threshold update based on soft loss."""


    def __init__(self, loss_softness: float, tau: float, alpha: float, bound: UCBBound,
                 thresh_lower_bound=0, thresh_upper_bound=1):
        self.loss_softness = loss_softness
        self.tau = tau
        self.alpha = alpha
        self.bound = bound
        self.thresh_lower_bound = thresh_lower_bound
        self.thresh_upper_bound = thresh_upper_bound

    def get_thresh_predeploy(self, Y_cal, Y_cal_proba, error_bound=1e-6):
        """Updates the threshold. Assumes constant n throughout time.
        """
        n = len(Y_cal)

        partial_loss = lambda threshold: type_II_error_approx(Y_cal, Y_cal_proba, threshold, self.loss_softness) - self.tau * threshold

        # determine min threshold at which risk is controlled
        low, high = self.thresh_lower_bound, self.thresh_upper_bound
        while high - low > error_bound:
            mid = (low + high) / 2
            if partial_loss(mid) > bound:
                low = mid
            else:
                high = mid
        return (low + high) / 2


class BinaryThreshUpdater(ThreshUpdater):
    
    def __init__(self, thresh_lower_bound, thresh_upper_bound):
        self.thresh_lower_bound = thresh_lower_bound
        self.thresh_upper_bound = thresh_upper_bound
    
    def get_thresh_predeploy(self, Y_cal, Y_cal_proba):
        return (self.thresh_lower_bound + self.thresh_upper_bound) / 2

    def update_postdeploy(self, loss_violation: bool):
        mid = (self.thresh_lower_bound + self.thresh_upper_bound) / 2
        if loss_violation:
            self.thresh_lower_bound = mid
        else:
            self.thresh_upper_bound = mid


class FeaturesUpdater:
    """Abstract class for updating features based on model and strategic features."""
    def __init__(self, model: LogisticRegression, strat_features: Union[np.ndarray, None], eps: float):
        self.model_params = model.coef_[0]
        if strat_features is not None:
            self.strat_coef = np.zeros((1, len(model.coef_[0])))
            self.strat_coef[0, strat_features] = model.coef_[0, strat_features]
        else:
            self.strat_coef = model.coef_[0]
        self.eps = eps

    def best_response(self, X, Y_proba, thresh):
        raise NotImplementedError


class SoftUtilityFeaturesUpdater(FeaturesUpdater):
    """Models the distribution shift of the features. Assumes access to model.
    
    Assumes a sigmoidal utility function (scaled + stretched) and a quadratic cost."""

    def __init__(self, model: LogisticRegression, strat_features: np.ndarray, eps: float, utility_softness: float):
        super().__init__(model, strat_features, eps)
        self.utility_softness = utility_softness

    def _get_C_gamma(self, gamma):
        """Find the largest abs value of the second derivative of the scaled sigmoid"""
        from scipy.optimize import minimize_scalar

        def second_deriv_scaled_sigmoid(x, gamma):
            return 2*(gamma**2)*np.exp(-2* gamma*x) / (np.exp(-gamma*x)+1)**3 \
                - (gamma**2)*np.exp(-gamma*x) / (np.exp(-gamma*x)+1)**2

        # function is odd and negative on [0, 100]
        result = minimize_scalar(lambda x: second_deriv_scaled_sigmoid(x, gamma), bounds=(0, 100), method='bounded')

        if result.x < 1e-5 or result.x > 100 - 1e-5:
            raise ValueError('gamma minimization outside of range')

        return -second_deriv_scaled_sigmoid(result.x, gamma).item()

    def _calc_g_eps(self):
        """Compute quadratic cost coefficient based on desired level of epsilon-sensitivity."""
        return 4 * self.eps / self.utility_softness / np.linalg.norm(self.strat_coef).item() \
            / self._get_C_gamma(self.utility_softness)

    def best_response(self, X, Y_proba, thresh):
        """Best response function for agents, linear utilities quadratic costs

        # Assume very utility function is soft threshold
        # Calculate the best response by computing the derivative of 
        # the utility minus the cost
        """
        # apply chain rule
        soft_indicator_deriv = sigmoid(self.utility_softness * (1 - thresh - Y_proba))
        soft_indicator_deriv *= (1 - soft_indicator_deriv)  # soft indicator fn

        proba_deriv = - self.utility_softness * Y_proba * (1 - Y_proba)  # predictor

        deriv_scale = (soft_indicator_deriv * proba_deriv).reshape(-1, 1)  # based on threshold

        X_strat = X + self._calc_g_eps() * deriv_scale * self.strat_coef
        # X_strat = X + self.eps * deriv_scale * strat_coef
        # X_strat = X + self.eps * strat_coef
        # print("Ratio eps / g(eps): ", self.eps / self._calc_g_eps())

        return X_strat


class NoCostFeaturesUpdater(FeaturesUpdater):
    """Update the most sensitive feature."""

    def __init__(self, model: LogisticRegression, strat_features: Union[np.ndarray,None], eps: float):
        super().__init__(model, strat_features, eps)

        sensitive_idx = self.strat_coef[0].argmax()
        self.onehot = np.zeros((1, len(model.coef_[0])))
        self.onehot[0, sensitive_idx] = 1

    # ||x(lambda) - x(lambda')|| <= eps ||lambda - lambda'|| <= 0.5 * (0.05) = 0.025

    def best_response(self, X, Y_proba, thresh):
        X_start = X + self.eps * thresh * self.onehot
        return X_start

