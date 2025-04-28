from sklearn.linear_model import LogisticRegression
import scipy.stats as stats
import numpy as np

def train_logistic_regression(X, Y, **kwargs):
    # fit_intercept=False since X already has bias term
    model = LogisticRegression(fit_intercept=False, **kwargs)  # intercept pre-built in X
    model.fit(X, Y)
    assert model.classes_[0] == 0 and model.classes_[1] == 1
    return model

def hoeffding_bound(T, n, delta, alpha, ell_max=1):

    # delta_prime = (args.delta * args.tau * epsilon) / (T * args.alpha)
    delta_prime = delta / T
    return ell_max * np.sqrt((1 / (2 * n)) * np.log(2 / delta_prime))

# Hoeffding-benktus bound

def _calc_hoeffding_bentkus_p_value(emp_risk, alpha, N):
    a = min(emp_risk, alpha)
    b = alpha
    # print("emp_risk, alpha: ", emp_risk, alpha)
    entropy = a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))
    left = np.exp(-N * entropy)
    right = np.exp(1) * stats.binom.cdf(np.ceil(emp_risk * N), N, alpha)
    return min(left, right)

def _get_hoeffding_bentkus_is_width_workable(c, emp_risk, delta_prime, N):
    upper_p_value = _calc_hoeffding_bentkus_p_value(emp_risk, emp_risk + c, N)
    lower_p_value = _calc_hoeffding_bentkus_p_value(1 - emp_risk, 1 - emp_risk + c, N)
    return lower_p_value + upper_p_value <= delta_prime

def hoeffding_bentkus_bound(T, n, delta, alpha, loss_max = 1, tol=1e-5):
    assert loss_max == 1

    delta_prime = delta/T
    low_emp_risk, high_emp_risk = tol, alpha
    while low_emp_risk < high_emp_risk - tol:
        mid_emp_risk = (low_emp_risk + high_emp_risk) / 2
        # print(mid_emp_risk)
        if _get_hoeffding_bentkus_is_width_workable(alpha - mid_emp_risk, mid_emp_risk, delta_prime, n):
            low_emp_risk = mid_emp_risk
        else:
            high_emp_risk = mid_emp_risk
    return alpha - low_emp_risk

def calc_clt_bound(T, delta_prime, N, emp_risk, loss_max = 1):
    """
    Calculate confidence width based on the maximum variance possible if the
    risk level is `emp_risk`.
    """
    p = emp_risk  / loss_max
    max_var = 1 / (N - 1) * (p * N * (loss_max - emp_risk) ** 2 + (1 - p) * N * emp_risk ** 2)
    return stats.norm.ppf(1 - delta_prime / 2) * np.sqrt(max_var) / np.sqrt(N)

def clt_bound(T, n, delta, alpha, loss_max = 1, tol=1e-5):
    """
    Given `alpha`, calculates the confidence width necessary to ensure risk
    control throughout all deployments of lambda.
    """
    # Aggregate `_clt_bound` over possible emp_risks.
    # Since _clt_bound is monotonically increasing in `emp_risk` until p=0.5
    # Then it decreases
    delta_prime = delta/T
    max_clt_bound = calc_clt_bound(T, delta_prime, n, 0.5 * loss_max, loss_max)
    # The `max_clt_bound` is already sufficient
    if 0.5 * loss_max + max_clt_bound < alpha:
        return max_clt_bound

    low_emp_risk, high_emp_risk = 0, loss_max * 0.5

    while low_emp_risk < high_emp_risk - tol:
        mid_emp_risk = (low_emp_risk + high_emp_risk) / 2
        confidence_width = calc_clt_bound(T, delta_prime, n, mid_emp_risk, loss_max)
        ucb = mid_emp_risk + confidence_width
        if ucb > alpha:
            high_emp_risk = mid_emp_risk
        else:
            low_emp_risk = mid_emp_risk
    return confidence_width

# Search for optimal T
def find_optimal_T(bound_type, n, alpha, tightness, delta, tau, max_T=1000, lambda_max=1):
    best_T = None
    best_result = None

    for T in range(1, max_T + 1):
        if bound_type == "hoeffding":
            c_delta = hoeffding_bound(T, n, delta, alpha)
        elif bound_type == "hoeffding_bentkus":
            c_delta = hoeffding_bentkus_bound(T, n, delta, alpha)
        elif bound_type == "clt":
            c_delta = clt_bound(T, n, delta, alpha, loss_max = 1, tol=1e-5)
        delta_lambda = (tightness - 2 * c_delta) / (2 * tau)

        if delta_lambda <= 0 or delta_lambda >= 1:
            continue  # Invalid Delta_lambda
        #print(delta_lambda * T)

        if T >= lambda_max / delta_lambda:
            best_T = T
            break  # Choose the one with minimum T

    return best_T, delta_lambda

def type_II_error(Y, Y_proba, threshold):
    """Y=actual, Y_proba=predicted probability, threshold=threshold"""
    return np.mean((Y == 1) * (Y_proba < 1. - threshold))

def modify(Y_proba, threshold, gamma, M):
    shift = gamma / M
    condition = (1 - threshold) < (Y_proba - shift)

    Y_proba_new = Y_proba.copy()
    Y_proba_new[~condition] -= shift
    Y_proba_new = np.clip(Y_proba_new, 0, 1)

    return Y_proba_new

def binary_search_solver(objective, lam_low, lam_high, tol=1e-6, max_iter=100):
    if objective(lam_low) <= 0:
        return lam_low
    
    if objective(lam_high) > 0:
        raise ValueError("No crossing found: objective > 0 for all λ in [lam_low, lam_high].")

    for _ in range(max_iter):
        mid = (lam_low + lam_high) / 2.0
        if objective(mid) <= 0:
            lam_high = mid
        else:
            lam_low = mid
        if lam_high - lam_low < tol:
            break
    return lam_high

def grid_search_solver(objective, lam_low, lam_high, num_points=100):
    lambdas = np.linspace(lam_low, lam_high, num_points)
    for lam in lambdas:
        if objective(lam) <= 0:
            return lam
    raise ValueError("No crossing found: objective > 0 for all λ in [lam_low, lam_high].")

def run_trajectory(Y, Y_proba, n, alpha, tightness, delta, tau, gamma, M, bound_type, verbose=False):
    # Precompute bound
    if bound_type == "hoeffding":
        T, delta_lambda = find_optimal_T(bound_type, n, alpha, tightness, delta, tau)
        bound = hoeffding_bound(T, n, delta, alpha)
    elif bound_type == "hoeffding_bentkus":
        T, delta_lambda = find_optimal_T(bound_type, n, alpha, tightness, delta, tau)
        bound = hoeffding_bentkus_bound(T, n, delta, alpha)
    elif bound_type == "clt":
        T, delta_lambda = find_optimal_T(bound_type, n, alpha, tightness, delta, tau)
        bound = clt_bound(T, n, delta, alpha, loss_max = 1, tol=1e-5)
    elif bound_type == "bernstein":
        T, delta_lambda = find_optimal_T(bound_type, n, alpha, tightness, delta, tau)
        bound = bernstein_bound(T, n)
    else:
        raise ValueError(f"Unknown bound_type: {bound_type}")
     
    # Initialization
    thresh = 1 # lambda_max = 1
    threshes = [thresh]

    # Metrics
    # loss_at_lambda_t = []
    err_at_lambda_t = []
    # loss_at_lambda_tp1 = []
    err_at_lambda_tp1 = []

    iters = tqdm(range(1, T + 1)) if verbose else range(1, T + 1)
    # args.N = 1000 Number of samples per iteration

    ### === NEW PART: prepare global indices without replacement ===
    total_size = len(Y)
    #assert n_sample * args.T <= total_size, "Not enough samples for no-replacement sampling! Reduce T or sample size."

    all_indices = np.random.permutation(total_size)  # Shuffle once
    pointer = 0  # Pointer to track where we are

    for t in iters:
        # Deploy threshold
        Y_proba_t = modify(Y_proba, thresh, gamma, M)

        # === Take a fresh batch without global repeats ===
        idx = all_indices[pointer: pointer + n]
        pointer += n  # Move pointer for next iteration

        Y_sample = Y[idx]
        Y_proba_sample = Y_proba_t[idx]

        # Calculate loss and error
        # loss_t = float(np.mean(piecewise_loss(Y_sample, Y_proba_sample, thresh)))
        err_t = float(type_II_error(Y_sample, Y_proba_sample, thresh))
        # loss_at_lambda_t.append(loss_t)
        err_at_lambda_t.append(err_t)

        # Update threshold
        func = lambda nt: np.mean(type_II_error(Y_sample, Y_proba_sample, nt)) + tau * (thresh - nt) + bound - alpha
        mid_thresh = binary_search_solver(func, 0, 1)
        new_thresh = min(mid_thresh, thresh)

        # Evaluate new threshold
        # loss_tp1 = float(np.mean(piecewise_loss(Y_sample, Y_proba_sample, new_thresh)))
        err_tp1 = float(type_II_error(Y_sample, Y_proba_sample, new_thresh))
        # loss_at_lambda_tp1.append(loss_tp1)
        err_at_lambda_tp1.append(err_tp1)

        if new_thresh > thresh - delta_lambda:
            threshes.append(new_thresh)
            stopping_iter = t
            break

        thresh = new_thresh
        threshes.append(thresh)

    return {
        "Stopping iteration": stopping_iter,
        #"loss_at_lambda_t": loss_at_lambda_t,
        "err_at_lambda_t": err_at_lambda_t,
        #"loss_at_lambda_tp1": loss_at_lambda_tp1,
        "err_at_lambda_tp1": err_at_lambda_tp1,
        "threshes": threshes
    }

def evaluate_prc_over_seeds(X_cal, Y_cal, X_test, Y_test, model, n, alpha, tightness, delta, tau, gamma, M, bound_type, num_runs=1000):
    #all_test_losses = []
    all_test_errors = []
    all_final_thresholds = []
    all_stopping_iters = []

    # Step 1: Split calibration/test
    #X_cal, X_test, Y_cal, Y_test = train_test_split(
    #        X_temp, Y_temp, train_size=10000, random_state=42
    #    )

    for seed in range(num_runs):
        # Step 2: Predict probabilities
        Y_proba_cal = model.predict_proba(X_cal)[:, 1]
        Y_proba_test = model.predict_proba(X_test)[:, 1]

        # Step 3: Run PRC
        results_3 = run_trajectory(Y_cal, Y_proba_cal, n, alpha, tightness, delta, tau, gamma, M, bound_type)
        final_thresh = results_3["threshes"][-1]
        stopping_iter = results_3["Stopping iteration"]
        all_stopping_iters.append(stopping_iter)
        
        # Step 4: Evaluate on test set (simulate D(λ_T) using final_thresh) 
        Y_proba_test_mod = modify(Y_proba_test, final_thresh, gamma, M)

        # Step 5: Evaluate test loss and error under D(λ_T)
       # test_loss = float(np.mean(piecewise_loss(Y_test, Y_proba_test_mod, final_thresh)))
        test_error = float(type_II_error(Y_test, Y_proba_test_mod, final_thresh))

        # Step 6: Store results
        #all_test_losses.append(test_loss)
        all_test_errors.append(test_error)
        all_final_thresholds.append(final_thresh)

    # Compute statistics and bound
    #losses = np.array(all_test_losses)
    errors = np.array(all_test_errors)

    return {
    #"losses": losses,
    "errors": errors,
    "thresholds": all_final_thresholds,
    "stopping_iters": all_stopping_iters
}