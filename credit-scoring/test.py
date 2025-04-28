import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

%matplotlib inline

from collections import Counter
from sklearn.model_selection import train_test_split
import scipy.stats as stats
from tqdm import tqdm
from data_prep import load_data
from model import train_logistic_regression, hoeffding_bound
from model import _calc_hoeffding_bentkus_p_value, _get_hoeffding_bentkus_is_width_workable, hoeffding_bentkus_bound
from model import calc_clt_bound, clt_bound, find_optimal_T
from model import type_II_error, modify, binary_search_solver, grid_search_solver
from model import run_trajectory, evaluate_prc_over_seeds

class args:
    alpha = 0.2         # risk control level
    tightness = 0.09    # tightness parameter, may throw error if too low
    delta = 0.1         # failure probability or confidence parameter

    tau = 1            # safety parameter
    # L = 5              # Lipschitz constant

    gamma = 0.2        

    N = 1000            # number of samples in cohort

    lambda_max = 1.0    # maximum value for lambda

    ell_max = 1.0

import os
os.chdir(r'C:\Users\chenbt\Downloads\rcpp-main\rcpp-main\GiveMeSomeCredit')

# Set random seed for reproducibility
np.random.seed(42)

# Load data
path_to_csv_file = './cs-training.csv'
X_all, Y_all, data = load_data(path_to_csv_file)

# Report full feature dimension
d = X_all.shape[1] - 1
print('d =', d)

# Show class distribution in the full dataset
print("Class distribution in full dataset:", Counter(Y_all))

# Identify indices for each class
idx_0 = np.where(Y_all == 0)[0]
idx_1 = np.where(Y_all == 1)[0]

# Choose equal number of samples from each class
n_samples_per_class = min(len(idx_0), len(idx_1), 1500)  # capped at 1500 per class
idx_balanced = np.concatenate([
    np.random.choice(idx_0, size=n_samples_per_class, replace=False),
    np.random.choice(idx_1, size=n_samples_per_class, replace=False),
])

# Shuffle balanced indices
np.random.shuffle(idx_balanced)

# Create balanced training set
X_train = X_all[idx_balanced]
Y_train = Y_all[idx_balanced]

# Remaining indices (not used in training) go to calibration/test pool
idx_remaining = np.setdiff1d(np.arange(len(Y_all)), idx_balanced)
X_temp = X_all[idx_remaining]
Y_temp = Y_all[idx_remaining]

# Show class distributions
print("Class distribution in training set:", Counter(Y_train))
print("Class distribution in calibration/test set:", Counter(Y_temp))

# Clean up
del X_all, Y_all

# model training
model = train_logistic_regression(X_train, Y_train)
Y_hat = model.predict_proba(X_train)[:,1]
counts, _, _ = plt.hist(Y_hat, bins=30, density=True, color='skyblue', edgecolor='black', alpha=0.75, label='Histogram of y_hat')
M= counts.max()

# split calibration and test set
X_cal, X_test, Y_cal, Y_test = train_test_split(
    X_temp, Y_temp, train_size=10000, random_state=42
)
Y_proba = model.predict_proba(X_cal)[:,1]

# run the algorithm for 1000 runs
results_3 = evaluate_prc_over_seeds(X_cal, Y_cal, X_test, Y_test, model, args.N, args.alpha, args.tightness, args.delta, args.tau, args.gamma, M, bound_type="clt")

#### for visualization
# Extract values
T_vals = np.array(results_3["stopping_iters"])
#loss_vals = np.array(results_3["losses"])
error_vals = np.array(results_3["errors"])

plt.figure(figsize=(8, 5))

# Plot T vs Loss
# plt.scatter(T_vals, loss_vals, color='blue', label='Loss', alpha=0.6, marker='o')

# Plot T vs Error
plt.scatter(T_vals, error_vals, color='red', label='Type II Error', alpha=0.6, marker='x')

# Plot upper and lower bound
plt.axhline(args.alpha, linestyle='--', color='#d62728', label=r'Upper Bound $\alpha$', linewidth=1.5)  # red
plt.axhline(args.alpha - args.tightness, linestyle='--', color='#2ca02c', label=r'Lower Bound', linewidth=1.5)  # green

plt.xlabel("Stopping Iteration $T$", fontsize=14)
plt.ylabel("Value", fontsize=14)
plt.xlim(3, 9)
plt.ylim(0.1,0.3)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
#plt.title("Scatter Plot of Loss and Type II Error vs Stopping Iteration", fontsize=14)
plt.show()

# Define bounds
upper_bound = args.alpha
lower_bound = args.alpha - args.tightness

# Calculate percentage inside the bounds
# loss_inside = np.logical_and(loss_vals >= lower_bound, loss_vals <= upper_bound)
error_inside = np.logical_and(error_vals >= lower_bound, error_vals <= upper_bound)

# loss_inside_percentage = 100 * np.mean(loss_inside)
error_inside_percentage = 100 * np.mean(error_inside)

# print(f"Percentage of Loss inside bounds: {loss_inside_percentage:.2f}%")
print(f"Percentage of Type II Error inside bounds: {error_inside_percentage:.2f}%")