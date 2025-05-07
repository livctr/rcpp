"""Loading data from file"""

import pandas as pd
import numpy as np
from sklearn import preprocessing


import numpy as np
import pandas as pd
from sklearn import preprocessing

def load_data(file_loc, train_size=3000):
    """Load data from csv file, split into a balanced train set of specified size, 
    and return the rest.

    Parameters
    ----------
    file_loc : str
        Path to the '.csv' training data file (with index in column 0).
    train_size : int
        Total number of samples in the returned training set (balanced 50/50).

    Returns
    -------
    X_tr : np.ndarray
        Feature matrix for the train set (zero‐mean/unit‐variance + bias column).
    Y_tr : np.ndarray
        Labels for the train set (0/1).
    X_rest : np.ndarray
        Feature matrix for the remainder of the data.
    Y_rest : np.ndarray
        Labels for the remainder of the data.
    data : pandas.DataFrame
        The full raw DataFrame (after dropping NAs).
    """
    # --- load & preprocess full dataset ---
    data = pd.read_csv(file_loc, index_col=0)
    data.dropna(inplace=True)
    
    # features
    X_all = data.drop('SeriousDlqin2yrs', axis=1)
    X_all = preprocessing.scale(X_all)
    X_all = np.hstack([X_all, np.ones((X_all.shape[0], 1))])  # add bias term
    
    # labels
    Y_all = data['SeriousDlqin2yrs'].to_numpy()
    
    # --- find indices by class ---
    pos_idx = np.where(Y_all == 1)[0]
    neg_idx = np.where(Y_all == 0)[0]
    
    # how many per class
    n_per_class = train_size // 2
    if n_per_class > len(pos_idx):
        raise ValueError(f"Not enough positives ({len(pos_idx)}) to sample {n_per_class}")
    if n_per_class > len(neg_idx):
        raise ValueError(f"Not enough negatives ({len(neg_idx)}) to sample {n_per_class}")
    
    # sample without replacement
    pos_sample = np.random.choice(pos_idx,  size=n_per_class, replace=False)
    neg_sample = np.random.choice(neg_idx,  size=n_per_class, replace=False)
    train_idx = np.concatenate([pos_sample, neg_sample])
    
    # shuffle train indices
    np.random.shuffle(train_idx)
    
    # the rest
    all_idx = np.arange(len(Y_all))
    rest_idx = np.setdiff1d(all_idx, train_idx, assume_unique=True)
    
    # build splits
    X_tr   = X_all[ train_idx ]
    Y_tr   = Y_all[ train_idx ]
    X_rest = X_all[ rest_idx ]
    Y_rest = Y_all[ rest_idx ]
    
    return X_tr, Y_tr, X_rest, Y_rest, data

