import numpy as np


def normalize_dataset(T, thr=1e-8):
    """Normalize each column to unit norm (and centered).
    Note unit norm is different from unit variance.

    Parameters
    ----------
    T : (n_timepoints, n_nodes) array
        Each column is a time-series.
    thr : float
        Threshold. If the norm of a time-series is less than `thr`, it will
        be considered as noise and replaced by zeros.

    Returns
    -------
    T : (n_timepoints, n_nodes) array

    """
    n_timepoints, n_nodes = T.shape

    # Center the dataset (demean)
    T -= np.tile(T.mean(axis=0).reshape((1, n_nodes)), (n_timepoints, 1))

    # Normalization
    for j in range(n_nodes):
        magT = np.sqrt(np.sum(T[:, j]**2))
        if magT > thr:
            T[:, j] /= magT
        else:
            T[:, j] = 0

    return T
