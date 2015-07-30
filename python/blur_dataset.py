import numpy as np

from interp_f import interp_f


def blur_dataset(V, s, cart_coords, nbrs, total_nbrs, resolution, U=None):
    """
    Parameters
    ----------
    V : (n_nodes, K) array
    s : (K, ) array
    U : (n_timepoints, K) array, optional
    cart_coords : (3, n_nodes) array
    nbrs : (max_nbrs, n_nodes) array
    total_nbrs : (n_nodes, ) array
    resolution : int or float?

    Returns
    -------
    V : array
    s : array
    U : array, optional
        U will be returned only if it's provided by function call.
    """
    Q = blur_dataset_no_svd(V, s, cart_coords, nbrs, total_nbrs, resolution, U)

    U_new, s, V = np.linalg.svd(Q, full_matrices=False)
    V = V.T
    if U is None:
        return V, s
    else:
        return V, s, U.dot(U_new)


def blur_dataset_no_svd(V, s, cart_coords, nbrs, total_nbrs, resolution,
                        U=None):
    """
    Parameters
    ----------
    V : (n_nodes, K) array
    s : (K, ) array
    U : (n_timepoints, K) array, optional
    cart_coords : (3, n_nodes) array
    nbrs : (max_nbrs, n_nodes) array
    total_nbrs : (n_nodes, ) array
    resolution : int or float?

    Returns
    -------
    Q : (K, n_nodes) array
    """
    n_nodes, K = V.shape
    # Y would be (K, n_nodes)
    Y = np.tile(s.reshape((K, 1)), (1, n_nodes)) * V.T
    dtype = Y.dtype

    Q = np.zeros((K, n_nodes), dtype=dtype)

    # This is the old interpolation code in Matlab
    # But using the same thresholds as the new code
    for j in range(n_nodes):
        curr_cart_coords = cart_coords[:, [j]]
        n_nbrs = total_nbrs[j]
        curr_nbrs = nbrs[:n_nbrs, j]
        nbr_cart_coords = cart_coords[:, curr_nbrs]

        A, non_zero_locs = interp_f(
            curr_cart_coords, nbr_cart_coords, resolution)
        A = A[non_zero_locs]
        curr_nbrs = curr_nbrs[non_zero_locs]
        Q[:, j] = Y[:, curr_nbrs].dot(A)
        qnorm = np.linalg.norm(Q[:, j])
        if qnorm > 1e-8:
            Q[:, j] /= qnorm
        else:
            Q[:, j] = 0

    return Q

    # "New method", not implemented in Python yet.

    # locs = np.where(nbrs == -99)
    # max_nbrs = total_nbrs.max()

    # full_upd_nbrs = nbrs.copy()  # Do we need deep copy here?
    # full_upd_nbrs[locs] = locs[1]  # locs[1] == which column
    # cc = np.repeat(cart_coords, max_nbrs, axis=1)
    # nc = cart_coords[:, full_upd_nbrs.T.ravel()]

    # gds = compute_geodesic_distances(cc, nc)
    # A_vals, non_zero_locs = gds_to_interp_vals(
    #     gds, resolution, dtype, np.where(nbrs.T.ravel() == -99)[0])

    # NI = full_upd_nbrs.ravel()[non_zero_locs]  # non-zero neighbor indices
    # A = A_vals.reshape((n_nodes, max_nbrs)).T

    # return A
