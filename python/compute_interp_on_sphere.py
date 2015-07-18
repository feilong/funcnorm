import numpy as np
from compute_geodesic_distances import compute_geodesic_distances
from interp_f import gds_to_interp_vals


def compute_interp_on_sphere(T, cart_coords, neighbors, warp, nn=False):
    """
    Parameters
    ----------
    T : (n_timepoints, n_nodes) array
    cart_coords : (3, n_nodes) array
    neighbors : (max_nbrs, n_nodes) array
    warp : (3, n_nodes) array
    nn : bool
        Whether using the nearest neighbor or 3 nearest neighbors.

    Returns
    -------
    TW : (n_timepoints, n_nodes) array

    """
    n_nodes = T.shape[1]

    warp_cart_coords = cart_coords + warp
    TW = np.zeros(T.shape)

    num_nbrs = np.zeros((n_nodes, ))
    neighbors = np.vstack(
        [np.array(range(n_nodes), dtype='int')[np.newaxis, :], neighbors])

    for j in range(n_nodes):
        num_nbrs[j] = np.sum(neighbors[:, j] != -99)

    for j in range(n_nodes):
        curr_cart_coords = warp_cart_coords[:, [j]]

        c_nbrs = neighbors[:num_nbrs[j], j]
        prev_nbr = j

        while True:
            projs = curr_cart_coords.T.dot(cart_coords[:, c_nbrs])
            I = np.argmax(projs)
            closest_neighbor = c_nbrs[I]

            c_nbrs = neighbors[:num_nbrs[closest_neighbor], closest_neighbor]

            if prev_nbr == closest_neighbor:
                break

            prev_nbr = closest_neighbor

        # Sort projections in descending order (closest grid points first)
        I = np.argsort(
            curr_cart_coords.T.dot(cart_coords[:, c_nbrs]).ravel())[::-1]

        if nn:
            tri_nbrs = c_nbrs[[I[0]]]
        else:
            tri_nbrs = c_nbrs[I[:3]]

        tri_cart_coords = cart_coords[:, tri_nbrs]
        gds = compute_geodesic_distances(curr_cart_coords, tri_cart_coords)
        interp_vals, non_zero_locs = gds_to_interp_vals(gds, 1)
        if len(non_zero_locs) == 0:
            raise ValueError('non_zero_locs empty for node %d' % j)

        TW[:, j] = T[:, tri_nbrs].dot(interp_vals.T)

    return TW
