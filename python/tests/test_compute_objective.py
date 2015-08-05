import numpy as np

from ..compute_objective import compute_objective
from ..compute_spherical_from_cartesian import compute_spherical_from_cartesian
from ..compute_neighbor_resolutions import compute_neighbor_resolutions
from ..compute_geodesic_distances import compute_geodesic_distances
from ..regularization import compute_initial_oriented_areas

from .test_utils import random_dataset, random_nbrs, random_coords


def test_compute_objective_smoke_test():
    seed, n_nodes, n_timepoints = 0, 300, 100
    res = 2
    n_triangles = 500

    cart_coords, spher_coords, coord_maps = random_coords(seed, n_nodes)
    nbrs, total_nbrs = random_nbrs(seed, n_nodes)
    nbrs, res_nbr_sizes, total_nbrs = compute_neighbor_resolutions(nbrs, res)
    x = random_coords(seed+1, n_nodes)[1]
    spher_coords_list = [compute_spherical_from_cartesian(cart_coords, i+1)
                         for i in range(3)]

    # upd_nbrs, upd_total_nbrs = random_nbrs(seed+1, n_nodes)
    # upd_nbrs, upd_res_nbr_sizes, upd_total_nbrs = \
    #     compute_neighbor_resolutions(upd_nbrs, res)
    upd_nbrs, upd_res_nbr_sizes, upd_total_nbrs = (
        nbrs, res_nbr_sizes, total_nbrs)

    ds1 = random_dataset(seed, n_nodes, n_timepoints)
    ds2 = random_dataset(seed+1, n_nodes, n_timepoints)

    U, s, Vt = np.linalg.svd(ds1, full_matrices=False)
    V1ST = np.tile(s[:, np.newaxis], (1, n_nodes)) * Vt
    W2TU1 = ds2.T.dot(U)

    K = s.shape[0]
    regularization = {'mode': 'metric_and_areal_distortion',
                      'lambda_metric': 1.0,
                      'lambda_areal': 1.0,
                      }
    regularization['metric_distances'] = -99 * np.ones(nbrs.shape)
    for j in range(n_nodes):
        curr_cart_coords = cart_coords[:, [j]]
        n_nbrs = total_nbrs[j]
        curr_nbrs = nbrs[:n_nbrs, j]
        nbr_cart_coords = cart_coords[:, curr_nbrs]
        regularization['metric_distances'][:n_nbrs, j] = \
            compute_geodesic_distances(curr_cart_coords, nbr_cart_coords)
    regularization['triangles'] = np.array(
        [np.random.choice(range(n_nodes), (3, ), False)
         for _ in range(n_triangles)])
    regularization['oriented_areas'], regularization['oriented_normals'] = \
        compute_initial_oriented_areas(
            regularization['triangles'], cart_coords)

    f, g = compute_objective(
        x=x, cart_coords=cart_coords, coord_maps=coord_maps,
        spher_coords_list=spher_coords_list,
        nbrs=nbrs, res_nbr_sizes=res_nbr_sizes, total_nbrs=total_nbrs,
        upd_nbrs=upd_nbrs, upd_res_nbr_sizes=upd_res_nbr_sizes,
        upd_total_nbrs=upd_total_nbrs,
        K=K, res=res, regularization=regularization, V1ST=V1ST, W2TU1=W2TU1)
