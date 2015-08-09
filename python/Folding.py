import numpy as np

from .utils import mapping


def _triangles_to_vectors(triangles, cart):
    As = cart[triangles[:, 1], :] - cart[triangles[:, 0], :]
    Bs = cart[triangles[:, 2], :] - cart[triangles[:, 0], :]
    return As, Bs


def _dcart_dspher(cart, maps):
    dp_dphi = np.zeros(cart.shape, cart.dtype)
    dp_dtheta = np.zeros(cart.shape, cart.dtype)
    mags = np.linalg.norm(cart, axis=1)
    for i in range(cart.shape[0]):
        idx = mapping[maps[i]]
        xx, yy, zz = cart[i, idx]
        phi = np.arccos(xx/mags[i])
        theta = np.arctan2(yy, zz)
        dp_dphi[i, idx] = (-np.sin(phi),
                           np.cos(phi) * np.sin(theta),
                           np.cos(phi) * np.cos(theta))
        dp_dtheta[i, idx] = (0,
                             np.sin(phi) * np.cos(theta),
                             -np.sin(phi) * np.sin(theta))
    return dp_dphi, dp_dtheta


def _calc_areal_terms(triangles, cart, maps, orig_tri_areas, tri_normals,
                      compute_derivatives=True):
    As, Bs = _triangles_to_vectors(triangles, cart)
    abcrosses = np.cross(As, Bs)
    new_areas = 0.5 * np.sum(abcrosses * tri_normals, axis=1)
    locs = np.where(new_areas <= 0)[0]
    a_terms = new_areas - orig_tri_areas
    areal = np.sum(a_terms[locs]**2)
    if not compute_derivatives:
        return areal
    dareal_dphi = np.zeros((cart.shape[0], ), cart.dtype)
    dareal_dtheta = np.zeros((cart.shape[0], ), cart.dtype)
    dp_dphi, dp_dtheta = _dcart_dspher(cart, maps)
    bcns = np.cross(Bs, tri_normals)
    ncas = np.cross(tri_normals, As)
    for loc in locs:
        tmp1 = a_terms[loc] * bcns[loc, :]
        tmp2 = a_terms[loc] * ncas[loc, :]
        tmps = [-tmp1 - tmp2, tmp1, tmp2]
        for node in range(3):
            cI = triangles[loc, node]
            tmp = tmps[node]  # (3, )
            dareal_dphi[cI] += np.sum(tmp * dp_dphi[cI, :])
            dareal_dtheta[cI] += np.sum(tmp * dp_dtheta[cI, :])
    return areal, dareal_dphi, dareal_dtheta


def _calc_oriented_areas(triangles, cart):
    As, Bs = _triangles_to_vectors(triangles, cart)
    ns = np.cross(As, Bs)
    n_norms = np.linalg.norm(ns, axis=1)
    tri_areas = 0.5 * n_norms
    oriented_normals = ns / np.tile(n_norms[:, np.newaxis], (1, 3))
    return tri_areas, oriented_normals
