import numpy as np

mapping = [[0, 1, 2], [1, 2, 0], [2, 1, 0]]


def _normalize(cart):
    norm = np.linalg.norm(cart, axis=1)
    normalized = cart / np.tile(norm[:, np.newaxis], (1, 3))
    return normalized


def _calc_spher_coords(cart, maps):
    n_nodes = cart.shape[0]
    if isinstance(maps, int):
        maps = np.ones((n_nodes, ), dtype='int') * maps
    if str(cart.dtype).startswith('complex'):
        raise ValueError("Cartesian coordinates are complex numbers.")
    spher = np.zeros((n_nodes, 2), dtype=cart.dtype)
    mags = np.linalg.norm(cart, axis=1)
    count = 0
    for i in range(3):
        idx = np.where(maps == i)[0]
        if len(idx):
            c = cart[idx, :][:, mapping[i]]
            spher[idx, 0] = np.arccos(c[:, 0] / mags[idx])
            spher[idx, 1] = np.arctan2(c[:, 1], c[:, 2])
            count += len(idx)
    if count != n_nodes:
        raise ValueError("_calc_spher_coords: Incorrect coordinate maps.")
    return spher


def _calc_cart_coords(spher, maps):
    n_nodes = spher.shape[0]
    if isinstance(maps, int):
        maps = np.ones((n_nodes, ), dtype='int') * maps
    if str(spher.dtype).startswith('complex'):
        raise ValueError("Cartesian coordinates are complex numbers.")
    cart = np.zeros((n_nodes, 3), dtype=spher.dtype)
    phis, thetas = spher[:, 0], spher[:, 1]
    count = 0
    for i in range(3):
        idx = np.where(maps == i)[0]
        if len(idx):
            cart[idx, mapping[i][0]] = np.cos(phis[idx])
            cart[idx, mapping[i][1]] = np.sin(phis[idx]) * np.sin(thetas[idx])
            cart[idx, mapping[i][2]] = np.sin(phis[idx]) * np.cos(thetas[idx])
            count += len(idx)
    if count != n_nodes:
        raise ValueError("_calc_cart_coords: Incorrect coordinate maps.")
    return cart


def _calc_cart_warped_from_spher_warp(cart, spher_warp, maps, spher=None):
    if spher is None:
        spher = _calc_spher_coords(cart, maps)
    spher_warped = spher + spher_warp
    cart_warped = _calc_cart_coords(spher_warped, maps)
    return cart_warped


def _calc_spher_warp_from_cart_warp(cart, cart_warp, maps, spher=None):
    if spher is None:
        spher = _calc_spher_coords(cart, maps)
    spher_warped = _calc_spher_coords(cart + cart_warp, maps)
    spher_warp = spher_warped - spher
    return spher_warp
