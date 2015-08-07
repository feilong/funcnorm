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


def _calc_nbrs(triangles, n_nodes, max_nbrs=6):
    nbrs = -99 * np.ones((n_nodes, max_nbrs), dtype='int')
    num_nbrs = np.zeros((n_nodes, ), dtype='int')
    n_triangles = triangles.shape[0]
    for i in xrange(n_triangles):
        nodes = triangles[i, :]
        for n1 in nodes:
            for n2 in nodes:
                if n1 == n2 or np.any(nbrs[n1, :] == n2):
                    continue
                count = num_nbrs[n1]
                if count >= nbrs.shape[1]:
                    nbrs = np.hstack(
                        [nbrs, -99 * np.ones((n_nodes, 1), 'int')])
                nbrs[n1, count] = n2
                num_nbrs[n1] += 1
    if np.max(num_nbrs) < max_nbrs:
        nbrs = nbrs[:, :np.max(num_nbrs)]
    return nbrs, num_nbrs


def _parse_surface_file(surf_file, dtype='float', max_nbrs=6):
    with open(surf_file, 'r') as f:
        lines = f.read().splitlines()
    lines = [_ for _ in lines if not _.startswith('#')]
    n_nodes, n_triangles = [int(_) for _ in lines[0].split()]
    cart = np.zeros((n_nodes, 3), dtype=dtype)
    for i in range(n_nodes):
        cart[i, :] = [float(_) for _ in lines[i+1].split()[:3]]
    triangles = np.zeros((n_triangles, 3), dtype='int')
    for i in xrange(n_triangles):
        triangles[i, :] = [int(_) for _ in lines[i+n_nodes+1].split()[:3]]
    nbrs, num_nbrs = _calc_nbrs(triangles, n_nodes)
    return cart, nbrs, triangles


def surf_from_file(surf_file, dtype='float', max_nbrs=6):
    cart, nbrs, triangles = _parse_surface_file(surf_file, dtype, max_nbrs)
    surf = Surface(cart, nbrs, triangles)
    return surf


class Surface(object):
    def __init__(self, cart, nbrs, triangles):
        """
        Try using (n_nodes, 3) arrays for cart.
        Try using coord_maps with [0, 1, 2]
        """
        self.cart = cart
        self.nbrs = nbrs
        self.triangles = triangles

        self.n_nodes = cart.shape[0]

        self.warp_cart = None
        self.cart_warped = None

    def _calc_coord_maps(self):
        if self.cart_warped is None:
            self.maps = np.argmin(self.cart, axis=1)
        else:
            self.maps = np.argmin(np.max(self.cart, self.cart_warpped), axis=1)

    def _calc_spher_coords(self, maps=None):
        self._spher = _calc_spher_coords(self.cart, self.mapping)
