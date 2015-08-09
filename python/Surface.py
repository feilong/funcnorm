import numpy as np

from .Coordinates import _calc_spher_coords


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


def _calc_nbr_res0(orig_nbrs, max_res):
    n_nodes, max_nbrs = orig_nbrs.shape
    nbrs = -99 * np.ones((n_nodes, min(max_nbrs+1, 200)), dtype='int')
    res_nbr_sizes = np.zeros((n_nodes, max_res), dtype='int')
    for j in range(n_nodes):
        n_nbrs = np.sum(orig_nbrs[j, :] != -99)
        if orig_nbrs[j, 0] == j:
            nbrs[j, :n_nbrs] = orig_nbrs[j, :n_nbrs]
        else:
            n_nbrs += 1
            nbrs[j, :n_nbrs] = [j] + orig_nbrs[j, :n_nbrs].tolist()
        res_nbr_sizes[j, 0] = n_nbrs
        start, end = 1, n_nbrs
        for res in range(1, max_res):
            prev_nbrs = nbrs[j, start:end]
            nbrs_nbrs = np.setdiff_1d(orig_nbrs[prev_nbrs, :], [-99])
            new_nbrs = np.setdiff_1d(nbrs_nbrs, nbrs[j, :end])
            new_size = len(new_nbrs)
            res_nbr_sizes[j, res] = new_size
            start, end = end, end + new_size
            if nbrs.shape[0] < end:
                shape = (nbrs.shape[0], end-nbrs.shape[1])
                nbrs = np.hstack([nbrs, -99*np.ones(shape, 'int')])
                nbrs[j, start:end] = new_nbrs
    num_nbrs = np.sum(nbrs != -99, axis=1)
    return nbrs, res_nbr_sizes, num_nbrs


def _calc_nbr_res(nbrs, max_res):
    n_nodes, max_nbrs = nbrs.shape
    res_nbr_sizes = np.zeros((n_nodes, max_res), dtype='int')
    nbrs = np.hstack([nbrs, -99 * np.ones((n_nodes, 1), 'int')])
    for j in range(n_nodes):
        if nbrs[j, 0] != j:
            nbrs[j, 1:] = nbrs[j, :-1]
            nbrs[j, 0] = j
        n_nbrs = np.sum(nbrs[j, :] != -99)
        res_nbr_sizes[j, 0] = n_nbrs
        start, end = 1, n_nbrs
        for res in range(1, max_res):
            prev_nbrs = nbrs[j, start:end]
            nbrs_nbrs = np.setdiff1d(nbrs[prev_nbrs, :], [-99])
            new_nbrs = np.setdiff1d(nbrs_nbrs, nbrs[j, :end])
            new_size = len(new_nbrs)
            res_nbr_sizes[j, res] = new_size
            start, end = end, end + new_size
            if nbrs.shape[1] < end:
                shape = (nbrs.shape[0], end-nbrs.shape[1])
                nbrs = np.hstack([nbrs, -99*np.ones(shape, 'int')])
                nbrs[j, start:end] = new_nbrs
    num_nbrs = np.sum(nbrs != -99, axis=1)
    return nbrs, res_nbr_sizes, num_nbrs


def _update_nbr_res(cart, cart_warped, nbrs, res_nbr_sizes, num_nbrs,
                    upd_nbrs, upd_res_nbr_sizes, upd_num_nbrs):
    """
    Replace neighbors of warped node with neighbors of its nearest node in the
    un-warped space.
    """
    n_nodes = cart.shape[0]
    for i in range(n_nodes):
        closest_nbr_pre = upd_nbrs[i, 0]
        while True:
            curr_nbrs = upd_nbrs[i, :upd_num_nbrs[i]]
            nbrs_cart = cart[curr_nbrs, :]
            projections = nbrs_cart.dot(cart_warped[i, :])
            closest_nbr = curr_nbrs[np.argmax(projections)]
            upd_nbrs[i, :] = nbrs[closest_nbr, :]
            upd_res_nbr_sizes[i, :] = res_nbr_sizes[closest_nbr, :]
            upd_num_nbrs[i] = num_nbrs[closest_nbr]
            if closest_nbr_pre == closest_nbr:
                break
            closest_nbr_pre = closest_nbr
    # return upd_nbrs, upd_res_nbr_sizes, upd_num_nbrs


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

    def calc_coord_maps(self):
        if self.cart_warped is None:
            self.maps = np.argmin(self.cart, axis=1)
        else:
            self.maps = np.argmin(np.maximum(self.cart, self.cart_warped),
                                  axis=1)

    def _calc_spher_coords(self):
        self.spher = _calc_spher_coords(self.cart, self.maps)

    def calc_nbr_res(self, max_res):
        self.nbrs, self.res_nbr_sizes, self.num_nbrs = _calc_nbr_res(
            self.nbrs, max_res)

    def init_upd_nbr_res(self):
        self.upd_nbrs = self.nbrs.copy()
        self.upd_res_nbr_sizes = self.res_nbr_sizes.copy()
        self.upd_num_nbrs = self.num_nbrs.copy()

    def update_nbr_res(self):
        _update_nbr_res(
            self.cart, self.cart_warped,
            self.nbrs, self.res_nbr_sizes, self.num_nbrs,
            self.upd_nbrs, self.upd_res_nbr_sizes, self.upd_num_nbrs)

    def calc_coords_list(self):
        self.coords_list = []
        for i in range(3):
            spher = _calc_spher_coords(self.cart,
                                       i * np.ones((self.n_nodes, )))
            self.coords_list.append(spher)
