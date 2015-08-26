import numpy as np
import logging

from .Coordinates import _calc_spher_coords, _normalize
from .Metric import _calc_metric_dist
from .Folding import _calc_oriented_areas
from .Interpolation import _calc_geodesic_dist, _gds_to_interp_weights, \
    _interp_time_series
from .Coordinates import _calc_spher_warp_from_cart_warp, \
    _calc_cart_warped_from_spher_warp

logger = logging.getLogger('funcnorm')


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


def _calc_nbr_res(orig_nbrs, max_res):
    n_nodes, max_nbrs = orig_nbrs.shape
    nbrs = -99 * np.ones((n_nodes, min(max_nbrs+1, 200)), dtype='int')
    res_nbr_sizes = np.zeros((n_nodes, max_res), dtype='int')
    for j in range(n_nodes):
        n_nbrs = np.sum(orig_nbrs[j, :] != -99)
        if orig_nbrs[j, 0] == j:
            nbrs[j, :n_nbrs] = orig_nbrs[j, :n_nbrs]
        else:
            n_nbrs += 1
            nbrs[j, :n_nbrs] = [j] + orig_nbrs[j, :n_nbrs-1].tolist()
        res_nbr_sizes[j, 0] = n_nbrs
        start, end = 1, n_nbrs
        for res in range(1, max_res):
            prev_nbrs = nbrs[j, start:end]
            nbrs_nbrs = np.setdiff1d(orig_nbrs[prev_nbrs, :], [-99])
            new_nbrs = np.setdiff1d(nbrs_nbrs, nbrs[j, :end])
            new_size = len(new_nbrs)
            res_nbr_sizes[j, res] = new_size
            start, end = end, end + new_size
            if nbrs.shape[1] < end:
                shape = (n_nodes, end-nbrs.shape[1])
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
    # TODO: clean up surface after each alignment
    def __init__(self, cart, nbrs, triangles):
        """
        Try using (n_nodes, 3) arrays for cart.
        Try using coord_maps with [0, 1, 2]
        """
        self.cart = cart
        self.nbrs = nbrs
        self.triangles = triangles
        self.num_nbrs = np.sum(nbrs != -99, axis=1)

        self.n_nodes = cart.shape[0]
        self.n_triangles = triangles.shape[0]

        self.cart_warped = None

    def clean_up(self):
        self.nbrs = self.orig_nbrs
        self.num_nbrs = np.sum(nbrs != -99, axis=1)
        del self.res_nbr_sizes
        del self.upd_nbrs, upd_num_nbrs, upd_res_nbr_sizes

        del self.cart_warped, self.maps

        logger.debug("Completed cleaning up Surface, remaining keys: %s" %
                     ', '.join(self.__dict__.keys()))

    def calc_coord_maps(self):
        if self.cart_warped is None:
            self.maps = np.argmin(self.cart, axis=1)
        else:
            self.maps = np.argmin(np.maximum(self.cart, self.cart_warped),
                                  axis=1)
        self._calc_spher_coords()

    def _calc_spher_coords(self):
        self.spher = _calc_spher_coords(self.cart, self.maps)

    def calc_nbr_res(self, max_res):
        self.orig_nbrs = self.nbrs
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

    def normalize_cart(self):
        self.cart = _normalize(self.cart)

    def init_metric(self, dtype='float'):
        self.orig_md = _calc_metric_dist(self.cart, self.orig_nbrs,
                                         self.num_nbrs, dtype)

    def init_areal(self):
        self.tri_areas, self.tri_normals = _calc_oriented_areas(
            self.triangles, self.cart)

    def trim_nbrs(self):
        self.num_nbrs -= self.res_nbr_sizes[:, -1]
        self.upd_num_nbrs -= self.upd_res_nbr_sizes[:, -1]
        self.res_nbr_sizes = self.res_nbr_sizes[:, :-1]
        self.upd_res_nbr_sizes = self.upd_res_nbr_sizes[:, :-1]
        # max_nbrs = max(self.num_nbrs.max(), self.upd_num_nbrs.max())
        max_nbrs = self.num_nbrs.max()
        self.nbrs = self.nbrs[:, :max_nbrs]
        self.upd_nbrs = self.upd_nbrs[:, :max_nbrs]
        for i in range(self.n_nodes):
            self.nbrs[i, self.num_nbrs[i]:] = -99
            self.upd_nbrs[i, self.upd_num_nbrs[i]:] = -99

    def calc_spher_warp(self, cart_warp):
        spher_warp = _calc_spher_warp_from_cart_warp(
            self.cart, cart_warp, self.maps, self.spher)
        return spher_warp

    def calc_cart_warped(self, spher_warp):
        self.cart_warped = _calc_cart_warped_from_spher_warp(
            self.cart, spher_warp, self.maps, self.spher)

    def calc_avg_corr(self, ds1, ds2, res, thr=1e-8):
        corrs = []
        for j in range(self.n_nodes):
            curr_coords = self.cart_warped[[j], :]
            curr_nbrs = self.nbrs[j, :self.num_nbrs[j]]
            nbr_coords = self.cart[curr_nbrs, :]
            gds = _calc_geodesic_dist(curr_coords, nbr_coords)
            A, non_zero = _gds_to_interp_weights(gds, res)
            curr_nbrs = curr_nbrs[non_zero]
            Q = ds1[:, curr_nbrs].dot(A)
            qnorm = np.linalg.norm(Q)
            print qnorm,
            if qnorm < thr:
                continue
            D = 1.0 / qnorm
            Q *= D
            corr = ds2[:, j].dot(Q)
            corrs.append(corr)
        return corrs

    def multi_hem(self, n_hems):
        n_nodes_per_hem = self.n_nodes
        n_triangles_per_hem = self.n_triangles

        logger.debug("Before multi_hem: {self.cart.shape}, {self.nbrs.shape}, "
                     "{self.triangles.shape}, {self.num_nbrs.shape}"
                     "".format(**locals()))
        logger.debug("Before multi_hem: {cart}, {nbrs}, "
                     "{triangles}, {num_nbrs}"
                     "".format(cart=self.cart.max(), nbrs=self.nbrs.max(),
                               triangles=self.triangles.max(),
                               num_nbrs=self.num_nbrs.max()))

        self.cart = np.tile(self.cart, (n_hems, 1))
        self.triangles = np.tile(self.triangles, (n_hems, 1))
        self.num_nbrs = np.tile(self.num_nbrs, (n_hems, ))

        self.n_nodes = self.cart.shape[0]
        self.n_triangles = self.triangles.shape[0]

        for hem_num in range(1, n_hems):
            nbrs = self.nbrs[:n_nodes_per_hem, :]
            nbrs[np.where(nbrs != -99)] += n_nodes_per_hem * hem_num
            self.nbrs = np.vstack([self.nbrs, nbrs])

        self.triangles += (
            np.tile(np.array(range(self.n_triangles))[:, np.newaxis], (1, 3))
            / n_triangles_per_hem) * n_nodes_per_hem

        logger.debug("After multi_hem: {self.cart.shape}, {self.nbrs.shape}, "
                     "{self.triangles.shape}, {self.num_nbrs.shape}"
                     "".format(**locals()))
        logger.debug("After multi_hem: {cart}, {nbrs}, "
                     "{triangles}, {num_nbrs}"
                     "".format(cart=self.cart.max(), nbrs=self.nbrs.max(),
                               triangles=self.triangles.max(),
                               num_nbrs=self.num_nbrs.max()))

    def interp_time_series(self, T, nn=False):
        return _interp_time_series(T, self, nn)
