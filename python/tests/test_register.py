import numpy as np

from .utils import random_dataset, random_triangles, random_cart

from ..Register import funcnorm_register
from ..Surface import Surface
from ..utils import init_logging

init_logging()


def test_funcnorm_register():
    np.random.seed(0)
    n_triangles = 3000
    n_nodes = 1000
    n_timepoints = 300
    res = 2
    triangles, nbrs, num_nbrs = random_triangles(n_nodes, n_triangles)
    cart = random_cart(n_nodes)
    surf = Surface(cart, nbrs, triangles)
    ds1 = random_dataset(n_timepoints, n_nodes)
    ds2 = random_dataset(n_timepoints, n_nodes)
    cart_warp = np.zeros((n_nodes, 3))

    cart_warp = funcnorm_register(ds1, ds2, surf, cart_warp, 0, 0, res)
