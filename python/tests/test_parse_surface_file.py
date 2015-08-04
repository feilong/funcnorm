import os
from scipy.io import savemat
from nose.tools import assert_equal, assert_true

from ..parse_surface_file import parse_surface_file


DIR = os.path.dirname(os.path.realpath(__file__))


def test_parse_surface_file():
    surf_file = os.path.join(DIR, '..', 'data', 'lh.sphere.reg.asc')
    n_nodes, coords = parse_surface_file(surf_file)
    assert_equal(n_nodes, 149063)
    assert_equal(coords['cart_coords'].shape, (3, n_nodes))
    assert_equal(coords['triangles'].shape, (3, 298122))
    assert_true(coords['triangles'].min() >= 0)
    assert_true(coords['triangles'].max() < n_nodes)
    assert_equal(coords['neighbors'].shape[1], n_nodes)


def test_convert_surface_to_mat():
    """test_convert_surface_to_mat
    This script will parse the surface in `../data/lh.sphere.reg.asc` and save
    the results in a Matlab file `../data/lh.sphere.reg.mat`.
    Some of the testings may use the results directly rather than the surface
    file.
    """
    surf_file = os.path.join(DIR, '..', 'data', 'lh.sphere.reg.asc')
    out_file = os.path.join(DIR, '..', 'data', 'lh.sphere.reg.mat')
    n_nodes, coords = parse_surface_file(surf_file)
    savemat(out_file, {'coords': coords})