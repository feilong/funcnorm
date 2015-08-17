import os
from scipy.io import loadmat
from numpy.testing import assert_array_equal
import logging

from ..io import load_time_series

DIR = os.path.dirname(os.path.realpath(__file__))
logger = logging.getLogger('funcnorm')


def test_load_time_series():
    fname = os.path.join(DIR, '..', '..', 'results',
                         'ab00_lh_2mm_fwhm0_raidersP1_on_sphere.reg.niml.dset')
    mat_file = os.path.join(
        DIR, '..', '..', 'results',
        'ab00_lh_2mm_fwhm0_raidersP1_on_sphere.reg.niml.mat')
    if not os.path.exists(fname) or not os.path.exists(mat_file):
        logger.warn("Data file not found, skip current test.")
        return
    ds = load_time_series(fname)
    mat = loadmat(mat_file)
    assert_array_equal(ds, mat['data'])
