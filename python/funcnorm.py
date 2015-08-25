import os
import numpy as np
import logging

from .utils import init_logging
from .Surface import surf_from_file
from .io import load_time_series
from .Register import funcnorm_register

DIR = os.path.dirname(os.path.realpath(__file__))
logger = logging.getLogger('funcnorm')


def calc_datasets_correlations(ds1, ds2, thr=1e-8):
    n_nodes = ds1.shape[0]
    corrs = np.zeros((n_nodes, ))
    for j in range(n_nodes):
        mag = np.linalg.norm(ds1[:, j]) * np.linalg.norm(ds2[:, j])
        corr = ds1[:, j].dot(ds2[:, j]) / mag if mag > thr else 0.0
        corrs[j] = corr
    return corrs


def load_orig_time_series(subj, hems):
    # TODO: outdir
    Ts = []
    for hem in hems:
        fname = os.path.join(
            DIR, os.pardir, 'results',
            '%s_%s_2mm_fwhm0_raidersP1_on_sphere.reg.niml.dset' % (subj, hem))
        T = load_time_series(fname)
        Ts.append(T)
    Ts = np.hstack(Ts)
    logger.debug("Loaded time series of subject %s, shape is %s." % \
                 (subj, Ts.shape))
    return Ts


def determine_subj_ordering(subjects, hems):
    """
    Notes
    -----
    I followed the implementation of the matlab version.
    The first subject might get changed in the last part.
    Probably we should improve it in the future.
    """
    logger.info("Determining the order of subjects...")
    logger.info("Original order of subjects is: %r" % subjects)
    datasets = {}
    for subj in subjects:
        datasets[subj] = load_orig_time_series(subj, hems)
    n_subjects = len(subjects)
    n_timepoints, n_nodes = datasets.values()[0].shape
    corr_mat = np.zeros((n_subjects, n_subjects))
    for i, subj1 in enumerate(subjects):
        for j, subj2 in enumerate(subjects):
            if j == i:
                corr_mat[i, j] = 1.0 * n_timepoints
            elif j < i:
                corr_mat[i, j] = corr_mat[j, i]
            else:
                corr_mat[i, j] = calc_datasets_correlations(
                    datasets[subj1], datasets[subj2]).mean()
    corrs = np.mean(corr_mat, axis=0)
    logger.debug("Sum of correlations for each subject: %s" % corrs)
    idx = np.argsort(corrs)[::-1]
    subjects2 = [subjects[_] for _ in idx]
    logger.info("Order of subjects by inter-correlation: " + \
                 ', '.join(subjects2))

    atlas = np.array([datasets[_] for _ in subjects2[1:]]).sum(axis=0)
    logger.debug("Creating atlas, shape is %s." % (atlas.shape, ))
    corrs = np.zeros((n_subjects, ))
    for i, subj in enumerate(subjects2):
        if i != 0:
            atlas -= datasets[subj]
        corrs[i] = calc_datasets_correlations(datasets[subj], atlas).mean()
        if i != 0:
            atlas += datasets[subj]
    idx = np.argsort(corrs)[::-1]
    logger.debug("Correlations with atlas: %s" % corrs)
    subjects3 = [subjects2[_] for _ in idx]
    logger.info("Order of subjects by correlation with atlas: " + \
                 ', '.join(subjects3))
    return subjects3, atlas.shape


def _check_warp_completed(subjects, hems, dirs):
    for subj in subjects:
        for hem in hems:
            final = os.path.join(
                dirs['warps'], hem,
                'warp_{subj}_{hem}_final.bin'.format(**locals()))
            if not os.path.exists(final):
                return False
    return True


def _check_warp_file_exists(subj, hems, pass_num, dirs):
    zero_corrected = False
    corr_str = '_corrected' if zero_corrected else ''
    for hem in hems:
        warp_file = os.path.join(
            dirs['tmp'],
            'warp_{subj}_{hem}_{pass_num}{corr_str}.bin'.format(**locals()))
        if not os.path.exists(warp_file):
            return False
    return True


def determine_start_point(n_passes, subjects, hems, dirs):
    for hem in hems:
        hem_out_dir = os.path.join(dirs['warps'], hem)
        if not os.path.exists(hem_out_dir):
            os.makedirs(hem_out_dir)
    start_pass = 0
    start_subj = 0
    n_subj = len(subjects)
    if _check_warp_completed(subjects, hems, dirs):
        logger.info("Atlas construction has already been completed.")
        exit(0)
    for pass_num in range(n_passes):
        for subj_num, subj in enumerate(subjects):
            if not _check_warp_file_exists(subj, hems, pass_num, dirs):
                logger.info("The algorithm will start on pass #{pass_num} "
                            "and subject #{subj_num} ({subj}) out of "
                            "{n_subjects}".format(**locals()))
                return pass_num, subj_num


def test_funcnorm():
    """
    nbrs: (36002, 6)
    triangles: (72000, 3)
    cart: (36002, 3)
    """
    # subjects = ['ag00', 'ab00', 'ap00']
    subjects = ['ab00', 'ag00', 'ap00', 'aw00', 'er00', 'gw00', 'ls00', 'mg00',
                'pk00', 'ro00', 'sg00']
    hems = ['lh', 'rh']
    init_logging()
    logger.info("\033[32mHello world!\033[0m")

    subjects, shape = determine_subj_ordering(subjects, hems)

    dirs = {'align': os.path.join(DIR, os.pardir, 'results', 'align'),
            'tmp': os.path.join(DIR, os.pardir, 'results', 'align', 'tmp'),
            'warps': os.path.join(DIR, os.pardir, 'results', 'align', 'tmp'),
        }
    for folder in dirs.values():
        if not os.path.exists(folder):
            os.makedirs(folder)

    surf_file = os.path.join(DIR, os.pardir, 'results', 'standard2mm_sphere.reg.asc')
    surf = surf_from_file(surf_file)
    surf.normalize_cart()
    surf.multi_hem(len(hems))

    lambda_metric /= 4.0
    lambda_areal /= 2.0
    # BUG? Matlab version used n_nodes in funcnorm but n_triangles in
    # funcnorm_register, which is pretty weird.

    pass_num, subj_num = determine_start_point()

    for pass_num in range(pass_num, n_passes):
        logger.info("Running through pass #%d..." % pass_num)
        for subj_num in range(subj_num, len(subjects)):
            subj = subjects[subj_num]
            if pass_num == 0 and subj_num == 0:
                _save_warp(np.zeros((n_nodes, 3)), subj, 0, 0)
                atlas_TS = _load_orig_time_series(subj)
                save_tmp_time_series(atlas_TS, subj)
                continue
            logger.info("Re-creating atlas from previous run...")
            if pass_num == 0:
                atlas_TS = np.array(
                    [_load_tmp_time_series(subjects[j]) for j in range(subj_num)]
                ).mean(axis=0)
            else:
                atlas_TS = np.array(
                    [_load_tmp_time_series(subjects[j])
                     for j in range(len(subjects)) if j != subj_num]
                ).mean(axis=0)
            logger.info("Completed re-creating atlas from previous run.")

            # TODO separate logging file
            TS = _load_orig_time_series(subj)
            logger.info("Performing alignment of subject #{subj_num} ({subj}) "
                        "on pass #{pass_num}".format(**locals()))
            if num_pass == 0:
                warp = np.zeros((n_nodes, 3))
            else:
                warp = _load_warp(subj, pass_num-1, pass_num>1)
            if pass_num == 0:
                warp = funcnorm_register(atlas_TS, TS, surf, warp, # why atlas_TS first?
                                         lambda_metric, lambda_areal, max_res)
            else:
                warp = funcnorm_register(TS, atlas_TS, surf, warp,
                                         lambda_metric, lambda_areal, max_res)
            logger.info("Completed alignment of subject #{subj_num} ({subj}) "
                        "on pass #{pass_num}".format(**locals()))
            _save_warp(warp, subj, 0, 0)
            TS = _compute_interp_on_sphere(TS, surf, warp)  # TODO
            _save_tmp_time_series(TS, subj)
            if pass_num == 0:
                atlas_TS += TS
        if pass_num > 0:
            logger.info("Zero correcting the warps...")
            warps = [_load_warp(subj, pass_num, 0) for subj in subjects]
            warps_zero = compute_zero_correction(surf, warps)
            for subj_num, subj in enumerate(subjects):
                _save_warp(warps_zero[subj_num], subj, pass_num, 1)
            logger.info("Completed zero correcting the warps.")
        subj_num = 0
        logger.info("Completed running through pass #%d..." % pass_num)

    for subj_num, subj in enumerate(subjects):
        _save_final_warp(subj)

    logger.info("Completed alignment.")





    # fname1 = os.path.join(DIR, '..', 'results',
    #                     'ag00_lh_2mm_fwhm0_raidersP1_on_sphere.reg.niml.dset')
    # fname2 = os.path.join(DIR, '..', 'results',
    #                     'ap00_lh_2mm_fwhm0_raidersP1_on_sphere.reg.niml.dset')
    # ds1 = load_time_series(fname1)
    # ds2 = load_time_series(fname2)
    # n_timepoints, n_nodes = ds1.shape
    # funcnorm_register(ds1, ds2, surf, np.zeros((n_nodes, 3)),
    #                   30.0, 30.0, 3)



# from .parse_surface_file import parse_surface_file
# from .funcnorm_register import funcnorm_register

# def test_cart_coords_from_surf():
#     DIR = '/data/movies/raiders/dartmouth/fsrecon/ab00/ab00/SUMA/'
#     files = [
#         # 'ico64_lh.sphere.reg.asc',
#         'ico64_lh.sphere.asc',
#         'ico64_lh.smoothwm.asc',
#     ]
#     for fname in files:
#         n_nodes, coords = parse_surface_file(DIR + fname)
#         cart_coords = coords['cart_coords']
#         print fname, n_nodes, np.linalg.norm(cart_coords, axis=0)

# def test_funcnorm_register():
#     logger = logging.getLogger('funcnorm')
#     logger.info("Hello")

# def _test_gds():
#     init_logging()
#     surf_file = os.path.join(DIR, '..', 'results', 'standard2mm_sphere.reg.asc')
#     surf = surf_from_file(surf_file)
#     surf.normalize_cart()
#     surf.calc_nbr_res(2)
#     print surf.nbrs.shape
#     # for i in range(surf.n_nodes):
#         # surf.nbrs[i, :]
