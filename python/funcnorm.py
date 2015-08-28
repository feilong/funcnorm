import os
import numpy as np
import logging
from scipy.io import loadmat, savemat

from .utils import init_logging, renormalize_warps
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


def determine_subj_ordering(subjects, hems, dirs):
    """
    Notes
    -----
    I followed the implementation of the matlab version.
    The first subject might get changed in the last part.
    Probably we should improve it in the future.
    """
    cache_file = os.path.join(dirs['align'], 'subject_order.mat')
    if os.path.exists(cache_file):
        logger.info("Found existing subject order file, skip calculation...")
        mat = loadmat(cache_file)
        print tuple(mat['shape'].ravel())
        return mat['subjects'], mat['shape'].ravel()
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
    savemat(cache_file, {'subjects': subjects3, 'shape': atlas.shape})
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


def determine_start_point(n_passes, subjects, hems, dirs):
    for hem in hems:
        hem_out_dir = os.path.join(dirs['warps'], hem)
        if not os.path.exists(hem_out_dir):
            os.makedirs(hem_out_dir)
    n_subjects = len(subjects)
    start_pass = 0
    start_subj = 0
    n_subj = len(subjects)
    if _check_warp_completed(subjects, hems, dirs):
        logger.info("Atlas construction has already been completed.")
        exit(0)
    for pass_num in range(n_passes):
        for subj_num, subj in enumerate(subjects):
            if not _check_warp_file_exists(subj, hems, pass_num, 0, dirs):
                logger.info("The algorithm will start on pass #{pass_num} "
                            "and subject #{subj_num} ({subj}) out of "
                            "{n_subjects}".format(**locals()))
                return pass_num, subj_num


def _get_warp_filename(subj, hem, pass_num, zero_corrected, dirs):
    corr_str = '_corrected' if zero_corrected else ''
    return os.path.join(
        dirs['tmp'],
        'warp_{subj}_{hem}_{pass_num}{corr_str}.mat'.format(**locals()))


def _save_warp(warp, subj, hems, pass_num, zero_corrected, dirs):
    nph = warp.shape[0] / len(hems)
    for num_hem, hem in enumerate(hems):
        warp_data = warp[num_hem*nph:(num_hem+1)*nph, :]
        fname = _get_warp_filename(subj, hem, pass_num, zero_corrected, dirs)
        savemat(fname, {'warp': warp_data})


def _load_warp(subj, hems, pass_num, zero_corrected, dirs):
    warps = []
    for hem in hems:
        fname = _get_warp_filename(subj, hem, pass_num, zero_corrected, dirs)
        warps.append(loadmat(fname)['warp'])
    return np.vstack(warps)


def _check_warp_file_exists(subj, hems, pass_num, zero_corrected, dirs):
    for hem in hems:
        fname = _get_warp_filename(subj, hem, pass_num, zero_corrected, dirs)
        if not os.path.exists(fname):
            return False
    return True


def _get_tmp_time_series_filename(subj, hem, pass_num, dirs):
    return os.path.join(dirs['tmp'],
                        'TS_{subj}_{hem}_{pass_num}.mat'.format(**locals()))


def _save_tmp_time_series(TS, subj, hems, pass_num, dirs):
    nph = TS.shape[1] / len(hems)
    for num_hem, hem in enumerate(hems):
        TS_data = TS[:, num_hem*nph:(num_hem+1)*nph]
        fname = _get_tmp_time_series_filename(subj, hem, pass_num, dirs)
        savemat(fname, {'TS': TS_data})


def _load_tmp_time_series(subj, hems, pass_num, dirs):
    time_series = []
    for hem in hems:
        fname = _get_tmp_time_series_filename(subj, hem, pass_num, dirs)
        time_series.append(loadmat(fname)['TS'])
    return np.hstack(time_series)


def _check_tmp_time_series_exists(subj, hems, pass_num, dirs):
    for hem in hems:
        fname = _get_tmp_time_series_filename(subj, hem, pass_num, dirs)
        if not os.path.exists(fname):
            return False
    return True


def calc_warp(pass_num, subj_num, subj,
              surf, subjects, hems, dirs,
              lambda_metric, lambda_areal, max_res):
    if pass_num == 0 and subj_num == 0:
        _save_warp(np.zeros((surf.n_nodes, 3)), subj, hems, 0, 0, dirs)
        # TS = load_orig_time_series(subj, hems)
        # _save_tmp_time_series(TS, subj, hems, dirs)
        return
    logger.info("Creating atlas for alignment.")
    atlas_subj = range(len(subjects)) if pass_num > 0 else range(subj_num)
    atlas_pass_num = pass_num - 1 if pass_num else 0
    atlas_TS = np.array(
        [_load_tmp_time_series(subjects[j], hems, atlas_pass_num, dirs)
         for j in atlas_subj if j != subj_num]
        ).mean(axis=0)
    logger.info("Completed creating atlas for alignment.")

    # TODO separate logging file
    TS = load_orig_time_series(subj, hems)
    logger.info("Performing alignment of subject #{subj_num} ({subj}) "
                "on pass #{pass_num}".format(**locals()))
    warp = np.zeros((surf.n_nodes, 3)) if pass_num == 0 else \
           _load_warp(subj, hems, pass_num-1, pass_num>1, dirs)
    warp = funcnorm_register(TS, atlas_TS, surf, warp,
                             lambda_metric, lambda_areal, max_res)
    logger.info("Completed alignment of subject #{subj_num} ({subj}) "
                "on pass #{pass_num}".format(**locals()))
    _save_warp(warp, subj, hems, pass_num, 0, dirs)


def funcnorm():
    """
    nbrs: (36002, 6)
    triangles: (72000, 3)
    cart: (36002, 3)
    """
    # subjects = ['ag00', 'ab00', 'ap00']
    subjects = ['ab00', 'ag00', 'ap00', 'aw00', 'er00', 'gw00', 'ls00', 'mg00',
                'pk00', 'ro00', 'sg00']
    hems = ['lh', 'rh']
    lambda_metric = 30.0
    lambda_areal = 30.0
    n_passes = 4  # one more than the matlab version due to 0-based indexing
    max_res = 3

    init_logging()
    logger.info("\033[32mHello world!\033[0m")

    dirs = {'align': os.path.join(DIR, os.pardir, 'results', 'align'),
            'tmp': os.path.join(DIR, os.pardir, 'results', 'align', 'tmp'),
            'warps': os.path.join(DIR, os.pardir, 'results', 'align', 'warps'),
        }
    for folder in dirs.values():
        if not os.path.exists(folder):
            logger.info("Creating folder: %s" % folder)
            os.makedirs(folder)

    subjects, shape = determine_subj_ordering(subjects, hems, dirs)

    surf_file = os.path.join(DIR, os.pardir, 'results', 'standard2mm_sphere.reg.asc')

    lambda_metric /= 4.0
    lambda_areal /= 2.0
    # BUG? Matlab version used n_nodes in funcnorm but n_triangles in
    # funcnorm_register, which is pretty weird.

    # pass_num, subj_num = determine_start_point(n_passes, subjects, hems, dirs)
    pass_num, subj_num = 0, 0
    logger.info("Starting with pass #{pass_num} and subject #{subj_num}".format(**locals()))

    for pass_num in range(pass_num, n_passes):
        logger.info("Running through pass #%d..." % pass_num)
        for subj_num in range(subj_num, len(subjects)):
            subj = subjects[subj_num]
            logger.info("Running pass #{pass_num} and subject #{subj_num} "
                        "({subj})".format(**locals()))
            surf = surf_from_file(surf_file)
            surf.normalize_cart()
            surf.multi_hem(len(hems))
            warp_exists = _check_warp_file_exists(subj, hems, pass_num, 0, dirs)
            if not warp_exists:
                calc_warp(pass_num, subj_num, subj, surf, subjects, hems, dirs,
                          lambda_metric, lambda_areal, max_res)
            TS_exists = _check_tmp_time_series_exists(subj, hems, pass_num, dirs)
            if not TS_exists:
                logger.info("Calculating interpolated time series.")
                warp = _load_warp(subj, hems, pass_num, 0, dirs)
                TS = load_orig_time_series(subj, hems)
                if subj_num > 0 or pass_num > 0:
                    TS = surf.interp_time_series(warp, TS, False)
                _save_tmp_time_series(TS, subj, hems, pass_num, dirs)
                logger.info("Completed calculating interpolated time series.")
        if pass_num > 0:
            logger.info("Zero correcting the warps...")
            warps = [_load_warp(subj, hems, pass_num, 0, dirs)
                     for subj in subjects]
            warps_zero = renormalize_warps(surf.cart, warps)
            for subj_num, subj in enumerate(subjects):
                _save_warp(warps_zero[subj_num], subj, hems, pass_num, 1, dirs)
            logger.info("Completed zero correcting the warps.")
        subj_num = 0
        logger.info("Completed running through pass #%d..." % pass_num)

    for subj_num, subj in enumerate(subjects):
        _save_final_warp(subj)

    logger.info("Completed alignment.")


def test_funcnorm():
    try:
        funcnorm()
    except Exception as e:
        logger.exception(e)
