import os
import platform
from distutils.spawn import find_executable

from display_log_item import display_log_item


def funcnorm_prepare_wrapper(
        subjects, hems, experiment, vol_data_dir, suffix,
        inp_subj_dir, save_subj_dir, preprocess_movie, mm_res,
        fwhm, fs_surf, out_dir, log_file=None,
        warp_experiment=None, align_dir=None, inp_unix_path=None):
    """
    Parameters
    ----------
    subjects : list of str
        The names of the subjects in a list
    hems : list of str
        The hemispheres to align, e.g., ['lh', 'rh']
    experiment : str
        E.g. 'movie5min_P1'
    vol_data_dir : str
        The directory containing input volume files
    suffix : str
    inp_subj_dir : str
        System variable SUBJECTSDIR for FreeSurfer
    save_subj_dir : str
    preprocess_movie : bool
    mm_res : float or int
        Voxel size in mm
    fwhm : float or int
        The fwhm for SurfSmooth
    fs_surf : str
        Freesurfer surface file, e.g., 'sphere' or 'sphere.reg'
    out_dir : str
        The specified output directory
    log_file : None or str, optional
        File to store log items, relative to `out_dir`.

    Notes
    -----
    There was a parameter named `extensions` but never used in the code.
    """
    if inp_unix_path is None:
        inp_unix_path = os.getenv("PATH")

    # nsubj = len(subjects)
    # nhem = len(hems)

    display_log_item('Preparing the data with funcnorm_prepare.sh script...',
                     log_file)

    # Setup environment funcnorm_prepare.sh expects
    # FREESURFER_HOME
    fs_bin = find_executable('mris_convert')
    if fs_bin is None:
        raise IOError('mris_convert not found')
    fs_dir = os.path.abspath(os.path.join(fs_bin, os.pardir, os.pardir))
    os.putenv('FREESURFER_HOME', fs_dir)
    # unset LD_LIBRARY_PATH, which messes up FreeSurfer on Linux
    if platform.system() == 'Linux':
        orig_ldpath = os.getenv('LD_LIBRARY_PATH')
        os.unsetenv('LD_LIBRARY_PATH')
    # SUBJECTS_DIR
    os.putenv('SUBJECTS_DIR', inp_subj_dir)
    # PATH
    os.putenv('PATH', inp_unix_path)

    for subj in subjects:
        for hem in hems:
            display_log_item(
                'Preparing %s (%s) with funcnorm_prepare.sh' % (subj, hem),
                log_file)
            cmd = 'sh ./funcnorm_prepare.sh '\
                  '-subj {subj} -hem {hem} '\
                  '-input_dir {vol_data_dir} '\
                  '-experiment {experiment} '
            if warp_experiment:
                cmd += '-warp_experiment {warp_experiment} '
            cmd += '-suffix {suffix} '\
                   '-subjects_dir {save_subj_dir} '\
                   '-outputdir {out_dir} '\
                   '-fs_surf {fs_surf} '\
                   '-mm {mm_res} -fwhm {fwhm} '
            if align_dir:
                cmd += '-alignDir {align_dir} '
            if preprocess_movie:
                cmd += '-preprocess_movie '
            cmd += '-verbosity 2'
            cmd = cmd.format(**locals())

            print cmd
            retval = os.system(cmd)
            if retval:
                raise IOError('problem running funcnorm_prepare')

    if platform.system() == 'Linux':
        os.putenv('LD_LIBRARY_PATH', orig_ldpath)

    display_log_item(
        'Completed preparing the data with funcnorm_prepare.sh script',
        log_file)
