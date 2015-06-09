from datetime import datetime


def display_log_item(s, log_file=None, inc_time=True):
    """ Displays `s` on screen. Also display current time (optional) and
    write `s` to log file (optional).

    Parameters
    ----------
    s : str
        The string to be displayed (and logged if `log_file` is specified).
    log_file : None or str, optional
        If `log_file` is a string, what's displayed will also be logged into
        that file.
    inc_time : bool
        Whether append the current time at the end of the line.
    """
    if inc_time:
        s += '  **(%s)**' % datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write(s + '\n')
    print s
    # There is `pause(1e-6);` in the Matlab code. Why do we need that?
