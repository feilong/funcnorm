import numpy as np
import logging

logger = logging.getLogger('funcnorm')


def load_time_series(filename, in_dtype='f4', out_dtype=None):
    with open(filename, 'rb') as f:
        line = ''
        while 'SPARSE_DATA' not in line:
            line = f.readline()

        while 'ni_form' not in line:
            line = f.readline()
        idx = line.find('sbfirst')
        endian = line[idx-1]
        in_dtype = {'b': '>', 'l': '<'}[endian] + in_dtype

        while 'ni_type' not in line:
            line = f.readline()
        ni_type = line.split('"')[1]
        if '*' not in ni_type:
            n_timepoints = 1
        else:
            n_timepoints = int(ni_type.split('*')[0])

        while 'ni_dimen' not in line:
            line = f.readline()
        n_nodes = int(line.split('"')[1])

        line_start = f.tell()
        while '>' not in line:
            line_start = f.tell()
            line = f.readline()
        idx = line.find('>')
        f.seek(line_start + idx + 1, 0)

        logger.info('Loading binary data, endian: {endian}, '
                    'dtype: {in_dtype}, '
                    'shape: ({n_timepoints}, {n_nodes})'.format(**locals()))

        count = n_timepoints * n_nodes
        data = np.fromfile(f, in_dtype, count).\
            reshape((n_nodes, n_timepoints)).T

        if out_dtype is not None:
            data = data.astype(out_dtype)

        # print np.percentile(data.ravel(), range(5, 100, 10))

        return data
