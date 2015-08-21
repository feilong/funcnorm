import numpy as np
import logging.config

mapping = [[0, 1, 2], [1, 2, 0], [2, 1, 0]]


def _calc_geodesic_dist(coords1, coords2):
    if coords1.shape[1] != coords2.shape[1]:
        raise ValueError('coords1 and coords2 must have the '
                         'same number of columns.')
    if coords1.shape[0] == 1:
        coords1 = np.tile(coords1, (coords2.shape[0], 1))
    if coords1.shape[1] == 3:
        gds = np.arcsin(np.sqrt((np.cross(coords1, coords2)**2).sum(axis=1)))
    elif coords1.shape[1] == 2:
        phi1, theta1 = coords1[:, 0], coords1[:, 1]
        phi2, theta2 = coords2[:, 0], coords2[:, 1]
        gds = 2 * np.arcsin(np.sqrt(
            np.sin(phi1)*np.sin(phi2)*(np.sin((theta1-theta2)/2))**2
            + (np.sin((phi1 - phi2)/2))**2
        ))
    else:
        raise ValueError('Coordinates must have 2 or 3 columns.')
    return gds


def init_logging():
    logging.config.dictConfig({
        'version': 1,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s]: %(message)s'
            },
        },
        'handlers': {
            'console': {
                'level': 'INFO',
                'formatter': 'standard',
                'class': 'logging.StreamHandler',
            },
            'logfile-debug': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'DEBUG',
                'formatter': 'standard',
                'maxBytes': 10485760,
                'backupCount': 20,
                'filename': 'funcnorm_debug.log',
                'mode': 'a',
            },
            'logfile-info': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'INFO',
                'formatter': 'standard',
                'maxBytes': 10485760,
                'backupCount': 20,
                'filename': 'funcnorm_info.log',
                'mode': 'a',
            },
        },
        'loggers': {
            'funcnorm': {
                'level': 'DEBUG',
                'handlers': ['console', 'logfile-debug', 'logfile-info'],
                'propagate': False,
            }
        },
        'root': {
            'level': 'INFO',
            'handlers': ['console', 'logfile-debug', 'logfile-info'],
        }
    })
