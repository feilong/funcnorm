import numpy as np

mapping = [[0, 1, 2], [1, 2, 0], [2, 1, 0]]


def _calc_geodesic_dist(coords1, coords2):
    if coords1.shape[1] != coords2.shape[1]:
        raise ValueError("coords1 and coords2 must have the "
                         "same number of columns.")
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
        raise ValueError("Coordinates must have 2 or 3 columns.")
    return gds
