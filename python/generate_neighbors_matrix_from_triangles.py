"""
% FUNCTION [nbrs, nbrCounts] =
%     generateNeighborsMatrixFromTriangles(T, numNodes, maxNbrs)
% nbrs is maxNbrs x numNodes matrix
% nbrCounts is 1 x numNodes vector (number of neighbors of each node)
%
% T is 3 x numTriangles matrix
% numNodes is the total number of nodes
% maxNbrs is the maximum number of neighbors of any node

% This file is part of the Functional Normalization Toolbox,
% (c) 2008 by the authors.
% Please see AUTHORS and LICENSE file in the project root directory
"""
import numpy as np


def generate_neighbors_matrix_from_triangles(triangles, n_nodes, max_nbrs):
    """
    Parameters
    ----------
    triangles : (3, n_triangles) array
    n_nodes : int
        The total number of nodes.
    max_nbrs : int
        The estimated maximum number of neighbors of any node.
        Used to help initialization only.

    Returns
    -------
    nbrs : (max_nbrs, n_nodes) array
        Here `max_nbrs` is the actual maximum number of neighbors.
    nbr_counts : (n_nodes, ) array
        The number of neighbors for each node.

    Notes
    -----
    This function is only called by `parse_surface_file`.
    Probably we should merge this function into that file.

    Would it be better to use lists to store neighbor information?
    """

    nbrs = -99 * np.ones((max_nbrs, n_nodes), dtype='int')
    nbr_counts = np.zeros((n_nodes, ), dtype='int')

    n_triangles = triangles.shape[1]

    for j in xrange(n_triangles):
        nodes = triangles[:, j]
        for n1 in nodes:
            for n2 in nodes:
                if n1 == n2:
                    continue
                count = nbr_counts[n1]
                if np.sum(nbrs[:, n1] == n2) == 0:
                    if count >= nbrs.shape[0]:
                        nbrs = np.vstack([
                            nbrs,
                            -99 * np.ones((1, n_nodes), 'int')])
                    nbrs[count, n1] = n2
                    nbr_counts[n1] += 1

    return nbrs, nbr_counts
