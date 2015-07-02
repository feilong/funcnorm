import numpy as np

from generate_neighbors_matrix_from_triangles import \
    generate_neighbors_matrix_from_triangles


def parse_surface_file(surf_file, dtype='float', max_nbrs=6):
    """
    Parameters
    ----------
    surf_file : str
        Usually located in the output directory.
        E.g., '{out_dir}/standard{mm_res}mm_{fs_surf}.asc'
        This is actually a SUMA file
        mm_res is the resolution of the surface mesh in mm
        fs_surf is freesurfer surface type (e.g., 'sphere.reg')
    dtype : dtype or str, optional. Default is 'float'.
        The dtype of the numpy array coords['cart_coords'].
    max_nbrs : int, optional. Default is 6.
        The estimated maximum number of neighbors for any node.
        Used for initialization only.

    Returns
    -------
    n_nodes : int
        The number of nodes in the mesh
    coords : dict
        coords['cart_coords'] : (3, n_nodes) array
        coords['triangles'] : (3, n_triangles) array, triplets of node numbers
            Node numbers start with 0.
        coords['neighbors'] : (max_nbrs, n_nodes) array
            The nearest-neighbors of each node in the mesh.
    """

    with open(surf_file, 'r') as f:
        lines = f.read().splitlines()
    lines = [line for line in lines if not line.startswith('#')]

    n_nodes, n_triangles = [int(_) for _ in lines[0].split()]

    cart_coords = np.ones((3, n_nodes), dtype=dtype)
    for i in xrange(n_nodes):
        # We only need three numbers but there is always an extra zero
        # Why is there always a trailing zero?
        cart_coords[:, i] = [float(_) for _ in lines[i+1].split()[:3]]

    triangles = np.ones((3, n_triangles), dtype='int')
    for i in xrange(n_triangles):
        triangles[:, i] = [int(_) for _ in lines[i+n_nodes+1].split()[:3]]

    neighbors = generate_neighbors_matrix_from_triangles(
        triangles, n_nodes, max_nbrs)[0]

    coords = {
        'cart_coords': cart_coords,
        'triangles': triangles,
        'neighbors': neighbors,
    }
    return n_nodes, coords
