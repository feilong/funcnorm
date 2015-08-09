import numpy as np
from numpy.testing import assert_allclose


def _calc_oriented_area(cart, normal):
    A = cart[1, :] - cart[0, :]
    B = cart[2, :] - cart[0, :]
    oriented_area = 0.5 * np.sum(np.cross(A, B) * normal)
    return oriented_area


def _calc_oriented_area_derivatives(cart, normal):
    der = np.zeros((3, 3))
    A = cart[1, :] - cart[0, :]
    B = cart[2, :] - cart[0, :]
    bcn = np.cross(B, normal)
    nca = np.cross(normal, A)
    der[0, :] = 0.5 * (-bcn-nca)
    der[1, :] = 0.5 * bcn
    der[2, :] = 0.5 * nca
    return der


def _normalize(cart):
    norm = np.linalg.norm(cart, axis=1)
    normalized = cart / np.tile(norm[:, np.newaxis], (1, 3))
    return normalized


def test_oriented_area_derivatives():
    n_triangles = 300
    atol, rtol = 1e-6, 1e-6
    for t in range(n_triangles):
        cart = _normalize(np.random.random((3, 3)))
        normal = _normalize(np.random.random((1, 3))).ravel()

        Area = _calc_oriented_area(cart, normal)
        der = _calc_oriented_area_derivatives(cart, normal)

        cart2 = cart.copy()
        delta = 1e-8
        der2 = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                cart2[i, j] += delta
                Area2 = _calc_oriented_area(cart2, normal)
                der2[i, j] = (Area2 - Area) / delta
                cart2[i, j] = cart[i, j]
        assert_allclose(der, der2, atol=atol, rtol=rtol)
