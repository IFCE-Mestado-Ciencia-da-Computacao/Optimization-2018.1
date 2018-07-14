import numpy as np
from scipy.linalg import ldl

from minimization.util import concatenate


class KKTSolver(object):

    def __init__(self, method):
        self.method = method

    def calculate(self, matrix):
        """
        :param matrix:
        :return: delta_x, delta_v (or w, depende do que seja)
        """
        A, b = matrix.A, matrix.b
        result = self.method(A, b, matrix)

        return np.split(result, [matrix.x.shape[0]])


def lapark_method(A, b, *args):
    return np.linalg.solve(A, b)


def inverse_method(A, b, *args):
    return np.linalg.inv(A) @ b


def pseudo_inverse_method(A, b, *args):
    return np.linalg.pinv(A) @ b


def ldl_method(A, b, *args):
    L, D, P = ldl(A)

    z = np.linalg.solve(L, b)
    y = np.linalg.solve(D, z)
    x = np.linalg.solve(L.T, y)

    return x


def kkt_elimination(A, b, matrix):
    """
    :param A: A da matriz KKT
    :param b: b da matriz KKT
    :param matrix: matriz Kkt
    :return:
    """
    x = matrix.x
    A = matrix.h.A  # A matriz 'A' utilizada é a matriz de restrições
    H = matrix.f.hessian(x)
    H_inv = np.linalg.inv(H)

    # Extract 'g' and 'h' from -b
    g, h = np.split(-b, [x.shape[0]])

    w = np.linalg.solve(A @ H_inv @ A.T, h - A @ H_inv @ g)
    v = np.linalg.solve(H, -(g + A.T @ w))

    return concatenate([
        [v],
        [w]
    ])
