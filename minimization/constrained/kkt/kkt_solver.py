import numpy as np
from scipy.linalg import ldl


class KKTSolver(object):

    def __init__(self, method):
        self.method = method

    def calculate(self, matrix):
        """
        :param matrix:
        :return: delta_x, delta_v (or w, depende do que seja)
        """
        A, b = matrix.A, matrix.b
        result = self.method(A, b)

        return np.split(result, [matrix.x.shape[0]])


def lapark_method(A, b):
    return np.linalg.solve(A, b)


def inverse_method(A, b):
    return np.linalg.inv(A) @ b


def pseudo_inverse_method(A, b):
    return np.linalg.pinv(A) @ b


def ldl_method(A, b):
    L, D, P = ldl(A)

    z = np.linalg.solve(L, b)
    y = np.linalg.solve(D, z)
    x = np.linalg.solve(L.T, y)

    return x
