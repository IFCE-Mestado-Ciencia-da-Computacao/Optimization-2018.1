import numpy as np

from minimization.constrained import EqualityRestrictions
from minimization.function import Function
from minimization.util import concatenate


class KKT(object):

    def none(self):
        None



class KKTMatrix(object):

    def __init__(self, f: Function, h: EqualityRestrictions, x):
        pass

    def A_KKT_matrix(self, f: Function, h: EqualityRestrictions, x):
        A = h.A
        H = f.hessian(x)

        zeros_shape = A.shape[0]
        zeros = np.zeros((zeros_shape, zeros_shape))

        return concatenate([
            [H,   A.T],
            [A, zeros],
        ])

    def b_KKT_matrix(self, f: Function, h: EqualityRestrictions, x):
        A = h.A
        b = h.b

        return concatenate([
            [f.gradient(x) + A.T @ v],
            [A @ x - b]
        ])
