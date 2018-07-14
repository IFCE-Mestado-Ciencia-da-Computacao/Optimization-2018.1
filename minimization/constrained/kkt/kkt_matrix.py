import numpy as np

from minimization.constrained.equality_restrictions import LinearRestrictions
from minimization.function import Function
from minimization.util import concatenate


class KKTMatrixFeasible(object):

    def __init__(self, f: Function, h: LinearRestrictions, x):
        self.f = f
        self.h = h
        self.x = x

    @property
    def A(self):
        A = self.h.A
        H = self.f.hessian(self.x)

        zeros_shape = A.shape[0]
        zeros = np.zeros((zeros_shape, zeros_shape))

        return concatenate([
            [H,   A.T],
            [A, zeros],
        ])

    @property
    def b(self):
        x = self.x
        f = self.f
        A = self.h.A

        zeros_shape = A.shape[0]
        zeros = np.zeros((zeros_shape, 1))

        return - concatenate([
            [f.gradient(x)],
            [zeros]
        ])


class KKTMatrixInfeasible(KKTMatrixFeasible):

    def __init__(self, f: Function, h: LinearRestrictions, x, ν):
        super().__init__(f, h, x)
        self.ν = ν

    @property
    def b(self):
        x = self.x
        f = self.f
        A = self.h.A
        b = self.h.b
        ν = self.ν

        return - concatenate([
            [f.gradient(x) + A.T @ ν],
            [A @ x - b]
        ])
