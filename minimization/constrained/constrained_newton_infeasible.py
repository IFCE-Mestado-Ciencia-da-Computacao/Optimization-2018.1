import numpy as np
from numpy.linalg import norm

from minimization.constrained.equality_restrictions import LinearRestrictions
from minimization.constrained.kkt.kkt_matrix import KKTMatrixInfeasible
from minimization.constrained.kkt.kkt_solver import KKTSolver
from minimization.unconstrained.general_descent_method import GeneralDescentMethod
from minimization.util import concatenate


class ConstrainedNewtonInfeasible(GeneralDescentMethod):

    def __init__(self, line_search: 'BacktrackingLineSearchInfeasible', kkt_solver: KKTSolver, inspect=None):
        """
        :param KKTSolver kkt_solver:
        """
        self.line_search = line_search
        self.kkt_solver = kkt_solver
        self.inspect = inspect if inspect is not None else lambda f, h, x, ν, k: None

    def minimize(self, f, h: LinearRestrictions, x, tolerance=1e-4, iterations=60):
        ν = np.array([[.5] * len(h)]).T
        history = []

        r = ConstrainedNewtonInfeasible.r(f, h)

        for k in range(iterations):
            history.append(self.inspect(f, h, x, ν, k))

            # 1. Compute primal and dual Newton steps Δx_{nt}, Δν_{nt}
            delta_x, delta_ν = self.deltas(f, h, x, ν)

            # Backtracking line search on |r|_2
            t = self.line_search.calculate(f, h, x, ν, delta_x, delta_ν)

            # 3. Update
            x = x + t * delta_x
            ν = ν + t * delta_ν

            if (h.A@x == h.b) and norm(r(x, ν)) <= tolerance:
                break

        history.append(self.inspect(f, h, x, ν, len(history)))

        return np.array(history)

    def deltas(self, f, h, x, ν):
        matrix = KKTMatrixInfeasible(f, h, x, ν)

        return self.kkt_solver.calculate(matrix)

    @staticmethod
    def r(f, h: LinearRestrictions):
        """
        r(x, ν) = (∇f(x) + A.T ν, Ax − b)
        """
        return lambda x, ν: concatenate([
            [ConstrainedNewtonInfeasible.r_primal(f, h, x, ν)],
            [ConstrainedNewtonInfeasible.r_dual(h, x)]
        ])

    @staticmethod
    def r_primal(f, h: LinearRestrictions, x, ν):
        A = h.A

        return f.gradient(x) + A.T @ ν

    @staticmethod
    def r_dual(h: LinearRestrictions, x):
        A = h.A
        b = h.b

        return A@x - b
