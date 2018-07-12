import numpy as np
from numpy.linalg import inv

from minimization.constrained.kkt.kkt_matrix import KKTMatrixFeasible
from minimization.unconstrained.general_descent_method import GeneralDescentMethod


class ConstrainedNewtonFeasible(GeneralDescentMethod):

    def __init__(self, line_search, kkt_solver):
        """
        :param LineSearch line_search:
        :param KKTSolver kkt_solver:
        """
        self.line_search = line_search
        self.kkt_solver = kkt_solver

    def minimize(self, f, h, x, tolerance=1e-4, iterations=120):
        """
        :param h: Restrictions
        """
        history = []

        for k in range(iterations):
            # 1. Compute the Newton step and decrement Δx_{nt}, λ(x)
            delta_x = self.delta_x(f, h, x)
            lambda_square = - f.gradient(x).T @ delta_x

            history.append(np.concatenate([[f(x)], *x, lambda_square[0]]))

            # 2. Stopping criterion. quit if λ^2/2 <= tolerance
            if lambda_square/2 <= tolerance:
                break

            # 3. Line search. Choose step size t by backtracking line search.
            t = self.line_search.calculate(f, x, delta_x)

            # 4. Update. x := x + t∆x_{nt}
            x = x + t * delta_x

        return np.array(history)

    def delta_x(self, f, h, x):
        matrix = KKTMatrixFeasible(f, h, x)

        delta_x, w = self.kkt_solver.calculate(matrix)
        return delta_x


class ConstrainedNewtonInfeasible(object):

    def minimize(self, f, x, tolerance=1e-4, iterations=60):
        history = []

        for k in range(iterations):
            # Primal Newton step
            delta_x = - inv(f.hessian(x)) @ f.gradient(x)
            # Dual Newton step
            delta_v

            # Backtracking line search on ||r||_2
            t = 1
            while 1:
                t = beta * t

            t = self.line_search.calculate(f, x, delta_x)

            x = x + t * delta_x
            v = v + t * delta_v

            #if not A*x == b or not ||r(x, v)||_2 <= tolerance:
            #    break

        return np.array(history)

    def delta_x(self):
        """
        |∇2f(x) A^T| = |delta_x| = | ∇f(x)|
        |A        0| = |      w| = |Ax - b|
        """
        A = A

        de
        pass
