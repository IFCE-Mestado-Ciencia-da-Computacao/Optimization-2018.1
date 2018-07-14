from numpy.linalg import norm

from minimization.constrained.constrained_newton_infeasible import ConstrainedNewtonInfeasible
from minimization.constrained.equality_restrictions import LinearRestrictions
from minimization.unconstrained.line_search import BacktrackingLineSearch


class BacktrackingLineSearchInfeasible(BacktrackingLineSearch):

    def calculate(self, f, h: LinearRestrictions, x, ν, delta_x, delta_ν):
        r = ConstrainedNewtonInfeasible.r(f, h)
        α = self.alpha
        β = self.beta

        t = 1

        while norm(r(x + t*delta_x, ν + t*delta_ν)) > (1 - α*t)*norm(r(x, ν)):
            t = β*t

        return t
