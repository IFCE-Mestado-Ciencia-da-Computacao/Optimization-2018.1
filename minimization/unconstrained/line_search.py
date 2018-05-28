import numpy as np


class LineSearch(object):
    def calculate(self, f, x, delta_x) -> float:
        pass


class ConstantLineSearch(LineSearch):
    def __init__(self, constant=1e-2):
        self.constant = constant

    def calculate(self, f, x, delta_x) -> float:
        return self.constant


class ExactLineSearch(LineSearch):

    def __init__(self, minimum=0.01, maximum=0.9, number_steps=200):
        self.min = minimum
        self.max = maximum
        self.step_size = (maximum - minimum) / number_steps

    def calculate(self, f, x, delta_x):
        """
        $$t = arg min_{s>0} f(x + s \delta x)$$

        return t
        """
        min_y = np.inf
        min_t = np.inf

        for t in np.arange(self.min, self.max, self.step_size):
            y = f(x + t * delta_x)
            if y < min_y:
                min_y = y
                min_t = t

        return min_t


class BacktrackingLineSearch(LineSearch):

    def __init__(self, alpha=0.5, beta=0.5):
        """
        :param float alpha: $\alpha \in (0, 1/2)$
        :param float beta:  $\beta \in (0, 1)$
        """
        self.alpha = alpha
        self.beta = beta

    def calculate(self, f, x, delta_x):
        """
        given a descent direction ∆x for f at x ∈ dom f , α \in (0, 0.5), β \in (0, 1).

        t = 1.
        while f(x + t∆x) > f(x) + αt ∇f(x).T ∆x:
            t = βt.
        """
        gradient = f.gradient(x)
        t = 1

        while f(x + t*delta_x) > f(x) + self.alpha * t * gradient.T @ delta_x:
            t = self.beta * t

        return t
