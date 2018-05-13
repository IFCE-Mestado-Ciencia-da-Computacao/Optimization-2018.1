class LineSearch(object):
    def calculate(self, f, x) -> float:
        pass


class ConstantLineSearch(LineSearch):
    def __init__(self, constant=1e-2):
        self.constant = constant

    def calculate(self, f, x) -> float:
        return self.constant


class ExactLineSearch(LineSearch):

    def calculate(self, f, x):
        """
        $$t = arg min_{s>0} f(x + s \delta x)$$

        return t
        """
        #t = arg min_{t>0} f(x + t delta x)
        return


class BacktrackingLineSearch(LineSearch):

    def __init__(self, alpha=0.5, beta=0.5):
        """
        :param float alpha: $\alpha \in (0, 1/2)$
        :param float beta:  $\beta \in (0, 1)$
        """
        self.alpha = alpha
        self.beta = beta

    def calculate(self, f, x):
        """
        :param Function f:
        :param ndarray x:

        :return:
        """
        grad = f.gradient(x)
        delta_x = - grad

        t = 1

        while all(f(x + t*delta_x) > f(x) + self.alpha * t * grad.T * delta_x):
            t = self.beta * t

        return t
