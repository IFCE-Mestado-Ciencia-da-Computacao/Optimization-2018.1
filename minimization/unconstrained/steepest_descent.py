import numpy as np
from minimization.unconstrained.first_order_descent import FirstOrderDescent


class SteepestDescent(FirstOrderDescent):
    """
    l_1 form
    """

    def calculate_delta_x(self, f, x):
        gradient = f.gradient(x)
        index = np.argmax(gradient)

        eye = np.eye(len(gradient), dtype=int)
        return - (gradient * eye)[index]
