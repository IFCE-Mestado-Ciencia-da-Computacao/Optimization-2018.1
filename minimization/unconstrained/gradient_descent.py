from minimization.unconstrained.first_order_descent import FirstOrderDescent


class GradientDescent(FirstOrderDescent):

    def calculate_delta_x(self, f, x):
        return - f.gradient(x)
