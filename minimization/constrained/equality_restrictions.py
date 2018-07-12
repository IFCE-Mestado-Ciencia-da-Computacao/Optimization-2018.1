from abc import ABCMeta


class LinearRestrictions(metaclass=ABCMeta):
    """
    :param A
    :param b
    """
    def __call__(self, *args, **kwargs):
        pass
