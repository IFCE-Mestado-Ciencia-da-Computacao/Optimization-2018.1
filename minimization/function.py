from abc import ABCMeta, abstractmethod


class Function(metaclass=ABCMeta):
    def __call__(self, *args, **kwargs):
        pass

    @abstractmethod
    def gradient(self, x):
        pass

    def hessian(self, x):
        pass
