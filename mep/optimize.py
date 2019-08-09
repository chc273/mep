from scipy.optimize import minimize
from monty.json import MSONable
from abc import abstractmethod


class ScipyOptimizer:
    def __init__(self, model):
        self.model = model

    def minimize(self, x0, bounds=None, method=None, **kwargs):
        func = lambda x: self.model.predict_energy(x)[0]
        return minimize(func, x0, method=method, bounds=bounds, **kwargs)


class Optimizer(MSONable):
    def __init__(self):
        self.model = None

    @abstractmethod
    def step(self):
        """
        Update the coordinates movement vector

        """
        pass

    def set_model(self, model):
        self.model = model


class SGD(Optimizer):

    def __init__(self, alpha):
        self.alpha = alpha
        super().__init__()

    def step(self):
        forces = self.model.forces
        return [self.alpha * i for i in forces]
