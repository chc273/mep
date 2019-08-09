import logging
import numpy as np
from copy import deepcopy

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Callback:
    def __init__(self):
        self.model = None

    def opt_begin(self):
        pass

    def opt_end(self):
        pass

    def step_begin(self, n):
        pass

    def step_end(self, n):
        pass

    def set_model(self, model):
        self.model = model


class BasicCallback(Callback):

    def opt_begin(self):
        logger.info("NEB run begins")

    def opt_end(self):
        logger.info("NEB run finishes")

    def step_end(self, n):
        forces = self.model.forces
        logger.info("Step %d, the NEB force norm is %.3f" % (n+1, np.linalg.norm(np.concatenate(forces, axis=0))))


class History(Callback):
    def __init__(self):
        self.history = []
        super().__init__()

    def step_begin(self, n):
        path = self.model.path
        self.history.append(deepcopy(path.coords))

    def opt_end(self):
        path = self.model.path
        self.history.append(deepcopy(path.coords))


class StopOnConvergence(Callback):
    def __init__(self, force_tol=1e-1):
        self.force_tol = force_tol
        super().__init__()

    def step_end(self, n):
        forces = self.model.forces
        norm = np.linalg.norm(np.concatenate(forces, axis=0))
        if norm < self.force_tol:
            self.model.stop = True
            logger.info("Optimization stopped with convergence")


class CallbackList:
    def __init__(self, callbacks=None, queue_length=10):
        callbacks = callbacks or []
        self.callbacks = [c for c in callbacks]
        self.queue_length = queue_length

    def append(self, callback):
        self.callbacks.append(callback)

    def opt_begin(self):
        for callback in self.callbacks:
            callback.opt_begin()

    def opt_end(self):
        for callback in self.callbacks:
            callback.opt_end()

    def step_begin(self, n):
        for callback in self.callbacks:
            callback.step_begin(n)

    def step_end(self, n):
        for callback in self.callbacks:
            callback.step_end(n)

    def set_model(self, model):
        for callback in self.callbacks:
            callback.set_model(model)
