import numpy as np
from mep.optimize import SGD
from mep.callbacks import BasicCallback, StopOnConvergence, CallbackList, History


class NEB:
    def __init__(self, model, path):
        self.model = model
        self.path = path
        self.history = []
        self.stop = False
        self.energies = [0] * len(self.path)
        self.forces = [np.zeros_like(self.path[i].data) for i in range(len(self.path))]

    def update(self):
        self.energies, self.forces = self.get_neb_energies_forces()

    def get_neb_energies_forces(self):
        coords = self.path.inner_coords
        forces = [self.model.predict_force(i) for i in coords]
        energies = [self.model.predict_energy(i.data) for i in self.path.images]
        middle_tangents = self.path.get_unit_tangents(energies)
        f1 = [i-np.sum(i*j)*j for i, j in zip(forces, middle_tangents)]
        f2 = [i*j for i, j in zip(self.path.spring_forces, middle_tangents)]
        return energies, [i+j for i, j in zip(f1, f2)]

    def run(self, callbacks=None, optimizer=SGD(0.02), n_steps=100, force_tol=0.1, verbose=True):
        callbacks = callbacks or CallbackList()
        history = History()
        if verbose:
            callbacks.append(BasicCallback())
        callbacks.append(StopOnConvergence(force_tol))
        callbacks.append(history)
        callbacks.set_model(self)
        optimizer.set_model(self)
        callbacks.opt_begin()
        for i in range(n_steps):
            if self.stop:
                break
            self.update()
            callbacks.step_begin(i)
            update = optimizer.step()
            for k in range(1, self.path.n_images-1):
                self.path.images[k].move(update[k-1])
            callbacks.step_end(i)
        callbacks.opt_end()
        return history

    @property
    def energy_path(self):
        energies = np.array([i-self.energies[0][0] for i in self.energies]).ravel()
        distances = np.cumsum(self.path.image_distances).ravel() / np.sum(self.path.image_distances)
        return np.array([distances, energies]).T
