from typing import List
import numpy as np
from mep.optimize import Optimizer, SGD
from mep.callbacks import BasicCallback, StopOnConvergence, CallbackList, History, logger, Callback
from mep.models import Model
from mep.path import Path


class NEB:
    """
    Nudged elastic band (NEB) methods, with support of climbing image NEB

    Args:
        model: mep.models.Model, model that expose predict_energy and predict_energy_and_forces methods
        path: mep.path.Path
        climbing: bool, turn on CI-NEB
        n_climbs: int, number of images that perform climbing
    """
    def __init__(self, model: Model, path: Path, climbing: bool=False, n_climbs: int=3):
        self.model = model
        self.path = path
        self.climbing = climbing
        self.n_climbs = n_climbs
        self.history = []
        self.stop = False
        start_energy = self.model.predict_energy(self.path.images[0])[0]
        end_energy = self.model.predict_energy(self.path.images[-1])[0]
        self.energies = [start_energy] + [0] * (len(self.path) - 2) + [end_energy]
        self.forces = [np.zeros_like(self.path[i].data) for i in range(len(self.path))]
        self.model_forces = [np.zeros_like(self.path[i].data) for i in range(len(self.path))]
        self.middle_tangents = [np.zeros_like(self.path[i].data) for i in range(len(self.path) - 2)]

    def update(self, climbing=False, indices=None):
        if climbing:
            self.get_ci_energies_forces(indices)
        else:
            self.get_neb_energies_forces()

    def get_neb_energies_forces(self):
        """
        Update the neb energies and forces

        """
        energies = []
        forces = []
        for i in self.path.inner_images:
            e, f = self.model.predict_energy_and_forces(i)
            energies.append(e)
            forces.append(f)
        self.model_forces = forces
        self.energies[1:-1] = energies
        self.middle_tangents = self.path.get_unit_tangents(self.energies)
        f1 = [i-np.sum(i*j)*j for i, j in zip(self.model_forces, self.middle_tangents)]
        f2 = [i*j for i, j in zip(self.path.spring_forces, self.middle_tangents)]
        self.forces = [i+j for i, j in zip(f1, f2)]

    def get_ci_energies_forces(self, indices: List[int]):
        """
        Update CI NEB energies and forces for images of indices
        Args:
            indices: list of integer, the indices for images that perform CI-NEB

        Returns:

        """
        self.get_neb_energies_forces()
        for i in indices:
            if i < 1 or i > len(self.path) - 2:
                raise ValueError('Index %d is at boundary' % i)
            self.forces[i] = self.model_forces[i] - 2 * np.sum(self.model_forces[i] * self.middle_tangents[i-1]) * self.middle_tangents[i-1]

    def run(self, callbacks: Callback=None, optimizer: Optimizer=SGD(0.02),
            n_steps: int=100, n_climb_steps: int=100, force_tol: float=0.1,
            verbose: bool=True) -> History:
        """
        Run the NEB/CINEB
        Args:
            callbacks: Callbacks to run during optimization
            optimizer: Optimizers for update the image coordinates
            n_steps: number of NEB runs
            n_climb_steps: number of CI-NEB runs
            force_tol: force tolerance for convergence check
            verbose: whether to output logger information

        Returns:
            History object that records the trajectory
        """
        callbacks = callbacks or CallbackList()
        history = History()
        if verbose:
            callbacks.append(BasicCallback())
        callbacks.append(StopOnConvergence(force_tol))
        callbacks.append(history)
        callbacks.set_model(self)
        optimizer.set_model(self)
        callbacks.opt_begin()
        _log_if_verbose("Total NEB steps %d" % n_steps, verbose)
        self._step(optimizer, callbacks, n_steps=n_steps, climbing=False)
        if self.climbing:
            # get indices with max energies
            _log_if_verbose("Climbing image begins", verbose)
            _log_if_verbose("Total CI-NEB steps %d" % n_climb_steps, verbose)
            zipped = [(i, j) for i, j in enumerate(self.energies)]
            sort = sorted(zipped, key=lambda x: x[1])[::-1]
            indices = [i[0] for i in sort[:self.n_climbs]]
            _log_if_verbose("The following image index will be climbing %s" % indices, verbose)
            self._step(optimizer, callbacks, n_steps=n_climb_steps, climbing=True, indices=indices)
        callbacks.opt_end()
        return history

    def _step(self, optimizer, callbacks, n_steps=100, climbing=False, indices=None):
        for i in range(n_steps):
            if self.stop:
                break
            self.update(climbing=climbing, indices=indices)
            callbacks.step_begin(i)
            update = optimizer.step()
            for k in range(1, self.path.n_images-1):
                self.path.images[k].move(update[k-1])
            callbacks.step_end(i)

    @property
    def energy_path(self):
        energies = np.array([i-self.energies[0] for i in self.energies]).ravel()
        distances = np.cumsum(self.path.image_distances).ravel() / np.sum(self.path.image_distances)
        return np.array([distances, energies]).T


def _log_if_verbose(message, verbose):
    if verbose:
        logger.info(message)
