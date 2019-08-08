from typing import List
from pymatgen import Structure
from mep.models import Model


class NEB:
    """
    Nudged elastic band for anything

    Args:
        model (object)
    """

    def __init__(self, structures: List[Structure], model=None) -> None:
        self.model = model
        self.structures = structures

    def _compute_forces(self, structure):
        return self.model.predict_force(structure).reshape((-1, 3))


