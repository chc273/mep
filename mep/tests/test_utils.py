import unittest
from pymatgen import Molecule
from mep.utils import interpolate_molecule
import numpy as np


class TestUtils(unittest.TestCase):
    def test_interpolate_molecule(self):
        mol1 = Molecule(['C', 'O', 'O'], [[0, 0, 0], [1, 0, 0], [-1, 0, 0]])
        mol2 = Molecule(['C', 'O', 'O'], [[0, 0, 0], [2, 0, 0], [-2, 0, 0]])
        interpolated = interpolate_molecule(mol1, mol2, 5)
        self.assertTrue(np.linalg.norm(interpolated[2].cart_coords -
                                       np.array([[0, 0, 0], [1.5, 0, 0], [-1.5, 0., 0.]])) < 1e-2)

        mol1 = Molecule(['C', 'O', 'O'], [[0, 0, 0], [1, 0, 0], [-1, 0, 0]])
        mol2 = Molecule(['C', 'O', 'O'], [[0, 0, 0], [-1, 0, 0], [-2, 0, 0]])
        interpolated = interpolate_molecule(mol1, mol2, 4, autosort_tol=0.1)
        self.assertTrue(np.linalg.norm(interpolated[1].cart_coords -
                                       np.array([[0, 0, 0], [0, 0, 0], [-1, 0., 0.]])) < 1e-2)


if __name__ == "__main__":
    unittest.main()
