import unittest
from mep.path import Path, Image
import numpy as np
from pymatgen import Structure, Molecule, Lattice


class TestPath(unittest.TestCase):
    def test_image(self):
        s1 = Structure(Lattice.cubic(3.61), ['Mo', 'Mo'], [[0., 0., 0.], [0.5, 0.5, 0.5]])
        s2 = Structure(Lattice.cubic(3.61), ['Mo', 'Mo'], [[0., 0., 0.], [0.6, 0.6, 0.5]])
        mol1 = Molecule(['C', 'O', 'O'], [[0, 0, 0], [1, 0, 0], [-1, 0, 0]])
        mol2 = Molecule(['C', 'O', 'O'], [[0, 0, 0], [2, 0, 0], [-2, 0, 0]])

        image1 = Image(s1)
        image2 = Image(s2)
        interps = image1.interpolate(image2)
        self.assertTrue(np.linalg.norm(interps[-2].struct_or_mol.cart_coords - interps[-2].data) < 1e-3)
        self.assertTrue(isinstance(interps[1].struct_or_mol, Structure))

        image1 = Image(mol1)
        image2 = Image(mol2)
        interps = image1.interpolate(image2)
        self.assertTrue(np.linalg.norm(interps[-2].struct_or_mol.cart_coords - interps[-2].data) < 1e-3)
        self.assertTrue(isinstance(interps[1].struct_or_mol, Molecule))

    def test_path(self):
        mep = Path([[1, 2], [3, 4], [5, 0]])
        energies = [1, 2, 3]
        self.assertEqual(mep.n_images, 3)
        self.assertEqual(mep.n_springs, 2)
        self.assertListEqual(mep._k, [-5]*2)
        self.assertAlmostEqual(mep.spring_forces[0] / -5, np.sqrt(20) - np.sqrt(8))
        tan = mep.get_unit_tangents(energies)[0]
        self.assertTrue(np.linalg.norm(tan - np.array([[0.4472136, -0.89442719]]))<1e-2)



if __name__ == "__main__":
    unittest.main()
