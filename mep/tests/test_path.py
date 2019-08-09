import unittest
from mep.path import Path, Image, validate_structures
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

        image1.link_next(image2)
        self.assertTrue(image2.prev == image1)

        image3 = Image(image1)
        self.assertTrue(np.linalg.norm(np.array(image3.data) - np.array(image1.data))<1e-3)

        image4 = Image([1, 2])
        image4.update([3, 4])
        self.assertListEqual(image4.data, [3, 4])
        self.assertTrue(str(image4).startswith("Image with"))

        image4 = Image([1, 2])
        image5 = Image([3, 4])
        all_images = image4.interpolate(image5, 3)
        self.assertEqual(len(all_images), 3)
        self.assertEqual(str(image4), repr(image4))
        image4.struct_or_mol = 1
        self.assertEqual(image4.struct_or_mol, 1)

    def test_path(self):
        path = Path([[1, 2], [3, 4], [5, 0]])
        energies = [1, 2, 3]
        self.assertEqual(path.n_images, 3)
        self.assertEqual(path.n_springs, 2)
        self.assertListEqual(path._k, [-5]*2)
        self.assertAlmostEqual(path.spring_forces[0] / -5, np.sqrt(20) - np.sqrt(8))
        tan = path.get_unit_tangents(energies)[0]
        self.assertTrue(np.linalg.norm(tan - np.array([[0.4472136, -0.89442719]]))<1e-2)
        with self.assertRaises(ValueError):
            Path([1, 2], k=[1, 2, 3])

        self.assertTrue(str(path).startswith('Path with 3 images:'))
        self.assertTrue(repr(path) == str(path))
        self.assertListEqual(path.k, [-5] * 2)

    def test_validate(self):
        s1 = Structure(Lattice.cubic(3.61), ['Mo', 'Mo'], [[0., 0., 0.], [0.5, 0.5, 0.5]])
        s2 = Structure(Lattice.cubic(3.61), ['Mo', 'Mo'], [[0., 0., 0.], [0.6, 0.6, 0.5]])
        s3 = Structure(Lattice.cubic(3.1), ['Mo', 'Mo'], [[0., 0., 0.], [0.6, 0.6, 0.5]])

        self.assertTrue(validate_structures([s1, s2]))
        self.assertFalse(validate_structures([s1, s3]))


if __name__ == "__main__":
    unittest.main()
