import unittest
from mep.optimize import ScipyOptimizer
from mep.models import LEPS
import numpy as np


class TestOpt(unittest.TestCase):
    def test_leps(self):
        leps = LEPS()
        sp = ScipyOptimizer(leps)
        x = sp.minimize(x0=[1, 3], bounds=[[0, 4], [0, 4]]).x
        self.assertEqual(sp.model, leps)
        self.assertTrue(np.linalg.norm(np.array(x) - np.array([0.74202011, 4.])) < 1e-2)


if __name__ == "__main__":
    unittest.main()
