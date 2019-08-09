import unittest
from mep.models import LEPS, LEPSHarm


class TestModel(unittest.TestCase):
    def test_leps(self):
        leps = LEPS()
        vs = leps.predict_energy([1, 2])
        self.assertAlmostEqual(vs[0], -3.72904677)
        _, vsf = leps.predict_energy_and_forces([1, 2])
        self.assertAlmostEqual(vsf[0, 0], -4.127233908546302)

    def test_leps_harm(self):
        leps_harm = LEPSHarm()
        self.assertAlmostEqual(leps_harm.predict_energy([1, 2])[0], -3.500745277)
        self.assertAlmostEqual(leps_harm.predict_energy_and_forces([1, 2])[1][0][0], -4.9164392648126665)


if __name__ == "__main__":
    unittest.main()
