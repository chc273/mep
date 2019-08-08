import unittest
from mep.models import LEPS, LEPSHarm


class TestModel(unittest.TestCase):
    def test_leps(self):
        leps = LEPS()
        vs = leps.predict_energy([1, 2])
        self.assertAlmostEqual(vs[0], -3.72904677)
        vsf = leps.predict_force_diff([1, 2])
        self.assertAlmostEqual(vsf[0, 0], -4.14788987)

    def test_leps_harm(self):
        leps_harm = LEPSHarm()
        self.assertAlmostEqual(leps_harm.predict_energy([1, 2])[0], -3.500745277)
        self.assertAlmostEqual(leps_harm.predict_force_diff([1, 2])[0][0], -4.941616207514121)


if __name__ == "__main__":
    unittest.main()
