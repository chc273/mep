import unittest
from mep.path import Path
from mep.neb import NEB
from mep.models import LEPS


class TestNEB(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        x0 = [0.74, 4]
        x1 = [4, 0.74]
        cls.path = Path.from_linear_end_points(x0, x1, 5, 1)
        cls.leps = LEPS()

    def test_neb(self):
        neb = NEB(self.leps, self.path)
        with self.assertLogs() as cm:
            history = neb.run(verbose=True, n_steps=1)
        self.assertEqual(cm.output[0], 'INFO:mep.callbacks:NEB run begins')
        self.assertEqual(cm.output[-1], 'INFO:mep.callbacks:NEB run finishes')
        self.assertEqual(len(history.history), 2)

    def test_cineb(self):
        neb = NEB(self.leps, self.path, climbing=True, n_climbs=1)
        with self.assertLogs() as cm:
            history = neb.run(verbose=True, n_steps=1, n_climb_steps=1)
        self.assertEqual(cm.output[0], 'INFO:mep.callbacks:NEB run begins')
        self.assertEqual(cm.output[3], 'INFO:mep.callbacks:Climbing image begins')
        self.assertEqual(cm.output[-1], 'INFO:mep.callbacks:NEB run finishes')
        self.assertEqual(len(history.history), 3)


if __name__ == "__main__":
    unittest.main()
