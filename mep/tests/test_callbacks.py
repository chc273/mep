import unittest
from mep.callbacks import CallbackList, Callback, BasicCallback, History, StopOnConvergence


class DummyPath:
    def __init__(self, coords=[1, 1, 1]):
        self.coords = coords


class DummyModel:
    def __init__(self, forces=[[1, 2, 3]]):
        self.forces = forces
        self.path = DummyPath(forces[0])


class TestCallback(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = DummyModel()

    def test_callback(self):
        callback = Callback()
        callback.set_model(1)
        self.assertTrue(callback.model == 1)

    def test_callbacklist(self):
        c1 = Callback()
        callbacklist = CallbackList([c1, c1])
        callbacklist.append(c1)
        self.assertEqual(c1, callbacklist.callbacks[-1])
        callbacklist.set_model(1)
        self.assertEqual(c1.model, 1)

    def test_basiccallback(self):
        basic = BasicCallback()
        basic.set_model(self.model)
        with self.assertLogs() as cm:
            basic.opt_begin()
            basic.opt_end()
            basic.step_end(0)
        self.assertEqual(cm.output,
                         ['INFO:mep.callbacks:NEB run begins',
                          'INFO:mep.callbacks:NEB run finishs',
                          'INFO:mep.callbacks:Step 1, the NEB force norm is 3.742'])

    def test_history(self):
        history = History()
        history.set_model(self.model)
        history.step_begin(0)
        history.opt_end()
        self.assertEqual(len(history.history), 2)
        self.assertListEqual(history.history[0], [1, 2, 3])

    def test_stoponconvergence(self):
        stop = StopOnConvergence()
        stop.set_model(DummyModel())
        stop.model.forces = [[0.001, 0.001], [0.002, 0.001]]
        stop.step_end(1)
        self.assertTrue(stop.model.stop)


if __name__ == "__main__":
    unittest.main()
