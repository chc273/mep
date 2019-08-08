import numpy as np
from abc import abstractmethod
from copy import deepcopy


class Model:
    @abstractmethod
    def predict_energy(self, image):
        pass

    def predict_force(self, image, delta=0.0001):
        image = np.atleast_2d(image).astype(np.float)
        e0_all = [self.predict_energy(i)[0] for i in image]
        shapes = image.shape
        forces = []
        for i in range(shapes[0]):
            e0 = e0_all[i]
            force = []
            for j in range(shapes[1]):
                image_temp = deepcopy(image[i])
                image_temp[j] += delta
                e = self.predict_energy(image_temp)[0]
                force.append((e-e0)/delta)
            forces.append(np.array(force))
        return -np.array(forces)


class LEPS(Model):
    """
    Model 1 in
    https://www.worldscientific.com/doi/pdf/10.1142/9789812839664_0016

    """
    def __init__(self, a=0.05, b=0.30, c=0.05, dab=4.746, dbc=4.746, dac=3.445,
                 r0=0.742, alpha=1.942):
        self.a = a
        self.b = b
        self.c = c
        self.dab = dab
        self.dbc = dbc
        self.dac = dac
        self.r0 = r0
        self.alpha = alpha

    def predict_energy(self, image):
        image = np.atleast_2d(image)
        rab = image[:, 0]
        rbc = image[:, 1]
        rac = rab + rbc
        return self.v(rab, rbc, rac)

    def v(self, rab, rbc, rac):
        qab = self.q(self.dab, rab)
        qbc = self.q(self.dbc, rbc)
        qac = self.q(self.dac, rac)
        jab = self.j(self.dab, rab)
        jbc = self.j(self.dbc, rbc)
        jac = self.j(self.dac, rac)
        v = qab / (1 + self.a) + qbc / (1 + self.b) + qac / (1 + self.c) - \
            (jab**2 / (1 + self.a)**2 +
             jbc**2 / (1 + self.b)**2 +
             jac**2 / (1 + self.c)**2 -
             jab*jbc/(1 + self.a)/(1 + self.b) -
             jbc*jac/(1 + self.b)/(1 + self.c) -
             jab*jac/(1 + self.a)/(1 + self.c))**0.5
        return v

    def q(self, d, r):
        return 0.5 * d * (1.5 * np.exp(-2*self.alpha * (r - self.r0)) - np.exp(-self.alpha * (r - self.r0)))

    def j(self, d, r):
        return 0.25 * d * (np.exp(-2 * self.alpha * (r - self.r0)) - 6 * np.exp(-self.alpha * (r - self.r0)))


class LEPSHarm(LEPS):
    """
    model 2 in
    https://www.worldscientific.com/doi/pdf/10.1142/9789812839664_0016
    """
    def __init__(self, a=0.05, b=0.80, c=0.05, dab=4.746, dbc=4.746, dac=3.445, r0=0.742, alpha=1.942, rac=3.742,
                 kc=0.2025, cc=1.154):
        self.kc = kc
        self.rac = rac
        self.cc = cc
        super().__init__(a=a, b=b, c=c, dab=dab, dbc=dbc, dac=dac, r0=r0, alpha=alpha)

    def predict_energy(self, image):
        image = np.atleast_2d(image)
        rab = image[:, 0]
        x = image[:, 1]
        rbc = self.rac - rab
        return super().predict_energy(np.stack([rab, rbc], axis=1)) + \
            2 * self.kc * (rab - (self.rac/2 - x / self.cc)) ** 2


def _distance(x, y):
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    return np.linalg.norm(np.array(x[:, None, :]) - np.array(y[None, :, :]), axis=-1)
