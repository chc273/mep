from __future__ import annotations
from typing import List, Union
from collections import Sized
from pymatgen import Structure, Molecule, Site
import numpy as np
from mep.utils import interpolate_molecule


class Node:
    def __init__(self, data=None):
        self.data = data
        self.prev = None
        self.next = None

    def link_next(self, n):
        self.next = n
        self.next.prev = self

    def link_prev(self, prev):
        self.prev = prev
        self.prev.next = self


class Image(Node):
    """
    Intermediate image, can be a structure, molecule or general coordinates
    """
    def __init__(self, image: Union[Structure, Molecule, List, Image]):
        if isinstance(image, (Structure, Molecule)):
            self._struct_or_mol = image.copy()
            self.type = str(image.__class__)
            data = self._struct_or_mol.cart_coords
        elif isinstance(image, Image):
            self.struct_or_mol = image.struct_or_mol
            self.type = image.type
            data = image.data
        else:
            self.structure_or_mol = None
            self.type = 'Simple'
            data = np.atleast_2d(image)
        super().__init__(data)

    def move(self, direction: np.ndarray) -> None:
        """
        Move the image by direction
        Args:
            direction: numpy array, the direction vector
        """
        self.data += direction

    def update(self, new_data: np.ndarray) -> None:
        """
        Update the coordinates
        Args:
            new_data:

        Returns:

        """
        self.data = new_data

    def interpolate(self, other: Image, n: int=5) -> List:
        """
        Interpolate other image
        Args:
            other: other image
            n: number of total images
        Returns: a list of images
        """
        if isinstance(self.struct_or_mol, Structure):
            structures = self.struct_or_mol.interpolate(other.struct_or_mol, n)
            return [Image(i) for i in structures]
        elif isinstance(self.struct_or_mol, Molecule):
            return [Image(i) for i in interpolate_molecule(self.struct_or_mol, other.struct_or_mol, n)]
        else:
            vec = other.data - self.data
            return [Image(self.data + vec * i / (n-1)) for i in range(n)]

    def __str__(self):
        return "Image with %s type, and data %s" % (self.type, self.data)

    def __repr__(self):
        return str(self)

    @property
    def struct_or_mol(self):
        if isinstance(self._struct_or_mol, Molecule):
            for i, site in enumerate(self._struct_or_mol.sites):
                self._struct_or_mol._sites[i] = Site(site.species, self.data[i].ravel(),
                                                     properties=site.properties)
        elif isinstance(self._struct_or_mol, Structure):
            for i, site in enumerate(self._struct_or_mol.sites):
                fcoords = self._struct_or_mol._lattice.get_fractional_coords(self.data[i].ravel())
                self._struct_or_mol._sites[i].frac_coords = fcoords
        else:
            pass
        return self._struct_or_mol

    @struct_or_mol.setter
    def struct_or_mol(self, new_s):
        self._struct_or_mol = new_s

    def copy(self):
        return Image(self)


class Spring(Node):
    pass


class Path:
    def __init__(self, images, k=-5):
        self.images = [Image(i) for i in images]
        self.n_images = len(images)
        self.n_springs = self.n_images - 1
        if isinstance(k, Sized):
            if len(k) != self.n_springs:
                raise ValueError("The number spring constants should be 1 less than images")
            self._k = k
        else:
            self._k = [k] * self.n_springs
        self.springs = [Spring(i) for i in self._k]
        self.construct_path()
        self.inner_images = self.images[1:-1]
        self.n_inner = len(self.inner_images)

    def __str__(self):
        line = "Path with %d images: " % self.n_images
        for i in range(self.n_images):
            line += str(self.images[i])
            line += '\n'
        return line

    def __repr__(self):
        return str(self)

    def __len__(self):
        return self.n_images

    def __getitem__(self, index):
        return self.images[index]

    def construct_path(self):
        for i in range(len(self.springs)):
            self.images[i].link_next(self.springs[i])
            self.springs[i].link_next(self.images[i+1])

    def move_images(self, forces, delta=0.1):
        if len(forces) != self.n_inner:
            raise ValueError('Too many forces, remember the ends are fixed')
        for i, j in zip(self.inner_images, forces):
            i.move(j * delta)

    @property
    def image_distances(self):
        return [0] + [np.linalg.norm(i-j) for i, j in zip(self.coords[:-1], self.coords[1:])]

    def get_unit_tangents(self, energies=None):
        return [self.get_unit_tangent(i, energies) for i in range(1, self.n_images - 1)]

    def get_unit_tangent(self, i: int, energies: List[float]=None):
        """
        As described in Henkelmana et al. Journal of Chemical Physics
        https://aip.scitation.org/doi/pdf/10.1063/1.1323224?class=pdf

        Args:
            i: index of the image
            energies: list of energies for all images

        Returns:
        """
        if i < 1 or i > self.n_images - 2:
            raise ValueError('Only internal images can be calculated')
        vs = energies[(i - 1):(i + 2)]
        coords = [self.images[k].data for k in [i - 1, i, i + 1]]
        tau_plus = coords[-1] - coords[1]
        tau_minus = coords[1] - coords[0]
        dv_plus = vs[2] - vs[1]
        dv_minus = vs[1] - vs[0]

        if (dv_plus > 0) and (dv_minus > 0):
            tau = tau_plus
        elif (dv_plus < 0) and (dv_minus < 0):
            tau = tau_minus
        else:
            dv_max = max(abs(dv_plus), abs(dv_minus))
            dv_min = min(abs(dv_plus), abs(dv_minus))
            if vs[2] - vs[0] > 0:
                tau = tau_plus * dv_max + tau_minus * dv_min
            else:
                tau = tau_plus * dv_min + tau_minus * dv_max
        return tau / np.linalg.norm(tau)

    @property
    def coords(self):
        return [i.data for i in self.images]

    @property
    def inner_coords(self):
        return [i.data for i in self.inner_images]

    @property
    def spring_forces(self):
        _spring_forces = []
        for k, i in enumerate(self.inner_images, start=1):
            _spring_forces.append(i.prev.data * np.linalg.norm(self.images[k+1].data - i.data) -
                                  i.next.data * np.linalg.norm(i.data - self.images[k-1].data))
        return _spring_forces

    @classmethod
    def from_linear_end_points(cls, image_start, image_end, n, k=-5):
        dimage = np.array(image_end) - np.array(image_start)
        dximage = dimage / (n-1)
        return cls([np.array(image_start) + dximage * i for i in range(n)], k=k)

    @property
    def k(self):
        return self._k

    @k.setter
    def k(self, new_k):
        self._k = new_k


def _validate(structures: List[Structure]):
    """
    Validate the structures so that all of them should have the same lattice
    Args:
        structures: list of pymatgen structures

    Returns:

    """
    lattices = [i.lattice for i in structures]
    if len(set(lattices)) > 1:
        raise ValueError('Structures have more than one lattices!')
