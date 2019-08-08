from pymatgen import Molecule
import numpy as np
import collections


def interpolate_molecule(mol: Molecule, other_mol: Molecule, n: int=5, autosort_tol: float=0):
    """
    Interpolate molecule from mol to other_mol, with a total of n images. The algorithm is from
    pymatgen  https://pymatgen.org/_modules/pymatgen/core/structure.html#IStructure.interpolate

    Args:
        mol: Molecule, start molecule
        other_mol: Molecule, end molecule
        n: number of total images
        autosort_tol: float, the tolerance for sorting the sites to automatically match

    Returns:

    """
    if len(mol) != len(other_mol):
        raise ValueError("Molecule has different length")
    for i, j in zip(mol.sites, other_mol.sites):
        if i.species != j.species:
            raise ValueError("Species do not match!")
    start_coords = mol.cart_coords
    end_coords = other_mol.cart_coords
    if autosort_tol:
        dist_matrix = np.linalg.norm(start_coords[:, None, :] - end_coords[None, :, :], axis=-1)
        mappings = collections.defaultdict(list)
        unmapped = []
        for i, row in enumerate(dist_matrix):
            ind = np.where(row < autosort_tol)[0]
            if len(ind) == 1:
                mappings[i].append(ind[0])
            else:
                unmapped.append(i)
        if len(unmapped) > 1:
            raise ValueError("Failed to match structure with autosort_tol: %.3f, unmapped indices are %s" %
                             (autosort_tol, unmapped))
        sorted_end_coords = np.zeros_like(end_coords)
        matched = []
        for i, j in mappings.items():
            if len(j) > 1:
                raise ValueError("Autosort_tol = %.3f may be too large, causing multiple mapping from start"
                                 "structure %d to end structure %s" % (autosort_tol, i, j))
            sorted_end_coords[i] = end_coords[j[0]]
            matched.append(j[0])

        # Attach the displaced atom to the coords
        if len(unmapped) == 1:
            i = unmapped[0]
            j = list(set(range(len(start_coords))).difference(matched))[0]
            sorted_end_coords[i] = end_coords[j]

        end_coords = sorted_end_coords

    vec = end_coords - start_coords
    sp = mol.species_and_occu

    molecules = []
    for x in range(n):
        molecules.append(Molecule(sp, start_coords + vec * x /(n-1), site_properties=mol.site_properties))
    return molecules


def plot_energy_path(distances, energies, unit='eV'):
    """
    plot energy path
    Args:
        distances:
        energies:
        unit:

    Returns:

    """
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 22
    plt.figure(figsize=(6, 5.5))
    max_energy = max(energies)
    min_energy = min(energies)
    max_ind = np.argmax(energies)
    energy_range = max_energy - min_energy
    plt.plot(distances, energies, 'o-', markerfacecolor='w')
    plt.xlim([0, 1])
    plt.ylim([min_energy, max_energy + 0.1 * energy_range])
    plt.plot([0, distances[max_ind]], [max_energy, max_energy], '--')
    plt.text(distances[max_ind], max_energy, '$E_a=%.2f$ %s' % (max_energy, unit))
    plt.xlabel('Reaction Coordinates')
    plt.ylabel('$E$ (%s)' % unit)
    return plt
