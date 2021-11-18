from nrgten.atom import Atom
import numpy as np

def generate_massfile(pdb_filename, mass_filename):
    atoms = []
    with open(pdb_filename) as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith('ATOM') or line.startswith("HETATM"):
            atoms.append(Atom(line))
    xyz_data = np.zeros((len(atoms), 3))
    for i, atom in enumerate(atoms):
        xyz_data[i] = atom.xyz
    centroid = np.array([np.mean(xyz_data[:, 0]), np.mean(xyz_data[:, 1]), np.mean(xyz_data[:, 2])])
    medoid = None
    mindist = float('Inf')
    other_atom = None
    other_flag = True
    for atom in atoms:
        dist = np.linalg.norm(atom.xyz - centroid)
        if dist < mindist:
            mindist = dist
            medoid = atom
        elif other_flag:
            other_atom = atom
            other_flag = False
    if other_flag:
        other_atom = atoms[0]
    with open(mass_filename, "w") as f:
        f.write("CONNECT: {} -> {}\n".format(medoid.name, other_atom.name))
        f.write("N_MASSES: 1\n")
        f.write("CENTER: {}\n".format(medoid.name))
        f.write("NAME: {}\n".format(medoid.name))
