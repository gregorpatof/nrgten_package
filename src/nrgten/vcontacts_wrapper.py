import os
import platform
import subprocess



def get_surface_dict(pdb_file, n_atoms):
    """ Returns a dict of dicts based on atom numbers (input file must have
        unique atom numbers), containing surface area in contact between
        pairs of atoms.
    """
    vcon_file = pdb_file + ".vcon"
    if not os.path.isfile(vcon_file):
        raise ValueError("This Vcon file does not exist: {}".format(vcon_file))
    return get_surface_dict_private(vcon_file, n_atoms)


def get_surface_dict_private(vcon_file, n_atoms):
    sd = dict()
    with open(vcon_file) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.startswith('#'):
            ll = line.strip().split('|')
            if len(ll) > 1:
                from_num = int(ll[0])
                SAS = float(ll[-1])
                if from_num in sd:
                    raise ValueError("Atom number {0} encountered twice by Vcontacts".format(from_num))
                sd[from_num] = dict()
                j = i + 1
                while True:
                    line = lines[j]
                    ll = line.strip().split('|')
                    if len(ll) <= 1:
                        break
                    to_num = int(ll[0])
                    surf = float(ll[-2])
                    sd[from_num][to_num] = surf
                    j += 1
                i = j
            else:
                i += 1
        else:
            i += 1
    if len(sd) != n_atoms:
        raise ValueError("There were {} atoms in the PDB file but {} atoms with identified contacts".format(
            n_atoms, len(sd)))
    return sd

