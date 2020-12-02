from nrgten.atom import Atom
from nrgten.residue_list import ResidueList


def parse_pdb(pdb_file, atypes_dict):
    with open(pdb_file) as f:
        lines = f.readlines()
    return parse_pdb_from_lines(lines, atypes_dict)


def parse_pdb_from_lines(lines, atypes_dict, ignore_hetatms=False):
    resis = ResidueList(atypes_dict)
    for line in lines:
        if line[:4] == 'ATOM' or (not ignore_hetatms and line[:6] == 'HETATM'):
            resis.add(Atom(line))
    return resis.get_ordered_atoms(), resis.get_resilist(), resis.alt_flag
