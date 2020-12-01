class ResidueList:
    """ Class used to represent the list of all kept residues from a macromol
    file (PDB or other), as a dict of dicts. This version keeps a single list
    for every residue type but needs a dict of atom types."""

    def __init__(self, atypes_dict):
        self.list = {}
        self.atypes_dict = atypes_dict
        self.lastkey = None
        self.resicount = 0
        self.atoms = {} # stores atoms according to their atom number and validates there is no repeat
        self.alt_flag = False

    def add(self, atom):
        if atom.name[0] == 'H': # Hydrogen atoms are not considered
            return
        if self.atypes_dict is None:
            self.add_pvt(atom)
        elif atom.resiname in self.atypes_dict:
            self.add_pvt(atom)
        else:
            raise ValueError("Trying to add atom with unsupported residue type: \n{0}\n".format(atom.toString()))

    def add_pvt(self, atom):
        if atom.alt != ' ' and atom.alt != 'A': # treats alternate location at the chain level
            return
        if atom.insert == ' ':
            basekey = "{}|{}|{}".format(atom.resiname, atom.resinum, atom.chain)
        else:
            basekey = "{}|{}{}|{}".format(atom.resiname, atom.resinum, atom.insert, atom.chain)
        if basekey != self.lastkey:
            self.resicount += 1
        self.lastkey = basekey
        key = basekey + "|" + str(self.resicount)
        if key not in self.list:
            self.list[key] = {}
        d = self.list[key]
        if atom.name in d:  # treats cases with alternate locations at the residue level
            if atom.alt != ' ':
                self.alt_flag = True
                return
            print(d)
            print(atom.name)
            print(key)
            raise ValueError("Trying to add same atom twice!")
        if atom.num in self.atoms:
            raise ValueError("Atom number {0} found twice in pdb file!".format(atom.num))
        self.atoms[atom.num] = atom
        d[atom.name] = atom

    def get_resilist(self):
        return self.list

    def get_ordered_atoms(self):
        return [self.atoms[k] for k in sorted(self.atoms.keys())]
