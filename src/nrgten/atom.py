import numpy as np

class Atom:
    __slots__ = ['num', 'name', 'alt', 'resiname', 'chain', 'resinum', 'insert',
                 'xyz', 'occ', 'bfact', 'seg_id', 'elem', 'charge', 'parent_resi', 'macromol_index']

    def __init__(self, line):
        n = len(line)
        if n < 54:
            raise ValueError("Trying to create Atom with line shorter than 54")
        self.parent_resi = None
        self.macromol_index = None
        self.seg_id = None
        self.num = int(line[6:11])
        self.name = line[12:16].strip()
        self.alt = line[16]
        self.resiname = line[17:20].strip()
        self.chain = line[21]
        self.resinum = int(line[22:26])
        self.insert = line[26]
        self.xyz = np.array([float(line[30:38]),  float(line[38:46]), float(line[46:54])])
        # placeholders for short lines
        self.occ = 0
        self.bfact = 0
        self.elem = "  "
        self.charge = "  "
        try:
            if n >= 80:
                self.charge = line[78:80]
                self.elem = line[76:78]
                self.bfact = self.get_float(line[60:66])
                self.occ = self.get_float(line[54:60])
            elif n >= 78:
                self.elem = line[76:78]
                self.bfact = self.get_float(line[60:66])
                self.occ = self.get_float(line[54:60])
            elif n >= 66:
                self.bfact = self.get_float(line[60:66])
                self.occ = self.get_float(line[54:60])
            elif n >= 60:
                self.occ = self.get_float(line[54:60])
        except ValueError as e:
            raise

    def get_float(self, s):
        """ Returns 0 for empty strings, float(s) otherwise.
        """
        if len(s.strip()) == 0:
            return 0.0
        return float(s)


    def __getitem__(self, i):
        if i == 0:
            return self.x
        elif i == 1:
            return self.y
        elif i == 2:
            return self.z
        else:
            raise ValueError("Trying to index Atom object at index other than 0, 1 or 2 (x, y, z)")

    def translate(self, vec):
        assert len(vec) == 3
        self.xyz += np.array(vec)

    def set_xyz(self, xyz):
        self.xyz = np.array([xyz[0], xyz[1], xyz[2]])

    def set_parent_resi(self, resi):
        self.parent_resi = resi

    def set_macromol_index(self, i):
        self.macromol_index = i

    def is_equal(self, other):
        assert isinstance(other, Atom)
        a=self.num == other.num
        b=self.name == other.name
        c=self.alt == other.alt
        d=self.resiname == other.resiname
        e=self.chain == other.chain
        f=self.resinum == other.resinum
        g=self.insert == other.insert
        h=np.array_equal(self.xyz, other.xyz)
        i=self.occ == other.occ
        j=self.bfact == other.bfact
        k=self.seg_id == other.seg_id
        l=self.elem == other.elem
        m=self.charge == other.charge
        n=self.parent_resi == other.parent_resi
        o=self.macromol_index == other.macromol_index
        if not (self.num == other.num and
                self.name == other.name and
                self.alt == other.alt and
                self.resiname == other.resiname and
                self.chain == other.chain and
                self.resinum == other.resinum and
                self.insert == other.insert and
                np.array_equal(self.xyz, other.xyz) and
                self.occ == other.occ and
                self.bfact == other.bfact and
                self.seg_id == other.seg_id and
                self.elem == other.elem and
                self.charge == other.charge and
                self.parent_resi == other.parent_resi and
                self.macromol_index == other.macromol_index):
            return False
        return True

    def __str__(self):
        return self.toString()

    def __repr__(self):
        return self.toString()

    def toString(self):
        x, y, z = self.xyz
        spaces10 = "          "
        if len(self.name) <= 3:
            return "ATOM  {:>5}  {:3}{}{:>3} {}{:>4}    {:>8.3f}{:>8.3f}{:>8.3f}{:>6.2f}{:>6.2f}{}{}{}".format(
                self.num, self.name, self.alt, self.resiname, self.chain, self.resinum, x, y, z,
                self.occ, self.bfact, spaces10, self.elem, self.charge)
        # difference is the added space for atom name (weird "center-alignment" in PDB)
        return "ATOM  {:>5} {:4}{}{:>3} {}{:>4}    {:>8.3f}{:>8.3f}{:>8.3f}{:>6.2f}{:>6.2f}{}{}{}".format(
            self.num, self.name, self.alt, self.resiname, self.chain, self.resinum, x, y, z,
            self.occ, self.bfact, spaces10, self.elem, self.charge)
