import collections

accepted_from_to = set(['(n)', '(n+1)', '(n-1)'])


class MassDef:
    ''' Class used to represent the definition of masses in a given
        (macro)molecule.
    '''

    def __init__(self, massdef_file):
        self.connect = None
        self.connect_init = None
        self.redefine_resis = None
        self.n_masses = None
        self.centers = None
        self.mass_names = None
        self.mass_atom_sets = None
        self.name = None
        self.parse_massdef(massdef_file)
        if self.connect_init is None:
            self.connect_init = self.connect

    def get_atoms_center_name(self, i):
        assert self.centers is not None and self.mass_names is not None and self.mass_atom_sets is not None
        assert len(self.centers) == len(self.mass_names) and len(self.centers) == len(self.mass_atom_sets)
        assert i < len(self.centers) and i >= 0
        return (self.mass_atom_sets[i], self.centers[i], self.mass_names[i])

    def parse_massdef(self, massdef_file):
        with open(massdef_file) as f:
            lines = f.readlines()
        i = 0
        self.name = massdef_file.split('/')[-1].split('.')[0]
        while True:
            if i >= len(lines):
                break
            line = lines[i]
            ll = line.strip().split(':')
            tag = ll[0].strip()
            rest = None
            if len(ll) >= 2:
                rest = ll[1].strip()

            if tag == "CONNECT":
                assert self.connect is None
                rl = rest.split('->')
                self.connect = (rl[0].strip(), rl[1].strip())

            elif tag == "CONNECT_INIT":
                assert self.connect_init is None
                rl = rest.split('->')
                self.connect_init = (rl[0].strip(), rl[1].strip())

            elif tag == "REDEFINE_RESIS":
                assert self.redefine_resis is None
                if self.connect_init is None:
                    raise ValueError("Trying to redefine residues without a CONNECT_INIT line")
                rl = rest.split('|')
                from_to = rl[0].strip()
                ftl = from_to.split('->')
                self.validate_ftl(ftl)
                fr = self.format_from_to(ftl[0])
                to = self.format_from_to(ftl[1])
                resilist = [x.strip() for x in rl[1].strip().split(',')]
                direction = to - fr
                self.redefine_resis = (direction, resilist)

            elif tag == "N_MASSES":
                assert self.n_masses is None
                self.n_masses = int(rest)

            elif tag == "CENTER":
                assert self.n_masses == 1
                self.centers = [rest.strip()]

            elif tag == "NAME":
                assert self.n_masses == 1
                assert self.mass_names is None
                self.mass_names = [rest.strip()]

            elif tag[:5] == "MASS_":
                n = int(tag.split('_')[-1])
                if n == 1:
                    assert self.centers is None
                    assert self.mass_names is None
                    assert self.mass_atom_sets is None
                    self.centers = []
                    self.mass_names = []
                    self.mass_atom_sets = []
                assert len(self.centers) == n - 1
                (name, atoms, center) = self.parse_one_mass(lines, i)
                i += 3
                self.centers.append(center)
                self.mass_names.append(name)
                self.mass_atom_sets.append(set(atoms))
            i += 1

    def parse_one_mass(self, lines, i):
        assert len(lines) > i + 3
        three_lines = lines[i + 1:i + 4]
        (name, atoms, center) = (None, None, None)
        for line in three_lines:
            ll = line.strip().split(':')
            tag = ll[0].strip()
            rest = None
            if len(ll) >= 2:
                rest = ll[1].strip()

            if tag == "NAME":
                assert name is None
                name = rest.strip()

            elif tag == "ATOMS":
                assert atoms is None
                atoms = [x.strip() for x in rest.split(',')]

            elif tag == "CENTER":
                assert center is None
                center = rest.strip()
        return (name, atoms, center)

    def format_from_to(self, s):
        s = s.strip()[2:-1]
        if len(s) > 0:
            s = int(s)
        else:
            s = 0
        return s

    def validate_ftl(self, ftl):
        ''' Used to ensure that the from -> to nomenclature used for residue
            redifinition is valid.
        '''
        for x in ftl:
            x = x.strip()
            if x not in accepted_from_to:
                raise ValueError("Error in REDEFINE_RESIS line : {0} is not accepted".format(x))


def massdefs_dicts_are_same(mdd1, mdd2):
    assert isinstance(mdd1, collections.Mapping) and isinstance(mdd2, collections.Mapping)
    if mdd1.keys() != mdd2.keys():
        return False
    for k in mdd1.keys():
        if not massdefs_are_same(mdd1[k], mdd2[k]):
            return False
    return True

def massdefs_are_same(md1, md2):
    assert isinstance(md1, MassDef) and isinstance(md2, MassDef)
    if not (md1.connect == md2.connect and
            md1.connect_init == md2.connect_init and
            md1.redefine_resis == md2.redefine_resis and
            md1.n_masses == md2.n_masses and
            md1.centers == md2.centers and
            md1.mass_names == md2.mass_names and
            md1.mass_atom_sets == md2.mass_atom_sets):
        return False
    return True