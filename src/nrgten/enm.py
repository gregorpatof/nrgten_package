from nrgten.macromol import get_macromol_list
from nrgten.massdef import MassDef, massdefs_dicts_are_same
import numpy as np
import os
import abc


class ENM(metaclass=abc.ABCMeta):
    """ Abstract base class (ABC) for Elastic Network Models (ENMs).
    """
    def __init__(self, pdb_file, added_atypes=None, added_massdef=None, atypes_list=None, massdef_list=None,
                 verbose=False, solve=False, use_pickle=False, ignore_hetatms=False, solve_mol=True, one_mass=False):
        self.dirpath = os.path.dirname(os.path.abspath(__file__))
        assert len(self.dirpath) > 0
        self.verbose = verbose
        self.pickle_version = "no clearing of macromol"
        self.one_mass = one_mass
        if atypes_list is None:
            if self.one_mass:
                atomfiles = ["ribonucleic_acids_1n.atomtypes", "amino_acids.atomtypes"]
            else:
                atomfiles = ["ribonucleic_acids.atomtypes", "amino_acids.atomtypes"]
            atypes_list = [os.path.join(*[self.dirpath, "config", x]) for x in
                           atomfiles]
        if added_atypes is not None:
            atypes_list.append(added_atypes)
        self.atypes_dict = self.build_atypes_dict(atypes_list)
        if massdef_list is None:
            if self.one_mass:
                massfiles = ["ribonucleic_acids_1n.masses", "amino_acids.masses"]
            else:
                massfiles = ["ribonucleic_acids.masses", "amino_acids.masses"]
            massdef_list = [os.path.join(*[self.dirpath, "config", x]) for x in
                            massfiles]
        if added_massdef is not None:
            massdef_list.append(added_massdef)
        self.massdefs = self.parse_massdefs(atypes_list, massdef_list)
        self.print_verbose(self.massdefs)
        self.pdb_file = pdb_file
        self.mols = get_macromol_list(self.pdb_file, self.atypes_dict, self.massdefs, ignore_hetatms=ignore_hetatms)
        self.mol = self.mols[0]
        self.not_there = set()  # used to add resi-atom type pairs that are undefined
        self.h, self.eigvecs, self.eigvals, self.eigvecs_mat = None, None, None, None
        self.bfacts, self.entropy = None, None
        if solve:
            if use_pickle:
                if not self.build_from_pickle():
                    self.h = self.build_hessian()
                    self.solve()
                    self.pickle()
                    if not self.build_from_pickle():
                        raise ValueError("Could not unpickle the freshest pickle!")
            else:
                self.h = self.build_hessian()
                self.solve()
        elif solve_mol:
            self.mol.solve()

    @abc.abstractmethod
    def build_hessian(self):
        pass

    @abc.abstractmethod
    def build_from_pickle(self):
        """ Must return boolean indicating success of building self from pickled version.
        """
        pass

    @abc.abstractmethod
    def pickle(self):
        """ Since many ENMs may end up being run on the same file, the name of the pickle must be unique (i.e.
            .encom.pickle, .anm.pickle, etc.
        """
        pass

    def clear_info(self):
        self.h, self.eigvecs = None, None
        # self.mol.clear_info()

    def reconstitute(self):
        self.eigvecs = np.transpose(self.eigvecs_mat)

    def get_filtered_eigvecs_mat(self, indices, filter):
        """ Returns only the eigenvectors corresponding to the selected masses.
        """
        if filter is None:
            filtered_vecs = self.eigvecs
        else:
            filtered_vecs = self.get_n_filtered_eigvecs(filter, 3*len(self.mol.masses), start=0)
        indices3n = []
        for i in indices:
            i3 = 3*i
            indices3n.append(i3)
            indices3n.append(i3+1)
            indices3n.append(i3+2)
        filtered_vecs = np.take(filtered_vecs, indices3n, axis=1)
        # for i in range(len(filtered_vecs)):
        #     filtered_vecs[i] /= np.linalg.norm(filtered_vecs[i])
        return np.transpose(self.gram_schmidt(filtered_vecs, 3*len(indices)))

    def is_equal(self, other):
        """
        Args:
            other: instance of ENM to test
        Returns: boolean

        """
        assert isinstance(other, ENM)
        if not (self.pickle_version == other.pickle_version and
                self.atypes_dict == other.atypes_dict and
                massdefs_dicts_are_same(self.massdefs, other.massdefs) and
                self.mol.is_equal(other.mol)):
            return False
        if len(self.mols) > 1:
            if not len(self.mols) == len(other.mols):
                return False
            for i in range(1, len(self.mols)):
                if not self.mols[i].is_equal(other.mols[i]):
                    return False
        return True

    def solve(self):
        if not self.mol.solved:
            self.mol.solve()
        if self.h is None:
            self.h = self.build_hessian()
        vals, vecs = np.linalg.eigh(self.h)
        self.eigvecs_mat = vecs  # this is the version with one vector per column
        vecs = np.transpose(vecs)
        self.eigvals = vals
        self.eigvecs = vecs  # this is a list of vectors (one vector per row)
        return self.eigvals, self.eigvecs

    def parse_massdefs(self, atypes_list, massdef_list):
        massdefs = dict()
        for i, (atypes_file, massdef_file) in enumerate(zip(atypes_list, massdef_list)):
            assert atypes_file.split('/')[-1].split('.')[0] == massdef_file.split('/')[-1].split('.')[0]
            resis = self.resis_from_atypes(atypes_file)
            massdef = MassDef(massdef_file)
            for r in resis:
                if r in massdefs:
                    raise ValueError(
                        "Trying to add residue {0} twice in a mass definition!".format(r))
                massdefs[r] = massdef
        return massdefs

    def resis_from_atypes(self, atypes):
        resis = []
        with open(atypes) as f:
            lines = f.readlines()
        for line in lines:
            resi = line.split('|')[0].strip()
            if resi != "ADD_TO_ABOVE":
                resis.append(resi)
        return resis

    def build_atypes_dict(self, atypes_list):
        atd = dict()
        for filename in atypes_list:
            d = self.get_atype_dict(filename)
            for key in d:
                if key in atd:
                    raise ValueError(
                        "Residue {0} was in many atomypes files (2nd of which is {1}".format(key, filename))
                atd[key] = d[key]
        return atd

    def get_atype_dict(self, filename):
        d = dict()
        with open(filename) as f:
            lines = f.readlines()
        for l in lines:
            ll = l.split('|')
            resi = ll[0].strip()
            rest = ll[1].strip()
            # atoms is a list of 2-element lists : [atom name, atom type number]
            atoms = [a.strip().split(':') for a in rest.split(',')]
            if len(resi) > 3 and resi == "ADD_TO_ABOVE":  # line with atom types to add to many residues
                for atom in atoms:
                    for key in d:
                        if atom[0] in d[key]:
                            raise ValueError("Trying to add atom {0} twice (file {1})".format(atom[0], filename))
                        d[key][atom[0]] = int(atom[1])
            elif len(resi) > 3:  # resiname is too long for PDB files
                raise ValueError(
                    "unsupported (longer than 3 characters) residue name {0} in file {1}".format(resi, filename))
            else:  # normal line, one residue
                if resi in d:
                    raise ValueError("Trying to add residue {0} 2 times (file {1})".format(resi, filename))
                d[resi] = dict()
                for atom in atoms:
                    if atom[0] in d[resi]:
                        raise ValueError(
                            "Trying to add atom {0} 2 times to residue {1} (file {2})".format(atom[0], resi, filename))
                    d[resi][atom[0]] = int(atom[1])
        return d

    def compute_bfactors(self, filter=None):
        """ Computes the root-mean-square flutuations of every mass, which is
            can be seen as a prediction of the experimental temperature factors,
            of B-factors.
        """
        assert self.eigvals is not None
        assert self.eigvecs is not None
        n = int(len(self.eigvecs) / 3)
        bfacts = []
        for i in range(n):
            bfact = 0
            for j in range(6, n * 3):  # skips 1st 6 rotation-translation motions
                temp = 0
                for k in range(3):
                    temp += self.eigvecs[j][3 * i + k] ** 2
                bfact += temp / self.eigvals[j]
            bfacts.append(bfact * 1000)
        self.bfacts = bfacts
        if filter is not None:
            kept_bfacts = []
            masses = self.mol.masses
            for i, bfact in enumerate(self.bfacts):
                resiname = masses[i][4]
                if resiname.split('|')[0].split('.')[-1] in filter:
                    kept_bfacts.append(bfact)
            return kept_bfacts
        else:
            return self.bfacts

    def get_exp_bfacts(self, method="average", filter=None):
        """ Extracts the experimental b-factors for every mass in the system,
            using the specified method (average, center, min, max).
        """
        bfacts = []
        resis = self.mol.resis
        for m in self.mol.masses:
            resiname = m[4] # 5th field is residue key
            if filter is not None and resiname.split('|')[0].split('.')[-1] not in filter:
                continue
            resi = resis[resiname]
            if method == "average":
                s = 0
                count = 0
                for a in resi:
                    s += resi[a].bfact
                    count += 1
                bfacts.append(s / count)
            elif method == "center":
                center = m[3]  # 4th field is center atom
                bfacts.append(resi[center].bfact)
            else:
                raise ValueError('No other method than "average" or "center" currently supported by get_exp_bfacts')
        return bfacts

    def compute_vib_entropy(self):
        """ Computes the vibrational entropy of the system.
        """
        ent = 0
        for i in range(6, len(self.eigvals)):
            ent += np.log(1 / self.eigvals[i])
        self.entropy = ent
        return ent

    def compute_vib_entropy_nussinov(self, beta=None, factor=1):
        if beta is None:
            beta = 1.55 * 10 ** -13  # beta is h/kT, this value is for T = 310 K
        beta *= factor
        e = 2.718281828459045
        pi = 3.1415926535897932384626433832795028841971693993751
        entro = 0
        for i in range(6, len(self.eigvals)):
            vi = (self.eigvals[i] ** 0.5) / (2 * pi)
            x = vi * beta
            entro += x / (e ** x - 1) - np.log(1 - e ** (-1 * x))
        return entro

    def compute_vib_enthalpy_nussinov(self, beta=None):
        if beta is None:
            beta = 1.55 * 10 ** -13  # beta is h/kT, this value is for T = 310 K
        e = 2.718281828459045
        h = 6.62607004 * 10 ** -34
        pi = 3.1415926535897932384626433832795028841971693993751
        enthal = 0
        for i in range(6, len(self.eigvals)):
            vi = (self.eigvals[i] ** 0.5) / (2 * pi)
            enthal += (0.5 + 1 / (e ** (beta * vi) - 1)) * h * vi
        return enthal

    def print_verbose(self, string):
        if self.verbose:
            print(string)

    def get_masses(self):
        return self.mol.masses

    def get_n_masses(self):
        return len(self.mol.masses)

    def get_xyz(self):
        return self.mol.get_xyz()

    def get_filtered_xyz(self, filter):
        return self.mol.get_filtered_xyz(filter)

    def get_filtered_3n_vector(self, filter):
        return self.mol.get_filtered_3n_vector(filter)

    def get_n_filtered_eigvecs(self, filter, n_vecs, start=6):
        masses = self.get_masses()
        n = len(masses)
        included = np.zeros((n))
        for i, m in enumerate(masses):
            if m[4].split('|')[0].split('.')[-1] in filter:
                included[i] = 1
        filtered_eigvecs = np.zeros((3*n, 3*int(np.sum(included))))
        for i in range(len(self.eigvecs)):
            eigvec = self.eigvecs[i]
            current_j = 0
            for j in range(len(eigvec)):
                if included[int(np.floor(j/3))] == 1:
                    filtered_eigvecs[i][current_j] = eigvec[j]
                    current_j += 1
        return filtered_eigvecs[start:start+n_vecs]

    def get_n_filtered_eigvecs_orthonorm(self, filter, n_vecs, start=6):
        filtered = self.get_n_filtered_eigvecs(filter, 3*len(self.mol.masses)-start, start=start)
        return self.gram_schmidt(filtered, n_vecs)

    def set_bfactors(self, vector):
        """ Sets the bfactors of all atoms in the file. Vector needs to be the same length as the number of masses.
        """
        assert self.get_n_masses() == len(vector)
        self.mol.set_bfactors(vector)

    def write_dynamical_signature(self, outfile):
        """ Writes a 'dynamical signature', which is the predicted b_factor at
            every mass, to the specified output file.
        """
        if not hasattr(self, "bfacts"):
            self.compute_bfactors()
        masses = self.mol.masses
        bfacts = self.bfacts
        assert len(masses) == len(bfacts)
        with open(outfile, "w") as f:
            for i in range(len(masses)):
                f.write("{0}\t{1}\n".format(masses[i][4], bfacts[i]))

    def write_normalized_dynamical_signature(self, outfile, total=1, type="amino_acids"):
        """ Writes a normalized 'dynamical signature', summing to parameter total.
        """
        if not hasattr(self, "bfacts"):
            self.compute_bfactors()
        masses = self.mol.masses
        bfacts = self.bfacts
        assert len(masses) == len(bfacts)
        kept_bfacts = []
        kept_masses = []
        for i, m in enumerate(masses):
            if m[5].name == type:
                kept_bfacts.append(bfacts[i])
                kept_masses.append(masses[i])
        kept_bfacts = np.array(kept_bfacts)
        kept_bfacts /= np.sum(kept_bfacts)
        kept_bfacts *= total
        with open(outfile, "w") as f:
            for i in range(len(kept_masses)):
                f.write("{0}\t{1}\n".format(kept_masses[i][4], kept_bfacts[i]))

    def write_to_file(self, filename):
        self.mol.write_to_file(filename)

    def write_to_filehandle(self, fh):
        self.mol.write_to_filehandle(fh)

    def translate_3n_matrix(self, vector):
        self.mol.translate_3n_matrix(vector)
        self.mol.update()

    def translate_3n_vector(self, vector):
        self.mol.translate_3n_vector(vector)
        self.mol.update()

    def translate_xyz(self, vector):
        self.mol.translate_xyz(vector)
        self.mol.update()

    def rotate(self, rotation_mat):
        self.mol.rotate(rotation_mat)
        self.mol.update()

    def gram_schmidt(self, vectors, target_n):
        """ From https://gist.github.com/iizukak/1287876/edad3c337844fac34f7e56ec09f9cb27d4907cc7#gistcomment-1871542.
            Gram-schmidt orthonormalization of row vectors.
        """
        basis = []
        for v in vectors:
            w = v - np.sum( np.dot(v,b)*b  for b in basis )
            if (w > 1e-10).any():
                basis.append(w/np.linalg.norm(w))
            if len(basis) == target_n:
                break
        if len(basis) < target_n:
            raise ValueError("Unable to produce {0} orthonormalized vectors for pdb file {1}".format(target_n, self.pdb_file))
        return np.array(basis)








