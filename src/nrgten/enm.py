from nrgten.macromol import get_macromol_list
from nrgten.massdef import MassDef, massdefs_dicts_are_same
import numpy as np
import os
import abc


class ENM(metaclass=abc.ABCMeta):
    """Abstract base class (ABC) for Elastic Network Models (ENMs).

    The ENM class implements methods that are common for all implemented ENMs (ENCoM, ANM and STeM as of this writing)
    as well as the basic constructor. Subclasses are responsible for implementing the build_hessian() method (which
    generates the Hessian matrix) as well as the pickle() and build_from_pickle() methods which allow the pickling
    of solved ENMs for greatly accelerated subsequent computation of various metrics and/or properties.

    Attributes:
        dirpath (str): The absolute path to the directory containing the enm.py module.
        verbose (bool): If True, verbose mode is on.
        pickle_version (str): Ensures that ENMs pickled in the past are consistent with the current pickling scheme.
        one_mass (bool): For nucleic acids, False means 3 masses per nucleotide and True means 1 mass/nucleotide
        atypes_dict (dict): Dictionary of atom types built from the .atomtypes configuration files.
        massdefs (MassDef): Mass definitions built from the .masses configuration files.
        pdb_file (str): The PDB file on which the ENM is run.
        mols (list): list of Macromol objects representing the macromolecules in the PDB file (one per MODEL).
        mol (Macromol): The first Macromol (and the only in case of single-model PDBs), on which the ENM will be solved.
        not_there (set): Set of atoms that are not defined in the .atomtypes and .masses config files (normally empty).
        h (numpy.ndarray): The Hessian matrix.
        eigvecs (numpy.ndarray): The eigenvectors, in row format (one eigenvector per row).
        eigvals (list): list of eigenvalues, from lowest to highest
        eigvecs_mat (numpy.ndarrary): The eigenvectors in column format (one eigenvector per column).
        bfacts (list): The computed b-factors.
        entropy (float): The computed vibrational entropy.
    """
    def __init__(self, pdb_file, added_atypes=None, added_massdef=None, atypes_list=None, massdef_list=None,
                 verbose=False, solve=False, use_pickle=False, ignore_hetatms=False, solve_mol=True, one_mass=False):
        """Constructor for the ENM abstract base class.

        Args:
            pdb_file (str): The PDB file to read.
            added_atypes (list, optional): list of .atypes configuration files to add.
            added_massdef (list, optional): list of .masses configuration files to add.
            atypes_list (list, optional): If supplied, the default .atypes configuration files are ignored and these
                                          are the only ones read.
            massdef_list (list, optional): If supplied, the default .masses configuration files are ignored and these
                                           are the only ones read.
            verbose (bool, optional): Triggers the verbose mode.
            solve (bool, optional): If True, the Hessian matrix will be built and solved.
            use_pickle (bool, optional): If True, the ENM object will be solved only once and subsequently built from
                                         its pickled representation. Uses a lot of disk space for large systems.
            ignore_hetatms (bool, optional): Flag to ignore HETATM records in the PDB file.
            solve_mol (bool, optional): If True, the underlying Macromol object will be solved, i.e. the connectivity
                                        of the residues will be inferred.
            one_mass (bool, optional): If True, nucleic acids will be built using only one mass per nucleotide instead
                                       of 3.
        """
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
        """Builds the Hessian matrix for the ENM. Implemented by subclasses.
        """
        pass

    @abc.abstractmethod
    def build_from_pickle(self):
        """Builds the solved ENM state from a pickled object. Implemented by subclasses.

        Note:
            Since ENMs in general have parameters, subclasses need to make sure to check that the input PDB and
            all parameters are identical between the pickled representation and the present run. See example in
            ENCoM class in encom module.

        Returns:
            True if successful, False otherwise.
        """
        pass

    @abc.abstractmethod
    def pickle(self):
        """Uses the pickle module to produce a serialized object. Implemented by subclasses.

        Note:
            Since many ENMs may end up being run on the same file, the extension of the pickled file must be unique
            (i.e. .encom.pickle, .anm.pickle, etc.)
        """
        pass

    def _clear_info(self):
        """Clears the Hessian and row eigenvectors before pickling, to save space.
        """
        self.h, self.eigvecs = None, None

    def _reconstitute(self):
        """Reconstitues the row eigenvectors from the column eigenvectors when building from pickle.
        """
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

    def compute_vib_entropy_rigid_rotor(self):
        """ Computes the vibrational entropy of the system using the rigid-rotor approximation.

            This was the form used in the original ENCoM paper (doi: ﻿10.1371/journal.pcbi.1003569)
        """
        ent = 0
        for i in range(6, len(self.eigvals)):
            ent += np.log(1 / self.eigvals[i])
        self.entropy = ent
        return ent

    def compute_vib_entropy(self, beta=None, factor=1):
        """ Vibrational entropy from the eigenfrequencies (without rigid-rotor approximation).

            beta is the scaling factor (higher beta means lower temperature)
        """
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

    def compute_vib_enthalpy(self, beta=None, factor=1):
        """ Vibrational enthalpy from the eigenfrequencies (without rigid-rotor approximation).

            beta is the scaling factor (higher beta means lower temperature)
        """
        if beta is None:
            beta = 1.55 * 10 ** -13  # beta is h/kT, this value is for T = 310 K
        beta *= factor
        e = 2.718281828459045
        h = 6.62607004 * 10 ** -34
        pi = 3.1415926535897932384626433832795028841971693993751
        enthal = 0
        for i in range(6, len(self.eigvals)):
            vi = (self.eigvals[i] ** 0.5) / (2 * pi)
            enthal += (0.5 + 1 / (e ** (beta * vi) - 1)) * h * vi
        return enthal

    def build_conf_ensemble(self, modes_list, filename, step=0.5, max_displacement=2.0, max_conformations=10000):
        """ Has the same functionality (better name) as the old ENCoM executable build_grid_rmsd.
            Creates a conformational ensemble by making every combination of the selected modes at every given rmsd step,
            up to a total deformation of max_displacement for each mode (and in both directions). Writes the output as
            a multi-model pdb file.
            Important note: the mode indices in the modes_list are assumed to start at 1, with the first nontrivial mode
            thus being at index 7.
        """
        # TODO : test that this still works
        assert isinstance(modes_list, list)
        grid_side = 1 + (2 * max_displacement / step)  # length of one side of the grid
        if abs((grid_side - 0.999999999999) % 2) > 0.000001:  # make sure that small floating point errors are ignored
            raise ValueError(
                "build_conf_ensemble was executed with step={0} and max_displacement={1}. max_displacement " +
                "has to be a multiple of step.".format(step, max_displacement))
        grid_side = int(round(grid_side))
        grid_side = int(grid_side)
        n_conf = int(grid_side ** len(modes_list))  # total number of points, or conformations, in the grid
        if n_conf > max_conformations:
            raise ValueError("build_conf_ensemble was executed with parameters specifying {0} conformations with " +
                             "max_conformations set at {1}. If you really want that many conformations, set " +
                             "max_conformations higher.".format(n_conf, max_conformations))
        eigvecs_list = []
        for mode_index in modes_list:
            if mode_index < 7:
                raise ValueError(
                    "build_conf_ensemble was run with mode index {0} in the modes_list, which is a trivial " +
                    "(rotational/translational) motion.".format(mode_index))
            eigvecs_list.append(np.copy(self.eigvecs[mode_index + 5]))
            modermsd = self.rmsd_of_3n_vector(eigvecs_list[-1])
            eigvecs_list[-1] /= modermsd
        with open(filename, "w") as fh:
            fh.write("REMARK Conformational ensemble written by nrgten.enm.ENM.build_conf_ensemble()\n" +
                     "REMARK from the NRGTEN Python package, copyright Najmanovich Research Group 2020\n" +
                     "REMARK This ensemble contains {0} conformations\n".format(n_conf))
            for i in range(n_conf):
                self.write_one_point(eigvecs_list, i, grid_side, step, fh)
            fh.write("END\n")

    def write_one_point(self, eigvecs_list, conf_n, grid_side, step, fh):
        """ Helper function for build_conf_ensemble. conf_n is the conformation number from 0 to n-1. This function computes
            contributions from every mode automatically, translates the enm coordinates, and resets it back to original
            after having written one model to the filehandle fh.
        """
        model_n = conf_n
        nsteps_list = []
        for i in range(len(eigvecs_list)):
            nsteps = conf_n % grid_side
            conf_n -= nsteps
            conf_n /= grid_side
            nsteps_list.append(nsteps - (grid_side - 1) / 2)
        t_vect = np.zeros(len(eigvecs_list[0]))
        for vec, nsteps in zip(eigvecs_list, nsteps_list):
            t_vect += vec * nsteps * step
        self.translate_3n_vector(t_vect)
        self.write_model(int(model_n + 1), fh)
        self.translate_3n_vector(-t_vect)

    def rmsd_of_3n_vector(self, vec):
        dists = np.zeros((int(len(vec) / 3)))
        for i in range(0, len(vec), 3):
            dists[int(i / 3)] = vec[i] ** 2 + vec[i + 1] ** 2 + vec[i + 2] ** 2
        return np.sqrt(np.mean(dists))

    def write_model(self, count, fh):
        """ Writes one model to the given filehandle, with model number count.
        """
        fh.write("MODEL     {:>4}\n".format(count))
        self.write_to_filehandle(fh)
        fh.write("ENDMDL\n")

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









