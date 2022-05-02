from nrgten.macromol import get_macromol_list
from nrgten.massdef import MassDef, massdefs_dicts_are_same
import numpy as np
import os
import abc
import sys


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
                 verbose=False, solve=False, use_pickle=False, ignore_hetatms=False, solve_mol=True, one_mass=False,
                 teruel2021_legacy=False):
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
            teruel2021_legacy (bool, optional): If True, the parameters will perfectly reproduce the vibrational entropy
                                                calculations in the Teruel et al. 2021 PLoS Computational Biology paper
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
        self.atypes_dict = self._build_atypes_dict(atypes_list)
        if massdef_list is None:
            if self.one_mass:
                massfiles = ["ribonucleic_acids_1n.masses", "amino_acids.masses"]
            else:
                massfiles = ["ribonucleic_acids.masses", "amino_acids.masses"]
            massdef_list = [os.path.join(*[self.dirpath, "config", x]) for x in
                            massfiles]
        if added_massdef is not None:
            massdef_list.append(added_massdef)
        self.massdefs = self._parse_massdefs(atypes_list, massdef_list)
        self._print_verbose(self.massdefs)
        self.pdb_file = pdb_file
        self.teruel2021_legacy = teruel2021_legacy
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
            bool: True if successful, False otherwise.
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

    def solve(self):
        """Solves the Hessian matrix to get eigenvectors and eigenvalues.

        Returns:
            tuple: (eigenvalues, eigenvectors) eigenvectors are in row format
        """
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

    def is_equal(self, other):
        """Compares two ENM objects for equality. Useful for unpickling.

        Args:
            other (ENM): instance of ENM to test.

        Note:
            This is the general method that subclasses can call in addition to any specifics about their implementation.

        Returns:
            bool: True if equal, False otherwise.
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

    def compute_bfactors(self, filter=None, modes_proportion=1):
        """Computes the predicted b-factors.

        The root-mean-square flutuations of every mass can be seen as a prediction of the experimental temperature
        factors, or B-factors.

        Note:
            In order to have more convenient numbers, the b-factors are scaled up by a factor of 1000. This does not
            affect the validity of the value as it is only an indicator of relative flexibility between the masses.

        Args:
            filter (set, optional): set of str names of masses to select (corresponding to the names in .masses config
                                    files).

        Returns:
            list: the computed b-factors
        """
        assert self.eigvals is not None
        assert self.eigvecs is not None
        n = int(len(self.eigvecs) / 3)
        bfacts = []
        end_j = int((n * 3 - 6) * modes_proportion) + 6
        for i in range(n):
            bfact = 0
            for j in range(6, end_j):  # skips 1st 6 rotation-translation motions
                temp = 0
                for k in range(3):
                    temp += self.eigvecs[j][3 * i + k] ** 2
                bfact += temp / self.eigvals[j]
            bfacts.append(bfact * 1000)
        filtered_bfacts = self.filter_bfactors(bfacts, filter)
        self.bfacts = filtered_bfacts
        return filtered_bfacts

    def get_exp_bfacts(self, method="average", filter=None):
        """Extracts the experimental b-factors for every mass in the system.

        Args:
            method (str): can currently be only "average" or "center". Average is the mean taken from all the atoms in
                          the residue. Center is the value from the center atom (on which the mass is positioned).
            filter (set, optional): set of str names of masses to select (corresponding to the names in .masses config
                                    files).
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
        """Computes the vibrational entropy of the system using the rigid-rotor approximation.

        Note:
            This was the form used in the original ENCoM paper (doi: ﻿10.1371/journal.pcbi.1003569)

        Returns:
            float: The vibrational entropy of the system.
        """
        ent = 0
        for i in range(6, len(self.eigvals)):
            ent += np.log(1 / self.eigvals[i])
        self.entropy = ent
        return ent

    def filter_bfactors(self, bfacts, filter):
        if filter is not None:
            kept_bfacts = []
            masses = self.mol.masses
            for i, bfact in enumerate(self.bfacts):
                resiname = masses[i][4]
                if resiname.split('|')[0].split('.')[-1] in filter:
                    kept_bfacts.append(bfact)
            return kept_bfacts
        else:
            return bfacts

    def compute_bfactors_boltzmann(self, beta=None, factor=1, filter=None, modes_proportion=1):
        assert self.eigvals is not None
        assert self.eigvecs is not None
        n = int(len(self.eigvecs) / 3)
        bfacts = []
        end_j = int((n * 3 - 6) * modes_proportion) + 6
        if beta is None:
            beta = 1.55 * 10 ** -13  # beta is h/kT, this value is for T = 310 K
        beta *= factor
        e = 2.718281828459045
        pi = 3.1415926535897932384626433832795028841971693993751
        entros = [None, None, None, None, None, None]
        for j in range(6, end_j):
            vj = (self.eigvals[j] ** 0.5) / (2 * pi)
            x = vj * beta
            entros.append(x / (e ** x - 1) - np.log(1 - e ** (-1 * x)))
        for i in range(n):
            bfact = 0
            for j in range(6, end_j):  # skips 1st 6 rotation-translation motions
                temp = 0
                for k in range(3):
                    temp += self.eigvecs[j][3 * i + k] ** 2
                bfact += temp * entros[j]
            bfacts.append(bfact * 1000)
        filtered_bfacts = self.filter_bfactors(bfacts, filter)
        self.bfacts = filtered_bfacts
        return filtered_bfacts

    def compute_vib_entropy(self, beta=None, factor=1):
        """Vibrational entropy from the eigenfrequencies (without rigid-rotor approximation).

        Note:
            This is a more exact form of the vibrational entropy and is the preferred way to compute it. The
            compute_vib_entropy_rigid_rotor method is there only for reproducibility purposes as the old ENCoM model
            used that form.

        Args:
            beta (float, optional): The scaling factor (higher beta means lower temperature). If None, the default value
                                    is used.
            factor (float, optional): Can be used to scale the default value when beta is set to None.

        Returns:
            float: The vibrational entropy of the system.
        """
        if beta is None:
            beta = 1.55 * 10 ** -13  # beta is h/kT, this value is for T = 310 K
        beta *= factor
        if self.teruel2021_legacy:
            beta *= 2*np.pi
        e = 2.718281828459045
        pi = 3.1415926535897932384626433832795028841971693993751
        entro = 0
        for i in range(6, len(self.eigvals)):
            vi = (self.eigvals[i] ** 0.5) / (2 * pi)
            x = vi * beta
            entro += x / (e ** x - 1) - np.log(1 - e ** (-1 * x))
        return entro



    def compute_vib_enthalpy(self, beta=None, factor=1):
        """Vibrational enthalpy from the eigenfrequencies (without rigid-rotor approximation).

        Args:
            beta (float, optional): The scaling factor (higher beta means lower temperature). If None, the default value
                                    is used.
            factor (float, optional): Can be used to scale the default value when beta is set to None.

        Returns:
            float: The vibrational entropy of the system.
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

    def get_filtered_eigvecs_mat(self, indices=None, filter=None, transpose=True, n_vecs=None, start=0):
        """Allows to select subvectors that correspond to certain masses.

        This is useful when there are multiple masses per residue and the user wants to reduce eigenvectors to have
        1 xyz component per residue, for example. It also allows to select specific indices, which is useful in the
        case of alignments between slightly different structures.

        Note:
            The filtered vectors are always orthonormalized.

        Args:
            indices (list, optional): list of indices to select, from 0 to n-1 (n being the number of masses).
            filter (set, optional): set of str names of masses to select (corresponding to the names in .masses config
                                    files).
            transpose (bool, optional): If True, returns column vectors, otherwise returns row vectors.
            n_vecs (int, optional): The number of vectors to return, starting from `start`
            start (int, optional): The first vector to return.

        Returns:
            numpy.ndarray: the filtered eigenvectors in column format, or row format if `transpose` is False.
        """
        if n_vecs is None:
            n_vecs = len(self.eigvecs)
        if filter is None:
            filtered_vecs = self.eigvecs[:n_vecs]
        else:
            filtered_vecs = self._get_n_filtered_eigvecs(filter, n_vecs, start=0)
        if indices is not None:
            indices3n = []
            for i in indices:
                i3 = 3*i
                indices3n.append(i3)
                indices3n.append(i3+1)
                indices3n.append(i3+2)
            filtered_vecs = np.take(filtered_vecs, indices3n, axis=1)
        vecs = self._gram_schmidt(filtered_vecs, min(n_vecs, 3*len(indices)))
        if transpose:
            return np.transpose(vecs)
        else:
            return vecs

    def _get_n_filtered_eigvecs(self, filter, n_vecs, start):
        masses = self.mol.masses
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

    def build_conf_ensemble(self, modes_list, filename, step=0.5, max_displacement=2.0, max_conformations=10000,
                            use_motion=False):
        """Creates a conformational ensemble as a multi-model PDB file.

        The idea is to make every combination of the selected modes at every given rmsd step, up to a total deformation
        for each mode (and in both directions).

        Note:
            The mode indices in the modes_list argument are assumed to start at 1, with the first nontrivial mode
            thus being at index 7.

        Args:
            modes_list (list): list of integer indices of modes to use for the conformational ensemble. The first
                               non-trivial mode is at index 7.
            filename (str): The filename where the conformational ensemble will be written.
            step (optional, float): The RMSD step between each grid point in the conformational ensemble.
            max_displacement (float, optional): The maximum RMSD displacement for each mode. Has to be a multiple of
                                                step.
            max_conformations (int, optional): The maximum number of conformations. Ensures that users do not
                                               accidentally generate huge PDB files.
        """
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
            eigvecs_list.append(np.copy(self.eigvecs[mode_index - 1]))
            modermsd = self._rmsd_of_3n_vector(eigvecs_list[-1])
            eigvecs_list[-1] /= modermsd
        with open(filename, "w") as fh:
            fh.write("REMARK Conformational ensemble written by nrgten.enm.ENM.build_conf_ensemble()\n" +
                     "REMARK from the NRGTEN Python package, copyright Najmanovich Research Group 2020\n" +
                     "REMARK This ensemble contains {0} conformations\n".format(n_conf))
            conf_seq = range(n_conf)
            if use_motion:
                if len(modes_list) > 1:
                    raise ValueError("Cannot use 'motion' style conformational ensemble with more than one mode")
                conf_seq = []
                for breaks in [[int((n_conf-1)/2), n_conf, 1], [n_conf-1, int((n_conf-1)/2)-1, -1],
                               [int((n_conf-1)/2)-1, -1, -1], [0, int((n_conf-1)/2), 1]]:
                    for i in range(*breaks):
                        conf_seq.append(i)
            for i in conf_seq:
                self._write_one_point(eigvecs_list, i, grid_side, step, fh)
            fh.write("END\n")

    def build_conf_ensemble_with_nrg(self, modes_list, filename_ensemble, filename_energies, base_energy, beta,
                                     eigval_scaling=0.1, step=0.5, max_displacement=2.0, max_conformations=10000,
                                     use_motion=False):
        assert isinstance(modes_list, list)
        eigvals_list = np.array([self.eigvals[x-1]*eigval_scaling for x in modes_list])

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
            eigvecs_list.append(np.copy(self.eigvecs[mode_index - 1]))
            modermsd = self._rmsd_of_3n_vector(eigvecs_list[-1])
            eigvecs_list[-1] /= modermsd
        energies = np.zeros(n_conf)
        print(eigvals_list)
        with open(filename_ensemble, "w") as fh:
            fh.write("REMARK Conformational ensemble written by nrgten.enm.ENM.build_conf_ensemble()\n" +
                     "REMARK from the NRGTEN Python package, copyright Najmanovich Research Group 2020\n" +
                     "REMARK This ensemble contains {0} conformations\n".format(n_conf))
            conf_seq = range(n_conf)
            if use_motion:
                if len(modes_list) > 1:
                    raise ValueError("Cannot use 'motion' style conformational ensemble with more than one mode")
                conf_seq = []
                for breaks in [[int((n_conf - 1) / 2), n_conf, 1], [n_conf - 1, int((n_conf - 1) / 2) - 1, -1],
                               [int((n_conf - 1) / 2) - 1, -1, -1], [0, int((n_conf - 1) / 2), 1]]:
                    for i in range(*breaks):
                        conf_seq.append(i)
            for i in conf_seq:
                energies[i] = self._write_one_point_nrg(eigvecs_list, eigvals_list, i, grid_side, step, fh)
            fh.write("END\n")
        energies += base_energy
        probabilities = np.e**(-beta*energies)
        probabilities /= np.sum(probabilities)
        with open(filename_energies, "w") as f:
            f.write("model energy beta probability\n")
            for i in range(len(energies)):
                f.write("{} {} {} {}\n".format(i+1, energies[i], beta, probabilities[i]))


    def _write_one_point(self, eigvecs_list, conf_n, grid_side, step, fh):
        """Helper function for build_conf_ensemble. conf_n is the conformation number from 0 to n-1.

        Computes contributions from every mode automatically, translates the enm coordinates, and resets it back to
        original after having written one model to the filehandle fh.
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
        self._translate_3n_vector(t_vect)
        self._write_model(int(model_n + 1), fh)
        self._translate_3n_vector(-t_vect)

    def _write_one_point_nrg(self, eigvecs_list, spring_constants, conf_n, grid_side, step, fh):
        """Helper function for build_conf_ensemble. conf_n is the conformation number from 0 to n-1.

        Computes contributions from every mode automatically, translates the enm coordinates, and resets it back to
        original after having written one model to the filehandle fh.
        """
        model_n = conf_n
        nsteps_list = []
        for i in range(len(eigvecs_list)):
            nsteps = conf_n % grid_side
            conf_n -= nsteps
            conf_n /= grid_side
            nsteps_list.append(nsteps - (grid_side - 1) / 2)
        t_vect = np.zeros(len(eigvecs_list[0]))
        nrg = 0
        for vec, nsteps, k in zip(eigvecs_list, nsteps_list, spring_constants):
            t_vect += vec * nsteps * step
            nrg += k * (nsteps*step)**2
        self._translate_3n_vector(t_vect)
        self._write_model(int(model_n + 1), fh)
        self._translate_3n_vector(-t_vect)
        return nrg


    def _rmsd_of_3n_vector(self, vec):
        dists = np.zeros((int(len(vec) / 3)))
        for i in range(0, len(vec), 3):
            dists[int(i / 3)] = vec[i] ** 2 + vec[i + 1] ** 2 + vec[i + 2] ** 2
        return np.sqrt(np.mean(dists))

    def _write_model(self, count, fh):
        """ Writes one model to the given filehandle, with model number count.
        """
        fh.write("MODEL     {:>4}\n".format(count))
        self._write_to_filehandle(fh)
        fh.write("ENDMDL\n")

    def _clear_info(self):
        """Clears the Hessian and row eigenvectors before pickling, to save space.
        """
        self.h, self.eigvecs = None, None

    def _reconstitute(self):
        """Reconstitues the row eigenvectors from the column eigenvectors when building from pickle.
        """
        self.eigvecs = np.transpose(self.eigvecs_mat)

    def _parse_massdefs(self, atypes_list, massdef_list):
        massdefs = dict()
        for i, (atypes_file, massdef_file) in enumerate(zip(atypes_list, massdef_list)):
            assert atypes_file.split('/')[-1].split('.')[0] == massdef_file.split('/')[-1].split('.')[0]
            resis = self._resis_from_atypes(atypes_file)
            massdef = MassDef(massdef_file)
            for r in resis:
                if r in massdefs:
                    raise ValueError(
                        "Trying to add residue {0} twice in a mass definition!".format(r))
                massdefs[r] = massdef
        return massdefs

    def _resis_from_atypes(self, atypes):
        resis = []
        with open(atypes) as f:
            lines = f.readlines()
        for line in lines:
            resi = line.split('|')[0].strip()
            if resi != "ADD_TO_ABOVE":
                resis.append(resi)
        return resis

    def _build_atypes_dict(self, atypes_list):
        atd = dict()
        for filename in atypes_list:
            d = self._get_atype_dict(filename)
            for key in d:
                if key in atd:
                    raise ValueError(
                        "Residue {0} was in many atomypes files (2nd of which is {1}".format(key, filename))
                atd[key] = d[key]
        return atd

    def _get_atype_dict(self, filename):
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

    def _print_verbose(self, string):
        if self.verbose:
            print(string)

    def get_n_masses(self):
        """Gives the number of masses in the ENM.

        Returns:
            int: the number of masses in the ENM.
        """
        return len(self.mol.masses)

    def get_xyz(self):
        """Gives the x,y,z coordinates of the masses in the form of an Nx3 matrix.

        Returns:
            numpy.ndarray: Nx3 matrix of x,y,z coordinates of the masses.
        """
        return self.mol.get_xyz()

    def get_filtered_xyz(self, filter):
        """Gives the x,y,z coordinates of selected masses in the form of an Nx3 matrix.

        Args:
            filter (set): set of str names of masses to select (corresponding to the names in .masses config
                          files).
        Returns:
            numpy.ndarray: Nx3 matrix of x,y,z coordinates of the selected masses.
        """
        return self.mol.get_filtered_xyz(filter)

    def get_filtered_3n_vector(self, filter):
        """Gives the x,y,z coordinates of selected masses in the form of a single vector.

        Note:
            The vector is in the format: [x1, y1, z1, x2, y2, z2, ..., xN, yN, zN].

        Args:
            filter (set): set of str names of masses to select (corresponding to the names in .masses config
                          files).
        Returns:
            The 3N-dimensional vector of coordinates for the selected masses.
        """
        return self.mol.get_filtered_3n_vector(filter)

    def set_bfactors(self, vector):
        """Sets the b-factors of all atoms in the file.

        Args:
            vector (list): the new b-factors, needs to be the same length as the number of masses.
        """
        assert self.get_n_masses() == len(vector)
        self.mol.set_bfactors(vector)

    def write_dynamical_signature(self, outfile, normalize=False):
        """Writes a 'dynamical signature' to the output file specified.

        The dynamical signature is simply the predicted b_factor at every mass.

        Args:
            outfile (str): the output file.
            normalize (bool, optional): if True, the signature is normalized to 1
        """
        if not hasattr(self, "bfacts") or self.bfacts is None:
            self.compute_bfactors()
        labels = self.get_mass_labels()
        bfacts = self.bfacts
        divider = 1
        if normalize:
            divider = np.sum(bfacts)
        assert len(labels) == len(bfacts)
        with open(outfile, "w") as f:
            for label, bfact in zip(labels, bfacts):
                f.write("{0}\t{1}\n".format(label, bfact / divider))

    def get_mass_labels(self):
        masses = self.mol.masses
        labels = []
        for mass in masses:
            labels.append(mass[4])
        return labels

    def _write_to_file(self, filename):
        self.mol.write_to_file(filename)

    def _write_to_filehandle(self, fh):
        self.mol.write_to_filehandle(fh)

    def _translate_3n_matrix(self, vector):
        self.mol.translate_3n_matrix(vector)
        self.mol.update()

    def _translate_3n_vector(self, vector):
        self.mol.translate_3n_vector(vector)
        self.mol.update()

    def _translate_xyz(self, vector):
        self.mol.translate_xyz(vector)
        self.mol.update()

    def _rotate(self, rotation_mat):
        self.mol.rotate(rotation_mat)
        self.mol.update()

    def _gram_schmidt(self, vectors, target_n):
        """ From https://gist.github.com/iizukak/1287876/edad3c337844fac34f7e56ec09f9cb27d4907cc7#gistcomment-1871542.
            Gram-schmidt orthonormalization of row vectors.
        """
        basis = []
        for v in vectors:
            x = np.zeros((len(v)))
            for b in basis:
                x += np.dot(v, b)*b
            w = v - x
            # w = v - np.sum(np.dot(v,b)*b  for b in basis) # Calling np.sum(generator) is deprecated
            if (w > 1e-10).any():
                basis.append(w/np.linalg.norm(w))
            if len(basis) == target_n:
                break
        if len(basis) < target_n:
            raise ValueError("Unable to produce {0} orthonormalized vectors for pdb file {1}".format(target_n, self.pdb_file))
        return np.array(basis)

    def _get_weighted_surf_atomnum(self, sd, num_a, num_b, resi_a, resi_b, anums_dict):
        try:
            i = self.atypes_dict[resi_a][anums_dict[num_a].name]
        except KeyError:
            self.not_there.add((resi_a, anums_dict[num_a]))
            return 0
        try:
            j = self.atypes_dict[resi_b][anums_dict[num_b].name]
        except KeyError:
            self.not_there.add((resi_b, anums_dict[num_b]))
            return 0
        return sd[num_a][num_b] * self.inter_mat[i][j]

    def _get_total_surf_atomnum(self, sd, anums_list, i, j, resi_a, resi_b, anums_dict):
        if resi_a == "LIG" and resi_b == "ASP":
            dum = 0
        surf_tot = 0
        for num_a in anums_list[i]:
            for num_b in anums_list[j]:
                try:
                    if num_b in sd[num_a]:
                        surf_tot += self._get_weighted_surf_atomnum(sd, num_a, num_b, resi_a, resi_b, anums_dict)
                    if num_a in sd[num_b]:
                        surf_tot += self._get_weighted_surf_atomnum(sd, num_b, num_a, resi_b, resi_a, anums_dict)
                except:
                    print("Error in surface computation, residues {0} and {1}, file {2}".format(resi_a, resi_b,
                                                                                                self.mol.pdb_file))
                    raise
        return surf_tot

    def _compute_Bij(self, masses, resis):
        """ Computes the Bij term in the ENCoM potential, which is the
            modulating factor of the long-range interaction (V4) term calculated
            according to atomic surface complementarity.
        """
        sd = self.mol.get_surface_dict()
        n = len(masses)
        bij = np.zeros((n, n))
        atomnums_list = self._make_atomnums_list(masses, resis)
        atomnums_dict = self._make_atomnums_dict(resis)
        self._print_verbose(self.atypes_dict)
        for i in range(n):
            ma = masses[i]  # mass_a
            ka = ma[4]  # key_a
            resiname_a = ka.split('|')[0].split('.')[0]
            for j in range(i + 1, n):
                mb = masses[j]
                kb = mb[4]
                resiname_b = kb.split('|')[0].split('.')[0]
                try:
                    surf_tot = self._get_total_surf_atomnum(sd, atomnums_list, i, j, resiname_a, resiname_b, atomnums_dict)
                except:
                    e = sys.exc_info()[0]
                    print("Error in surface computation, file {0}".format(self.mol.pdb_file))
                    raise e
                bij[i][j] = surf_tot
                bij[j][i] = surf_tot
        self._print_verbose(self.not_there)
        return bij

    def _make_atomnums_list(self, masses, resis):
        """ Makes a list which corresponds in index with the masses received and
            which contains lists of atom numbers for every mass.
        """
        assert len(masses) == len(resis)
        n = len(masses)
        atomnums_list = []
        for i in range(n):
            resi = resis[masses[i][4]]
            anums = []
            for a in resi:
                anums.append(resi[a].num)
            atomnums_list.append(anums)
        return atomnums_list

    def _make_atomnums_dict(self, resis):
        """ Makes a dictionary associating each atom number to an Atom object.
        """
        numdict = dict()
        for r in resis:
            for a in resis[r]:
                numdict[resis[r][a].num] = resis[r][a]
        return numdict



# End of ENM class, start of utility methods ###########################################################################


def generate_dynasigs_df(filenames, outname, id_func=None, beta_values=None, models=None, models_labels=None,
                         additional_info_dict=None, add_info_labels=None,
                         added_atypes_list=None, added_massdef_list=None, get_cf_from_file=False):
    if models is None:
        from nrgten.encom import ENCoM
        models = [ENCoM]
        models_labels = ["ENCoM"]
    if beta_values is None:
        beta_values = [1]
    if id_func is None:
        id_func = lambda x: x.split('/')[-1]
    assert isinstance(beta_values, list) and isinstance(models, list) and isinstance(models_labels, list)
    assert len(models) == len(models_labels)
    if added_atypes_list is not None:
        assert len(added_atypes_list) == len(filenames) == len(added_massdef_list)
    dynasigs_data = []
    dynasigs_masslabels = None
    ids_data = []
    betas_data = []
    vib_entro_data = []
    mod_labels_data = []
    cfs = []
    for mod, mod_lab in zip(models, models_labels):
        for i, filename in enumerate(filenames):
            filename_id = id_func(filename)
            if added_atypes_list is None:
                enm = mod(filename)
            else:
                enm = mod(filename, added_atypes=added_atypes_list[i], added_massdef=added_massdef_list[i])
            masslabels = [x.split('.')[-1] for x in enm.get_mass_labels()]  # only the mass name to allow for mutations
            for beta_val in beta_values:
                dynasigs_data.append(enm.compute_bfactors_boltzmann(beta=beta_val))
                vib_entro_data.append(enm.compute_vib_entropy(beta=beta_val))
                if dynasigs_masslabels is None:
                    dynasigs_masslabels = masslabels
                else:
                    validate_masslabels(masslabels, dynasigs_masslabels)
                ids_data.append(filename_id)
                betas_data.append(beta_val)
                mod_labels_data.append(mod_lab)
                if get_cf_from_file:
                    cf = None
                    with open(filename) as f:
                        lines = f.readlines()
                    for line in lines:
                        if line.startswith("REMARK CF="):
                            cf = float(line[10:])
                            break
                    if cf is not None:
                        cfs.append(cf)
                    else:
                        raise ValueError("Could not find CF line in file {}".format(filename))
    colnames = ["id", "model", "beta", "vib_entro"]
    if get_cf_from_file:
        colnames.append("cf")
    if additional_info_dict is not None:
        assert add_info_labels is not None and isinstance(add_info_labels, list)
        colnames = colnames + add_info_labels
    sig_len = len(dynasigs_masslabels)
    colnames = colnames + dynasigs_masslabels
    df_strs = [" ".join(colnames)]
    for i in range(len(dynasigs_data)):
        assert len(dynasigs_data[i]) == sig_len
        current_str = "{} {} {} {}".format(ids_data[i], mod_labels_data[i], betas_data[i], vib_entro_data[i])
        if get_cf_from_file:
            current_str = current_str + " {}".format(cfs[i])
        if additional_info_dict is not None:
            current_str = current_str + " " + " ".join(
                [str(additional_info_dict[ids_data[i]][label]) for label in add_info_labels])
        current_str = current_str + " " + " ".join([str(x) for x in dynasigs_data[i]])
        df_strs.append(current_str)
    with open(outname, "w") as f:
        f.write("\n".join(df_strs))
        f.write("\n")


def validate_masslabels(new_labels, old_labels):
    assert len(new_labels) == len(old_labels)
    for new, old in zip(new_labels, old_labels):
        if new != old:
            raise ValueError("problem with masslabels in generate_dynasigs_df: {} not equal to {}".format(new, old))









