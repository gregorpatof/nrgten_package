from nrgten.enm import ENM
import numpy as np
import os
import pickle


class ANM(ENM):
    """Class implementing the ANM (Anisotropic Network Model).

    Has 3 parameters: cutoff, power dependence and spring constant. Spring constant just rescales fluctuations but
    doesn't change their relative values. Cutoff can be infinity (fully connected model). The original ANM paper is:
    `doi.org/10.1016/S0006-3495(01)76033-X <https://doi.org/10.1016/S0006-3495(01)76033-X>`_ and the paper on varying
    the power dependence is: `doi.org/10.1073/pnas.0902159106 <https://doi.org/10.1073/pnas.0902159106>`_

    Note:
        ANM extends the `ENM` class, to see inherited attributes look at the documentation for `ENM`.

    Attributes:
        cut (float): the distance cutoff for interaction between 2 masses.
        pd (float): the power dependence of the interaction.
        kr (float): the interaction constant (does not change the solution but scales the values).
    """
    def __init__(self, pdb_file, cut=float('inf'), power_dependence=0, kr=1, solve=True, use_pickle=False,
                 ignore_hetatms=False, atypes_list=None, massdef_list=None, solve_mol=True, one_mass=False):
        """Constructor for the ANM class.

        Args:
            pdb_file (str): The PDB file from which to build the model.
            cut (float, optional): The interaction cutoff.
            power_dependence (float, optional): The power dependence. For consistency with published papers, a power
                                                dependence of 0 means that the distance will be squared in the Hessian.
            solve (bool, optional): If True, the Hessian matrix will be built and solved.
            use_pickle (bool, optional): If True, the ENM object will be solved only once and subsequently built from
                                         its pickled representation. Uses a lot of disk space for large systems.
            ignore_hetatms (bool, optional): Flag to ignore HETATM records in the PDB file.
            atypes_list (list, optional): If supplied, the default .atypes configuration files are ignored and these
                                          are the only ones read.
            massdef_list (list, optional): If supplied, the default .masses configuration files are ignored and these
                                           are the only ones read.
            solve_mol (bool, optional): If True, the underlying Macromol object will be solved, i.e. the connectivity
                                        of the residues will be inferred.
            one_mass (bool, optional): If True, nucleic acids will be built using only one mass per nucleotide instead
                                       of 3.
        """
        self.cut = cut
        self.pd = power_dependence
        self.kr = kr
        super().__init__(pdb_file, solve=solve, use_pickle=use_pickle, ignore_hetatms=ignore_hetatms,
                         atypes_list=atypes_list, massdef_list=massdef_list, solve_mol=solve_mol, one_mass=one_mass)

    def build_hessian(self):
        """Builds the Hessian matrix.

        Returns:
            The Hessian matrix.
        """
        if not self.mol.solved:
            self.mol.solve()
        masscoords = self.mol.masscoords
        distmat = self.mol.distmat
        n = len(masscoords)
        hessian = np.zeros((3*n, 3*n))
        for i in range(n):
            for j in range(i+1, n):
                dist = distmat[i][j]
                if dist <= self.cut:
                    dist_pd = dist ** (2 + self.pd)

                    # diagonal of the off-diagonal 3x3 element and update diagonal of diagonal element
                    for k in range(3):
                        val = 2 * self.kr * (masscoords[j][k] - masscoords[i][k]) ** 2 / dist_pd
                        hessian[3 * i + k][3 * j + k] = -val
                        hessian[3 * i + k][3 * i + k] += val
                        hessian[3 * j + k][3 * j + k] += val

                    # off-diagonals of the off-diagonal 3x3 element and update off-diagonal of diagonal element
                    for (k, l) in ((0, 1), (0, 2), (1, 2)):
                        val = 2 * self.kr * (masscoords[j][k] - masscoords[i][k]) * \
                              (masscoords[j][l] - masscoords[i][l]) / dist_pd
                        hessian[3 * i + k][3 * j + l] = -1 * val
                        hessian[3 * i + l][3 * j + k] = -1 * val
                        hessian[3 * i + k][3 * i + l] += val
                        hessian[3 * j + k][3 * j + l] += val
        for i in range(3 * n):
            for j in range(i + 1, 3 * n):
                hessian[j][i] = hessian[i][j]
        return hessian

    def _get_pickle_file(self):
        param_string = "{0}_{1}_{2}".format(self.cut, self.pd, self.kr)
        if self.one_mass:
            return '.'.join(self.pdb_file.split('.')[:-1]) + ".ANM_{0}_1n.pickle".format(param_string)
        else:
            return '.'.join(self.pdb_file.split('.')[:-1]) + ".ANM_{0}.pickle".format(param_string)

    def build_from_pickle(self):
        """Builds an ANM object from its pickled state.

        Returns:
            True if successful, False otherwise.
        """
        pickle_file = self._get_pickle_file()
        if not os.path.isfile(pickle_file):
            return False
        try:
            with open(pickle_file, 'rb') as f:
                pickled_anm = pickle.load(f)
        except EOFError:
            return False
        if not (self.cut == pickled_anm.cut and
                self.pd == pickled_anm.pd and
                self.kr == pickled_anm.kr):
            return False
        if not self.is_equal(pickled_anm):
            return False
        self.__dict__.update(pickled_anm.__dict__)
        self._reconstitute()
        return True

    def pickle(self):
        """Builds a pickled state from the ANM object.
        """
        super()._clear_info()
        pickle_file = self._get_pickle_file()
        with open(pickle_file, 'wb') as f:
            pickle.dump(self, f)

