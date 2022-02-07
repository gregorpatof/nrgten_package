import nrgten.eq_filler as eq_filler
import numpy as np
import sys
import os
import pickle
from nrgten.enm import ENM


class ENCoM(ENM):
    """Class implementing the latest version of ENCoM (Elastic Network Contact Model).

        It extends the ENM class, implementing its own build_hessian(), pickle() and build_from_pickle() methods. See
        ENM class for the attributes it inherits. For details about the ENCoM model, see the 2014 Frappier and
        Najmanovich paper: `doi.org/10.1371/journal.pcbi.1003569 <https://doi.org/10.1371/journal.pcbi.1003569>`_

        Note:
            This class also implements the STeM model (Generalized Spring Tensor Model, on which ENCoM is based). The
            original STeM paper can be found here:
            `doi.org/10.1186/1472-6807-10-S1-S3 <https://doi.org/10.1186/1472-6807-10-S1-S3>`_

        Attributes:
            dirpath (str): The absolute path to the directory containing the encom.py module.
            use_stem (bool): If True, the model reverts back to STeM.
            pd (float): The power dependency of the long-range interaction.
            kr (float): The weight of the first term (bond stretching) from the ENCoM potential.
            ktheta (float): The weight of the second term (angle bending) from the ENCoM potential.
            kphi (float): The weight of the third term (torsion angles) from the ENCoM potential.
            epsi (float): The weight of the fourth term (long-range interactions) from the ENCoM potential.
            bfact_params (bool): If True, the best parameters for b-factor prediction (according to the 2014 paper)
                                 are used
            ic (float): The interaction constant for favorable pairs of atom types.
            inter_mat (list): The interaction matrix for atom types, in a list of lists format where every first
                              element is None (to allow indexing starting at 1 instead of 0).
            V1_H (numpy.ndarray): The Hessian for the 1st potential term (bond stratching).
            V2_H (numpy.ndarray): The Hessian for the 2nd potential term (angle bending).
            V3_H (numpy.ndarray): The Hessian for the 3rd potential term (torsion angles).
            V4_H (numpy.ndarray): The Hessian for the 4th potential term (long-range interactions).
            bij (numpy.ndarray): Matrix of the Bij term in the ENCoM potential, which is the modulating factor for the
                                 long-range interaction (V4)
        """

    def __init__(self, pdb_file, kr=1000, ktheta=10000, kphi=10000, epsi=0.01, apply_epsi=False, interact_const=3,
                 power_dependenceV4=4, interact_mat=None, use_stem=False, kphi1=1, kphi3=0.5, bfact_params=False,
                 added_atypes=None, added_massdef=None, atypes_list=None, massdef_list=None, verbose=False, solve=True,
                 ignore_hetatms=False, use_pickle=False, solve_mol=True, one_mass=False):
        """Constructor for the ENCoM class.

        Args:
            pdb_file (str): The PDB file to read.
            kr (float, optional): The weight of the first term (bond stretching) from the ENCoM potential.
            ktheta (float, optional): The weight of the second term (angle bending) from the ENCoM potential.
            kphi (float, optional): The weight of the third term (torsion angles) from the ENCoM potential.
            epsi (float, optional): The weight of the fourth term (long-range interactions) from the ENCoM potential.
            apply_epsi (bool, optional): If True, the other constants are multiplied by epsi.
            interact_const (float, optional): The interaction constant for favorable pairs of atom types.
            power_dependenceV4 (float, optional): The power dependency of the long-range interaction.
            interact_mat (list, optional): The interaction matrix for atom types, in a list of lists format where every
                                           first element is None (to allow indexing starting at 1 instead of 0).
            use_stem (bool, optional): If True, the model reverts back to STeM (on which ENCoM is based).
            kphi1 (float, optional): Scales the kphi constant.
            kphi3 (float, optional): Scales the kphi constant.
            bfact_params (bool, optional): If True, the best parameters for b-factor prediction (according to the 2014
                                           paper) are used.
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
        self.use_stem = use_stem
        self.pd = power_dependenceV4
        if self.use_stem:
            kr = 100
            ktheta = 20
            kphi = 2.75
            epsi = 0.36
            apply_epsi = True
        else:
            kphi *= (kphi1 / 2 + kphi3 * 9 / 2)

        # Setting parameters ###########################################################################################
        if apply_epsi:
            kr *= epsi
            ktheta *= epsi
            kphi *= epsi
        self.kr = kr
        self.ktheta = ktheta
        self.kphi = kphi
        self.epsi = epsi
        self.ic = interact_const
        self.bfact_params = bfact_params
        if self.bfact_params: # optimal parameters for b-factor correlation from Frappier and Najmanovich 2014
            self.kr = 1000
            self.ktheta = 100000
            self.kphi = 1000
            self.epsi = 100
            self.ic = 3
        if interact_mat is None:
            ic = self.ic
            # Presence of None is to enable indexing using the 1-8 numbers from Sobolev et al
            interact_mat = [[None] * 9,
                            [None, ic, ic, ic, 1, ic, ic, ic, ic],
                            [None, ic, 1, ic, 1, ic, ic, ic, 1],
                            [None, ic, ic, 1, 1, ic, ic, 1, ic],
                            [None, 1, 1, 1, ic, ic, ic, ic, ic],
                            [None, ic, ic, ic, ic, ic, ic, ic, ic],
                            [None, ic, ic, ic, ic, ic, ic, ic, ic],
                            [None, ic, ic, 1, ic, ic, ic, 1, ic],
                            [None, ic, 1, ic, 1, ic, ic, ic, 1]]
        self.inter_mat = interact_mat
        ################################################################################################################

        self.V1_H, self.V2_H, self.V3_H, self.V4_H, self.bij = None, None, None, None, None
        super().__init__(pdb_file, added_atypes=added_atypes, added_massdef=added_massdef, atypes_list=atypes_list,
                         massdef_list=massdef_list, verbose=verbose, solve=solve, ignore_hetatms=ignore_hetatms,
                         use_pickle=use_pickle, solve_mol=solve_mol, one_mass=one_mass)

    def _get_pickle_file(self):
        # TODO : better logic
        pickle_file = self.pdb_file.split('.')[0]
        if self.use_stem:
            pickle_file += ".stem.pickle"
        elif self.bfact_params:
            pickle_file += ".encom_bfacts.pickle"
        elif self.one_mass:
            pickle_file += ".encom_1n.pickle"
        else:
            pickle_file += ".encom.pickle"
        return pickle_file

    def build_from_pickle(self):
        """Builds an ENCoM object from its pickled state.

        Returns:
            True if successful, False otherwise.
        """
        pickle_file = self._get_pickle_file()
        if not os.path.isfile(pickle_file):
            return False
        try:
            with open(pickle_file, 'rb') as f:
                pickled_enc = pickle.load(f)
        except EOFError:
            return False
        if not (self.kr == pickled_enc.kr and
                self.ktheta == pickled_enc.ktheta and
                self.kphi == pickled_enc.kphi and
                self.epsi == pickled_enc.epsi and
                self.inter_mat == pickled_enc.inter_mat and
                self.pd == pickled_enc.pd and
                self.use_stem == pickled_enc.use_stem):
            return False
        if not self.is_equal(pickled_enc):
            return False
        self.__dict__.update(pickled_enc.__dict__)
        self._reconstitute()
        return True

    def pickle(self):
        """Builds a pickled state from the ENCoM object.
        """
        self.V1_H, self.V2_H, self.V3_H, self.V4_H, self.bij = None, None, None, None, None
        super()._clear_info()
        pickle_file = self._get_pickle_file()
        with open(pickle_file, 'wb') as f:
            pickle.dump(self, f)

    def build_hessian(self):
        """Builds the Hessian matrix.

        Returns:
            The Hessian matrix.
        """
        if not self.mol.solved:
            self.mol.solve()
        masses = self.mol.masses
        connect = self.mol.connect
        bends = self.mol.bends
        torsions = self.mol.torsions
        resis = self.mol.resis
        self.V1_H = self._build_V1_hessian(masses, connect)
        self.V2_H = self._build_V2_hessian(masses, bends)
        self.V3_H = self._build_V3_hessian(masses, torsions)
        self.V4_H = self._build_V4_hessian(masses, resis)
        V_tot = self._sum_hessians()
        return V_tot

    def build_hybrid_hessian(self, other, solve=True):
        """Builds the Hessian matrix with V1, V2, V3 from self and V4 from other.
        """
        assert isinstance(other, ENCoM)
        if not self.mol.solved:
            self.mol.solve()
        masses = self.mol.masses
        connect = self.mol.connect
        bends = self.mol.bends
        torsions = self.mol.torsions
        if not other.mol.solved:
            other.mol.solve()
        resis = other.mol.resis
        other_masses = other.mol.masses
        if self.V1_H is None:
            self.V1_H = self._build_V1_hessian(masses, connect)
        if self.V2_H is None:
            self.V2_H = self._build_V2_hessian(masses, bends)
        if self.V3_H is None:
            self.V3_H = self._build_V3_hessian(masses, torsions)
        if other.V4_H is None:
            other.V4_H = other._build_V4_hessian(other_masses, resis)
        self.h = np.add(np.add(np.add(self.V1_H, self.V2_H), self.V3_H), other.V4_H)
        if solve:
            self.solve()

    def build_hybrid_hessian_bij(self, other, solve=True):
        assert isinstance(other, ENCoM)
        if not self.mol.solved:
            self.mol.solve()
        masses = self.mol.masses
        connect = self.mol.connect
        bends = self.mol.bends
        torsions = self.mol.torsions
        resis = self.mol.resis
        if not other.mol.solved:
            other.mol.solve()
        other_resis = other.mol.resis
        other_masses = other.mol.masses
        if self.V1_H is None:
            self.V1_H = self._build_V1_hessian(masses, connect)
        if self.V2_H is None:
            self.V2_H = self._build_V2_hessian(masses, bends)
        if self.V3_H is None:
            self.V3_H = self._build_V3_hessian(masses, torsions)
        if other.bij is None:
            other.bij = other._compute_Bij(other_masses, other_resis)
        temp_V4_H = self._build_V4_hessian(masses, resis, bij=other.bij)
        self.h = np.add(np.add(np.add(self.V1_H, self.V2_H), self.V3_H), temp_V4_H)
        if solve:
            self.solve()


    def _sum_hessians(self):
        return np.add(np.add(np.add(self.V1_H, self.V2_H), self.V3_H), self.V4_H)

    def _build_V1_hessian(self, masses, connect_mat):
        n = len(masses)
        hessian = np.zeros((3 * n, 3 * n))
        for i in range(n):
            for j in range(i + 1, n):
                if connect_mat[i][j] != 0:
                    dist = _distance(masses[i], masses[j])
                    dist_sq = dist ** 2

                    # diagonal of the off-diagonal 3x3 element and update diagonal of diagonal element
                    for k in range(3):
                        val = 2 * self.kr * (masses[j][k] - masses[i][k]) ** 2 / dist_sq
                        hessian[3 * i + k][3 * j + k] += -1 * val
                        hessian[3 * i + k][3 * i + k] += val
                        hessian[3 * j + k][3 * j + k] += val

                    # off-diagonals of the off-diagonal 3x3 element and update off-diagonal of diagonal element
                    for (k, l) in ((0, 1), (0, 2), (1, 2)):
                        val = 2 * self.kr * (masses[j][k] - masses[i][k]) * (masses[j][l] - masses[i][l]) / dist_sq
                        hessian[3 * i + k][3 * j + l] += -1 * val
                        hessian[3 * i + l][3 * j + k] += -1 * val
                        hessian[3 * i + k][3 * i + l] += val
                        hessian[3 * j + k][3 * j + l] += val
        for i in range(3 * n):
            for j in range(i + 1, 3 * n):
                hessian[j][i] = hessian[i][j]
        return hessian

    def _build_V2_hessian(self, masses, bends):
        n = len(masses)
        hessian = np.zeros((3 * n, 3 * n))
        axes = ["X", "Y", "Z"]
        indexes = ["i", "j", "k"]
        for (b_i, bend) in enumerate(bends):
            self._print_verbose(b_i)
            self._print_verbose(bend)
            (i, j, k) = bend
            p = np.array(masses[i][:3]) - np.array(masses[j][:3])
            norm_p = np.linalg.norm(p)
            q = np.array(masses[k][:3]) - np.array(masses[j][:3])
            norm_q = np.linalg.norm(q)
            prod_norms = norm_p * norm_q
            norm_poverq = norm_p / norm_q
            norm_qoverp = norm_q / norm_p
            prod_norms_sq = prod_norms ** 2
            dot = np.dot(p, q)
            G = dot / prod_norms
            const = 2 * self.ktheta / (1 - G ** 2)  # constant term for filling the hessian

            co = dict()  # dictionary of coordinates (Xi, Xj, Xk, Yi, Yj, ...)
            for ax in zip([0, 1, 2], axes):
                for index in zip([i, j, k], indexes):
                    co[ax[1] + index[1]] = masses[index[0]][ax[0]]
            dGd = dict()
            for ax in axes:
                dGd[ax + "i"] = ((co[ax + "k"] - co[ax + "j"]) * prod_norms - dot * norm_qoverp * (
                co[ax + "i"] - co[ax + "j"])) / prod_norms_sq
                dGd[ax + "k"] = ((co[ax + "i"] - co[ax + "j"]) * prod_norms - dot * norm_poverq * (
                co[ax + "k"] - co[ax + "j"])) / prod_norms_sq
                dGd[ax + "j"] = ((2 * co[ax + "j"] - co[ax + "i"] - co[ax + "k"]) * prod_norms - dot * norm_qoverp * (
                co[ax + "j"] - co[ax + "i"]) -
                                 dot * norm_poverq * (co[ax + "j"] - co[ax + "k"])) / prod_norms_sq
            doubles = [x for x in zip(bend, indexes)]
            self._hessian_helper_V2V3(hessian, doubles, dGd, const)
        return hessian

    def _hessian_helper_V1V4(self, hessian, doubles, bcoord, const):
        axes = ["X", "Y", "Z"]
        for a in range(len(doubles)):
            for b in range(len(doubles)):
                i_n = doubles[a][0]  # this could hold i or j
                i_str = doubles[a][1]
                j_n = doubles[b][0]
                j_str = doubles[b][1]
                sign = 0
                if i_n == j_n:
                    sign = 1
                else:
                    sign = -1
                for a_n, a_str in zip([0, 1, 2], axes):
                    for b_n, b_str in zip([0, 1, 2], axes):
                        hessian[3 * i_n + a_n][3 * j_n + b_n] += sign * const * bcoord[a_str] * bcoord[b_str]
        return hessian

    def _hessian_helper_V1V4_optim(self, hessian, i, j, bcoord, const):
        """ This code was generated with func_writer.py
        """
        i3 = 3 * i
        j3 = 3 * j
        hessian[i3 + 0][i3 + 0] += const * bcoord["X"] * bcoord["X"]
        hessian[i3 + 0][i3 + 1] += const * bcoord["X"] * bcoord["Y"]
        hessian[i3 + 0][i3 + 2] += const * bcoord["X"] * bcoord["Z"]
        hessian[i3 + 1][i3 + 0] += const * bcoord["Y"] * bcoord["X"]
        hessian[i3 + 1][i3 + 1] += const * bcoord["Y"] * bcoord["Y"]
        hessian[i3 + 1][i3 + 2] += const * bcoord["Y"] * bcoord["Z"]
        hessian[i3 + 2][i3 + 0] += const * bcoord["Z"] * bcoord["X"]
        hessian[i3 + 2][i3 + 1] += const * bcoord["Z"] * bcoord["Y"]
        hessian[i3 + 2][i3 + 2] += const * bcoord["Z"] * bcoord["Z"]
        hessian[i3 + 0][j3 + 0] += -1 * const * bcoord["X"] * bcoord["X"]
        hessian[i3 + 0][j3 + 1] += -1 * const * bcoord["X"] * bcoord["Y"]
        hessian[i3 + 0][j3 + 2] += -1 * const * bcoord["X"] * bcoord["Z"]
        hessian[i3 + 1][j3 + 0] += -1 * const * bcoord["Y"] * bcoord["X"]
        hessian[i3 + 1][j3 + 1] += -1 * const * bcoord["Y"] * bcoord["Y"]
        hessian[i3 + 1][j3 + 2] += -1 * const * bcoord["Y"] * bcoord["Z"]
        hessian[i3 + 2][j3 + 0] += -1 * const * bcoord["Z"] * bcoord["X"]
        hessian[i3 + 2][j3 + 1] += -1 * const * bcoord["Z"] * bcoord["Y"]
        hessian[i3 + 2][j3 + 2] += -1 * const * bcoord["Z"] * bcoord["Z"]
        hessian[j3 + 0][i3 + 0] += -1 * const * bcoord["X"] * bcoord["X"]
        hessian[j3 + 0][i3 + 1] += -1 * const * bcoord["X"] * bcoord["Y"]
        hessian[j3 + 0][i3 + 2] += -1 * const * bcoord["X"] * bcoord["Z"]
        hessian[j3 + 1][i3 + 0] += -1 * const * bcoord["Y"] * bcoord["X"]
        hessian[j3 + 1][i3 + 1] += -1 * const * bcoord["Y"] * bcoord["Y"]
        hessian[j3 + 1][i3 + 2] += -1 * const * bcoord["Y"] * bcoord["Z"]
        hessian[j3 + 2][i3 + 0] += -1 * const * bcoord["Z"] * bcoord["X"]
        hessian[j3 + 2][i3 + 1] += -1 * const * bcoord["Z"] * bcoord["Y"]
        hessian[j3 + 2][i3 + 2] += -1 * const * bcoord["Z"] * bcoord["Z"]
        hessian[j3 + 0][j3 + 0] += const * bcoord["X"] * bcoord["X"]
        hessian[j3 + 0][j3 + 1] += const * bcoord["X"] * bcoord["Y"]
        hessian[j3 + 0][j3 + 2] += const * bcoord["X"] * bcoord["Z"]
        hessian[j3 + 1][j3 + 0] += const * bcoord["Y"] * bcoord["X"]
        hessian[j3 + 1][j3 + 1] += const * bcoord["Y"] * bcoord["Y"]
        hessian[j3 + 1][j3 + 2] += const * bcoord["Y"] * bcoord["Z"]
        hessian[j3 + 2][j3 + 0] += const * bcoord["Z"] * bcoord["X"]
        hessian[j3 + 2][j3 + 1] += const * bcoord["Z"] * bcoord["Y"]
        hessian[j3 + 2][j3 + 2] += const * bcoord["Z"] * bcoord["Z"]

    def _hessian_helper_V2V3(self, hessian, doubles, dGd, const):
        axes = ["X", "Y", "Z"]
        for a in range(len(doubles)):
            for b in range(len(doubles)):  # a and b are combinations of indexes : ii, ij, ik, etc...
                i_n = doubles[a][0]  # this can hold i, j or k
                i_str = doubles[a][1]
                j_n = doubles[b][0]
                j_str = doubles[b][1]
                for a_n, a_str in zip([0, 1, 2], axes):  # axis index and str repr
                    for b_n, b_str in zip([0, 1, 2], axes):
                        hessian[3 * i_n + a_n][3 * j_n + b_n] += const * dGd[a_str + i_str] * dGd[b_str + j_str]
        return hessian

    def _build_V3_hessian(self, masses, torsions):
        n = len(masses)
        hessian = np.zeros((3 * n, 3 * n))
        axes = ["X", "Y", "Z"]
        indexes = ["i", "j", "k", "l"]
        for (t_i, torsion) in enumerate(torsions):
            self._print_verbose(t_i)
            self._print_verbose(torsion)
            (i, j, k, l) = torsion
            a = np.array(masses[j][:3]) - np.array(masses[i][:3])
            b = np.array(masses[k][:3]) - np.array(masses[j][:3])
            c = np.array(masses[l][:3]) - np.array(masses[k][:3])
            v1 = np.cross(a, b)
            v2 = np.cross(b, c)
            dotv1v2 = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            prod_norms = norm_v1 * norm_v2
            G = np.dot(v1, v2) / (norm_v1 * norm_v2)
            const = 2 * self.kphi / (1 - G ** 2)

            co = dict()  # dictionary of coordinates (Xi, Xj, Xk, Yi, Yj, ...)
            for ax in zip([0, 1, 2], axes):
                for index in zip([i, j, k, l], indexes):
                    co[ax[1] + index[1]] = masses[index[0]][ax[0]]
            dv1d = eq_filler.get_dv1d(co)
            dv2d = eq_filler.get_dv2d(co)
            dnorm_v1d = eq_filler.get_dnorm_v1d(co)
            dnorm_v2d = eq_filler.get_dnorm_v2d(co)
            G_const = prod_norms ** 2  # (|v1|*|v2|)^2
            dGd = dict()
            for ax in axes:
                for ind in indexes:
                    key = ax + ind
                    dGd[key] = ((np.dot(dv1d[key], v2) + np.dot(dv2d[key], v1)) * prod_norms - dotv1v2 * (
                    dnorm_v1d[key] * norm_v2 + dnorm_v2d[key] * norm_v1)) / G_const
            doubles = [x for x in zip(torsion, indexes)]
            self._hessian_helper_V2V3(hessian, doubles, dGd, const)
        return hessian

    def _build_V4_hessian(self, masses, resis, bij=None):
        n = len(masses)
        if bij is None:
            bij = self._compute_Bij(masses, resis)
            self.bij = bij
        hessian = np.zeros((3 * n, 3 * n))
        axes = ["X", "Y", "Z"]
        indexes = ["i", "j"]
        for i in range(n):
            for j in range(i + 1, n):
                if self._is_long_range(i, j):
                    self._print_verbose((i, j))
                    distij = self.mol.distmat[i][j]
                    bcoord = dict()  # corresponds to bx, by and bz in STeM.m
                    for ax in zip([0, 1, 2], axes):
                        bcoord[ax[1]] = masses[i][ax[0]] - masses[j][ax[0]]
                    distij_pd = distij ** self.pd
                    if self.use_stem:
                        const = 120 * self.epsi / distij_pd
                    else:
                        const = 120 * (self.epsi + bij[i][j]) / distij_pd
                    doubles = [x for x in zip([i, j], indexes)]
                    # self.hessian_helper_V1V4(hessian, doubles, bcoord, const)
                    self._hessian_helper_V1V4_optim(hessian, i, j, bcoord, const)
        return hessian

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

    def _is_long_range(self, i, j):
        if i == j:
            return False
        return self.mol.is_disconnected(i, j)


def _distance(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5

if __name__ == "__main__":
    test = ENCoM("/Users/Paracelsus/school/PhD/h21/projet/encom_rna/md_analysis/trajectories/mir125_AA_1_model1.pdb")
    print(test.compute_bfactors())
