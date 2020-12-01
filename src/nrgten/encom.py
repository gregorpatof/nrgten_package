import nrgens.eq_filler as eq_filler
import numpy as np
import sys
import os
import pickle
from nrgens.enm import ENM


class ENCoM(ENM):
    """ Elastic Network Contact Model, version 2.0. A coarse-grained normal
        mode analysis model applicable to proteins, nucleic acids and their
        complexes by default, and easily extendable to any macromolecule/small
        molecule.
    """

    def __init__(self, pdb_file, kr=1000, ktheta=10000, kphi=10000, epsi=0.01, apply_epsi=False,
                 interact_const=3, power_dependenceV4=4, interact_mat=None, added_atypes=None, added_massdef=None,
                 atypes_list=None, massdef_list=None, verbose=False, use_stem=False, kphi1=1, kphi3=0.5, solve=True,
                 ignore_hetatms=False, use_pickle=False, solve_mol=True, bfact_params=False, one_mass=False):
        """ pdb_file: string path to the PDB-formatted file to read
            vconpath: string path to the Vcontacts executable (should eventually
                      be part of the Python code)
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
        self.bfact_params = bfact_params
        if self.bfact_params: # optimal parameters for b-factor correlation from Frappier and Najmanovich 2014
            self.kr = 1000
            self.ktheta = 100000
            self.kphi = 1000
            self.epsi = 100
        if interact_mat is None:
            ic = interact_const
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

    def get_pickle_file(self):
        if self.bfact_params:
            pickle_file = self.pdb_file.split('.')[0] + ".encom_bfacts.pickle"
        elif self.one_mass:
            pickle_file = self.pdb_file.split('.')[0] + ".encom_1n.pickle"
        else:
            pickle_file = self.pdb_file.split('.')[0] + ".encom.pickle"
        return pickle_file

    def build_from_pickle(self):
        pickle_file = self.get_pickle_file()
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
        self.reconstitute()
        return True

    def pickle(self):
        self.V1_H, self.V2_H, self.V3_H, self.V4_H, self.bij = None, None, None, None, None
        super().clear_info()
        pickle_file = self.get_pickle_file()
        with open(pickle_file, 'wb') as f:
            pickle.dump(self, f)

    def build_hessian(self):
        if not self.mol.solved:
            self.mol.solve()
        masses = self.mol.masses
        connect = self.mol.connect
        bends = self.mol.bends
        torsions = self.mol.torsions
        resis = self.mol.resis
        self.V1_H = self.build_V1_hessian(masses, connect)
        self.V2_H = self.build_V2_hessian(masses, bends)
        self.V3_H = self.build_V3_hessian(masses, torsions)
        self.V4_H = self.build_V4_hessian(masses, resis)
        V_tot = self.sum_hessians()
        return V_tot

    def sum_hessians(self):
        return np.add(np.add(np.add(self.V1_H, self.V2_H), self.V3_H), self.V4_H)

    def sum_hessians_V1_V3_V4(self):
        return np.add(np.add(self.V1_H, self.V3_H), self.V4_H)

    def build_V1_hessian(self, masses, connect_mat):
        n = len(masses)
        hessian = np.zeros((3 * n, 3 * n))
        for i in range(n):
            for j in range(i + 1, n):
                if connect_mat[i][j] != 0:
                    dist = distance(masses[i], masses[j])
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

    def build_V2_hessian(self, masses, bends):
        n = len(masses)
        hessian = np.zeros((3 * n, 3 * n))
        axes = ["X", "Y", "Z"]
        indexes = ["i", "j", "k"]
        for (b_i, bend) in enumerate(bends):
            self.print_verbose(b_i)
            self.print_verbose(bend)
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
            self.hessian_helper_V2V3(hessian, doubles, dGd, const)
        return hessian

    def hessian_helper_V1V4(self, hessian, doubles, bcoord, const):
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

    def hessian_helper_V1V4_optim(self, hessian, i, j, bcoord, const):
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

    def hessian_helper_V2V3(self, hessian, doubles, dGd, const):
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

    def build_V3_hessian(self, masses, torsions):
        n = len(masses)
        hessian = np.zeros((3 * n, 3 * n))
        axes = ["X", "Y", "Z"]
        indexes = ["i", "j", "k", "l"]
        for (t_i, torsion) in enumerate(torsions):
            self.print_verbose(t_i)
            self.print_verbose(torsion)
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
            self.hessian_helper_V2V3(hessian, doubles, dGd, const)
        return hessian

    def build_V4_hessian(self, masses, resis):
        n = len(masses)
        bij = self.compute_Bij(masses, resis)
        hessian = np.zeros((3 * n, 3 * n))
        axes = ["X", "Y", "Z"]
        indexes = ["i", "j"]
        for i in range(n):
            for j in range(i + 1, n):
                if self.is_long_range(i, j):
                    self.print_verbose((i, j))
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
                    self.hessian_helper_V1V4_optim(hessian, i, j, bcoord, const)
        self.bij = bij
        return hessian

    def get_weighted_surf_atomnum(self, sd, num_a, num_b, resi_a, resi_b, anums_dict):
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

    def get_total_surf_atomnum(self, sd, anums_list, i, j, resi_a, resi_b, anums_dict):
        if resi_a == "LIG" and resi_b == "ASP":
            dum = 0
        surf_tot = 0
        for num_a in anums_list[i]:
            for num_b in anums_list[j]:
                try:
                    if num_b in sd[num_a]:
                        surf_tot += self.get_weighted_surf_atomnum(sd, num_a, num_b, resi_a, resi_b, anums_dict)
                    if num_a in sd[num_b]:
                        surf_tot += self.get_weighted_surf_atomnum(sd, num_b, num_a, resi_b, resi_a, anums_dict)
                except:
                    print("Error in surface computation, residues {0} and {1}, file {2}".format(resi_a, resi_b,
                                                                                                self.mol.pdb_file))
                    raise
        return surf_tot

    def test_list_dict(self, alist, adict):
        for l in alist:
            s = ""
            for i in l:
                a = adict[i]
                s += "{0}|{1}|{2} ".format(a.name, a.resiname, a.resinum)
            print(s)

    def compute_Bij(self, masses, resis):
        """ Computes the Bij term in the ENCoM potential, which is the
            modulating factor of the long-range interaction (V4) term calculated
            according to atomic surface complementarity.
        """
        sd = self.mol.get_surface_dict()
        n = len(masses)
        bij = np.zeros((n, n))
        atomnums_list = self.make_atomnums_list(masses, resis)
        atomnums_dict = self.make_atomnums_dict(resis)
        self.print_verbose(self.atypes_dict)
        for i in range(n):
            ma = masses[i]  # mass_a
            ka = ma[4]  # key_a
            resiname_a = ka.split('|')[0].split('.')[0]
            for j in range(i + 1, n):
                mb = masses[j]
                kb = mb[4]
                resiname_b = kb.split('|')[0].split('.')[0]
                try:
                    surf_tot = self.get_total_surf_atomnum(sd, atomnums_list, i, j, resiname_a, resiname_b, atomnums_dict)
                except:
                    e = sys.exc_info()[0]
                    print("Error in surface computation, file {0}".format(self.mol.pdb_file))
                    raise e
                bij[i][j] = surf_tot
                bij[j][i] = surf_tot
        self.print_verbose(self.not_there)
        return bij

    def make_atomnums_list(self, masses, resis):
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

    def make_atomnums_dict(self, resis):
        """ Makes a dictionary associating each atom number to an Atom object.
        """
        numdict = dict()
        for r in resis:
            for a in resis[r]:
                numdict[resis[r][a].num] = resis[r][a]
        return numdict

    def get_weighted_surf(self, d_ab, resi_a, resi_b):

        surf = 0
        for k in d_ab:
            atom1, atom2 = k.split('|')
            try:
                i = self.atypes_dict[resi_a][atom1]
            except KeyError:
                self.not_there.add((resi_a, atom1))
                continue
            try:
                j = self.atypes_dict[resi_b][atom2]
            except KeyError:
                self.not_there.add((resi_b, atom2))
                continue
            surf += d_ab[k] * self.inter_mat[i][j]
        return surf

    def is_long_range(self, i, j):
        if i == j:
            return False
        return self.mol.is_disconnected(i, j)

    def sort_eigvecs_vals(self, vals, vecs):
        tosort = zip(vals, vecs)
        tosort = sorted(tosort)
        vals = [x[0] for x in tosort]
        vecs = [x[1] for x in tosort]
        return (vals, vecs)

    def str_hessian(self, h, x=None, y=None):
        s = ""
        if x is None:
            x = int(len(h) / 3)
        if y is None:
            y = int(len(h) / 3)
        for i in range(3 * x):
            for j in range(3 * y):
                # s += "{:>8.1f} ".format(h[i][j])
                s += "{:>9.2E} ".format(h[i][j])
            s += "\n"
        return s

    def str_sqr_matrix(self, m):
        s = ""
        n = len(m)
        for i in range(n):
            for j in range(n):
                s += "{:>8.3f} ".format(m[i][j])
            s += "\n"
        return s

    def filter_bij(self):
        n = len(self.bij)
        fbij = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                if self.is_long_range(i, j):
                    fbij[i][j] = self.bij[i][j]
                fbij[j][i] = fbij[i][j]
        return fbij

    def emulate_template(self):
        fbij = self.filter_bij()
        n = len(fbij)
        for i in range(n):
            for j in range(n):
                fbij[i][j] += 0.001
        return fbij

    def print_hessian(self, h):
        s = self.str_hessian(h)
        print(s)

    def write_hessians(self, prefix):
        for tup in zip([self.V1_H, self.V2_H, self.V3_H, self.V4_H, self.V_tot], ["1", "2", "3", "4", "tot"]):
            with open("{0}_V_{1}.hess".format(prefix, tup[1]), "w") as f:
                f.write(self.str_hessian(tup[0]))

    def write_bij(self, prefix):
        print(self.bij)
        with open("{0}_Bij.mat".format(prefix), "w") as f:
            f.write(self.str_sqr_matrix(self.emulate_template()))

    def write_vecs_df(self, n_vecs, outfile):
        with open(outfile, 'w') as f:
            for i in range(self.n):
                for k in range(3):
                    f.write("{} ".format(i))
                    for v in range(6, 6 + n_vecs):
                        f.write("{:>10.5f} ".format(self.eigvecs[3 * i + k][v]))
                    f.write("\n")

def distance(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError("I need 1 arg : pdb (protein) file")
    pdb_file = sys.argv[1]

    # test2 = ENCoM("closed_clean.pdb")
    test = ENCoM("test_medium.pdb")

    bfacts = test.compute_bfactors()
    entro = test.compute_vib_entropy_nussinov()
    test.write_dynamical_signature("test.df")
    print(entro)



# print(bfacts)
# print(entro)
# test.write_hessians(pdb_file[:-4])
# print(compare_bfacts(bfacts, pdb_file[:-4] + ".cov"))
# print(bfacts)
# print(test.compute_vib_entropy())
# print(vals[6:50])
# print(len(vals))

# mat1, vals1 = parse_mat("hessian.dat")
# vals1, vecs1 = np.linalg.eigh(mat1)
# tmp = []
# for i, vec1 in enumerate(vecs1):
# 	tmp.append(vecs1[:, i])
# vecs1 = tmp
# test.eigvecs = vecs1
# test.eigvals = vals1

# print(compare_versions(vals, vecs, pdb_file[:-4] + ".dat"))
# print(compare_versions(vals1, vecs1, pdb_file[:-4] + ".dat"))
# print(vecs[6][:100])

# test.write_vecs_df(10, "test_vecs.df")
# d = dict()
# for t in test.not_there:
# 	if not t[0] in d:
# 		d[t[0]] = []
# 	d[t[0]].append(t[1])
# print(d)

# test.print_hessian(test.V_tot)

# vals, vecs = test.solve()
# print(vals)
# test.write_vecs_df(20, "eigvecs2.df")
# print(vecs)
