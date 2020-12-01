from nrgens.enm import ENM
import numpy as np
import os
import pickle


class ANM(ENM):
    """ Anisotropic Network Model. Has 3 parameters: cutoff, power dependence and spring constant. Spring constant just
        rescales fluctuations but doesn't change their relative values. Cutoff can be infinity (fully connected model).
    """
    def __init__(self, pdb_file, cut=float('inf'), power_dependence=0, kr=1, solve=True, use_pickle=False,
                 ignore_hetatms=False, atypes_list=None, massdef_list=None, solve_mol=True, one_mass=False):
        self.cut = cut
        self.pd = power_dependence
        self.kr = kr
        super().__init__(pdb_file, solve=solve, use_pickle=use_pickle, ignore_hetatms=ignore_hetatms,
                         atypes_list=atypes_list, massdef_list=massdef_list, solve_mol=solve_mol, one_mass=one_mass)

    def build_hessian(self):
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

    def get_pickle_file(self):
        param_string = "{0}_{1}_{2}".format(self.cut, self.pd, self.kr)
        if self.one_mass:
            return '.'.join(self.pdb_file.split('.')[:-1]) + ".ANM_{0}_1n.pickle".format(param_string)
        else:
            return '.'.join(self.pdb_file.split('.')[:-1]) + ".ANM_{0}.pickle".format(param_string)

    def build_from_pickle(self):
        pickle_file = self.get_pickle_file()
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
        self.reconstitute()
        return True

    def pickle(self):
        super().clear_info()
        pickle_file = self.get_pickle_file()
        with open(pickle_file, 'wb') as f:
            pickle.dump(self, f)

