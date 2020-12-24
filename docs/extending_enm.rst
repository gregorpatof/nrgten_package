Implementing custom elastic networks
====================================

The ENMs implemented as part of NRGTEN are built using an object-oriented approach.
This allows the quick and easy implementation of new ENMs by simply inheriting from
the ENM class.

As an example, let's say we want to implement an ENM called Random Elastic Network (RNM).
RNM uses the same uniform spring constant between pairs of masses as ANM, but instead
of connecting masses which are below a certain distance cutoff, it connects any pair
of masses with a probability alpha (a parameter of the model). Here is the code for RNM::

	from nrgten.enm import ENM
	import random
	import numpy as np

	class RNM(ENM):

	    def __init__(self, pdb_file, alpha=0.5, kr=1, solve=True, use_pickle=False,
	                 ignore_hetatms=False, atypes_list=None, massdef_list=None,
	                 solve_mol=True, one_mass=False):
	       self.alpha = alpha # the interaction probability
	       self.kr = kr # the spring constant
	       super().__init__(pdb_file, solve=solve, use_pickle=use_pickle,
	                        ignore_hetatms=ignore_hetatms, atypes_list=atypes_list,
	                        massdef_list=massdef_list, solve_mol=solve_mol,
	                        one_mass=one_mass)

	    def build_hessian(self):
	        if not self.mol.solved:
	            self.mol.solve()
	        masscoords = self.mol.masscoords
	        distmat = self.mol.distmat
	        n = len(masscoords)
	        hessian = np.zeros((3*n, 3*n))
	        for i in range(n):
	            for j in range(i+1, n):
	                if random.uniform(0, 1) <= self.alpha:
	                    dist_squared = distmat[i][j] ** 2

	                    # diagonal of the off-diagonal 3x3 element and update diagonal of diagonal element
	                    for k in range(3):
	                        val = 2 * self.kr * (masscoords[j][k] - masscoords[i][k]) ** 2 / dist_squared
	                        hessian[3 * i + k][3 * j + k] = -val
	                        hessian[3 * i + k][3 * i + k] += val
	                        hessian[3 * j + k][3 * j + k] += val

	                    # off-diagonals of the off-diagonal 3x3 element and update off-diagonal of diagonal element
	                    for (k, l) in ((0, 1), (0, 2), (1, 2)):
	                        val = 2 * self.kr * (masscoords[j][k] - masscoords[i][k]) * \
	                              (masscoords[j][l] - masscoords[i][l]) / dist_squared
	                        hessian[3 * i + k][3 * j + l] = -1 * val
	                        hessian[3 * i + l][3 * j + k] = -1 * val
	                        hessian[3 * i + k][3 * i + l] += val
	                        hessian[3 * j + k][3 * j + l] += val
	        for i in range(3 * n):
	            for j in range(i + 1, 3 * n):
	                hessian[j][i] = hessian[i][j]
	        return hessian

	    def build_from_pickle(self):
	        pass

	    def pickle(self):
	        pass

The **build_from_pickle** and **pickle** methods are required by the ENM abstract
base class so that the solved state of the object can be saved, greatly reducing
future computational load in case other properties are subsequently computed.
However in the case of our RNM the point is to have a random element in the
computation so there is no use in implementing these. See the ANM class documentation
for an example on how to implement these methods.

Apart from **build_from_pickle** and **pickle**, the only method RNM needs to
implement is **build_hessian**. In this example, the Hessian is built exactly as
in ANM, except that every pair of masses interacts according to the random probability alpha.

.. note::

	See the documentation for the ANM and ENM classes if you want more information
	about the optional arguments in the constructor.




