from nrgens.encom import ENCoM
from nrgens.anm import ANM
import numpy as np
import sys
import glob
import os
from scipy.stats import pearsonr
from multiprocessing import Pool, current_process


def build_conf_ensemble(enm, modes_list, filename, step=0.5, max_displacement=2.0, max_conformations=10000):
    """ Has the same functionality (better name) as the old ENCoM executable build_grid_rmsd.
        Creates a conformational ensemble by making every combination of the selected modes at every given rmsd step,
        up to a total deformation of max_displacement for each mode (and in both directions). Writes the output as
        a multi-model pdb file.
        Important note: the mode indices in the modes_list are assumed to start at 1, with the first nontrivial mode
        thus being at index 7.
    """
    assert isinstance(modes_list, list)
    grid_side = 1 + (2*max_displacement / step) # length of one side of the grid
    if abs((grid_side - 0.999999999999) % 2) > 0.000001: # make sure that small floating point errors are ignored
        raise ValueError("build_conf_ensemble was executed with step={0} and max_displacement={1}. max_displacement " +
                         "has to be a multiple of step.".format(step, max_displacement))
    grid_side = int(round(grid_side))
    grid_side = int(grid_side)
    n_conf = int(grid_side ** len(modes_list)) # total number of points, or conformations, in the grid
    if n_conf > max_conformations:
        raise ValueError("build_conf_ensemble was executed with parameters specifying {0} conformations with " +
                         "max_conformations set at {1}. If you really want that many conformations, set " +
                         "max_conformations higher.".format(n_conf, max_conformations))
    eigvecs_list = []
    for mode_index in modes_list:
        if mode_index < 7:
            raise ValueError("build_conf_ensemble was run with mode index {0} in the modes_list, which is a trivial " +
                             "(rotational/translational) motion.".format(mode_index))
        eigvecs_list.append(np.copy(enm.eigvecs[mode_index+5]))
        modermsd = rmsd_of_3n_vector(eigvecs_list[-1])
        eigvecs_list[-1] /= modermsd
    with open(filename, "w") as fh:
        fh.write("REMARK Conformational ensemble written by nrgens.metrics.build_conf_ensemble()\n" +
                 "REMARK from the nrgens Python package, copyright Najmanovich Research Group 2020\n" +
                 "REMARK This ensemble contains {0} conformations\n".format(n_conf))
        for i in range(n_conf):
            write_one_point(enm, eigvecs_list, i, grid_side, step, fh)
        fh.write("END\n")


def write_one_point(enm, eigvecs_list, conf_n, grid_side, step, fh):
    """ Helper function for build_conf_ensemble. conf_n is the conformation number from 0 to n-1. This function computes
        contributions from every mode automatically, translates the enm coordinates, and resets it back to original
        after having written one model to the filehandle fh.
    """
    nsteps_list = []
    for i in range(len(eigvecs_list)):
        nsteps = conf_n % grid_side
        conf_n -= nsteps
        conf_n /= grid_side
        nsteps_list.append(nsteps - (grid_side - 1) / 2)
    t_vect = np.zeros(len(eigvecs_list[0]))
    for vec, nsteps in zip(eigvecs_list, nsteps_list):
        t_vect += vec * nsteps * step
    enm.translate_3n_vector(t_vect)
    write_model(enm, int(conf_n+1), fh)
    enm.translate_3n_vector(-t_vect)


def get_pa_pi(enm_a, enm_i, beta=1, alignment=None):
    """ Computes Pa and Pi, the respective 'probabilities' of reaching the active state from the inactive state and vice
        versa. Beta is the Boltzmann scaling factor.
        Returns Pa, Pi, R (R = Pi/Pa, the inactivation ratio)
        Alignment is a list of 2 lists of matching indices. Example:
        The alignment of sequences
        enm_a: -XYY--Z
        enm_i: ZXYYXXZ
        would give alignment=[[0,1,2,3],[1,2,3,6]]
    """
    vals_a = enm_a.eigvals[6:]
    vals_i = enm_i.eigvals[6:]
    if alignment is None:
        assert len(vals_a) == len(vals_i)
    else:
        assert len(alignment[0]) == len(alignment[1])
        assert len(vals_a) >= len(alignment[0])*3-6 and len(vals_i) >= len(alignment[0])*3-6
    pjs_a = np.exp(-1 * np.array(vals_a)/beta)
    pjs_i = np.exp(-1 * np.array(vals_i)/beta)
    pjs_a /= np.sum(pjs_a)
    pjs_i /= np.sum(pjs_i)
    if alignment is None:
        overlaps_a = overlap(enm_i, enm_a, len(vals_a))
        overlaps_i = overlap(enm_a, enm_i, len(vals_a))
    else:
        overlaps_a = overlap_alignment(enm_i, enm_a, len(alignment[0])*3-6, [alignment[1], alignment[0]])
        overlaps_i = overlap_alignment(enm_a, enm_i, len(alignment[0])*3-6, alignment)
    n = len(overlaps_a)
    pa = np.sum(np.array([x[0] * x[1] for x in zip(overlaps_a, pjs_a[:n])]))
    pi = np.sum(np.array([x[0] * x[1] for x in zip(overlaps_i, pjs_i[:n])]))
    # print(overlaps_a)
    # print(overlaps_i)
    # pa = np.sum(np.array([x[0] * x[1] for x in zip(overlaps_a, pjs_a)]))
    # pi = np.sum(np.array([x[0] * x[1] for x in zip(overlaps_i, pjs_i)]))
    return pa, pi, pi/pa


def fit(reference, target, filter=None):
    """ Fits the target to the reference with optimal rotation and translation. Returns the vector of remaining
        conformational change.
    """
    ref_xyz = reference.get_filtered_xyz(filter)
    rmsd, r, transvec = rmsd_kabsch(ref_xyz, target.get_filtered_xyz(filter))
    target.rotate(r)
    target.translate_xyz(transvec)
    targ_xyz = target.get_filtered_xyz(filter)
    conf_change = np.array(listify(np.subtract(targ_xyz, ref_xyz)))
    target.translate_xyz(-transvec)
    target.rotate(np.transpose(r))
    return conf_change


def fit_alignment(reference, target, alignment, filter):
    """ Fits the target to the reference, by first keeping the masses that have centers in the filter set, and then
        applying the alignment, which is 2 lists of corresponding indices [ref, target].
    """
    assert len(alignment) == 2 and len(alignment[0]) == len(alignment[1])

    ref_xyz_before = reference.get_filtered_xyz(filter) # before applying alignment
    targ_xyz_before = target.get_filtered_xyz(filter)
    n = len(alignment[0])
    ref_xyz = np.zeros((n, 3))
    targ_xyz = np.zeros((n, 3))
    for i in range(n):
        ref_xyz[i] = ref_xyz_before[alignment[0][i]]
        targ_xyz[i] = targ_xyz_before[alignment[1][i]]
    rmsd, r, transvec = rmsd_kabsch(ref_xyz, targ_xyz)
    target.rotate(r)
    target.translate_xyz(transvec)

    # new target xyz after translation and rotation
    targ_xyz_before = target.get_filtered_xyz(filter)
    targ_xyz = np.zeros((n, 3))
    for i in range(n):
        targ_xyz[i] = targ_xyz_before[alignment[1][i]]
    conf_change = np.array(listify(np.subtract(targ_xyz, ref_xyz)))
    target.translate_xyz(-transvec)
    target.rotate(np.transpose(r))
    return conf_change


def rmsd_alignment(reference, target, alignment, filter):
    assert len(alignment) == 2 and len(alignment[0]) == len(alignment[1])

    ref_xyz_before = reference.get_filtered_xyz(filter)  # before applying alignment
    targ_xyz_before = target.get_filtered_xyz(filter)
    n = len(alignment[0])
    ref_xyz = np.zeros((n, 3))
    targ_xyz = np.zeros((n, 3))
    for i in range(n):
        ref_xyz[i] = ref_xyz_before[alignment[0][i]]
        targ_xyz[i] = targ_xyz_before[alignment[1][i]]
    rmsd, r, transvec = rmsd_kabsch(ref_xyz, targ_xyz)
    return rmsd


def fit_svd(reference, target, n_modes):
    """ This function is an attempt at replicating the fit_svd executable from
        ENCoM 1. The name could be more informative...
        Parameters:
        from_structure: ENCoM object to which the eigenvectors will be applied
        to_structure: ENCoM object of the target structure
        n_modes: number of normal modes to consider
    """
    conf_change = fit(reference, target)
    # eigvecs = reference.eigvecs_mat[:,6:n_modes+6]
    eigvecs = reference.eigvecs_mat
    x = np.dot(np.linalg.inv(eigvecs), conf_change)
    return x[6:n_modes+6]


def fit_svd_alignment(reference, target, n_modes, alignment, filter=None):
    """ Same as fit_svd, but with supplied alignment between reference and target structures (in the form of 2 lists of
        matching indices, applied to the masses after applying the filter, which is a set of center atoms to keep).
    """
    conf_change = fit_alignment(reference, target, alignment, filter)
    eigvecs = reference.get_filtered_eigvecs_mat(alignment[0], filter)
    u, s, v = np.linalg.svd(eigvecs)
    x = np.dot(np.dot(np.transpose(v), np.diag(s)), np.dot(np.transpose(u), conf_change))
    return x[6:n_modes+6]


def fit_svd_svd(reference, target, n_modes):
    conf_change = fit(reference, target)
    eigvecs = reference.eigvecs_mat
    u, s, v = np.linalg.svd(eigvecs)
    x = np.dot(np.dot(np.transpose(v), np.diag(s)), np.dot(np.transpose(u), conf_change))
    return x[6:n_modes+6]


def pca_ensemble(enm, macromols_list=None, variance_to_explain=0.99, filter=None):
    """ Principal component analysis on an ensemble of structures. Returns the necessary PCAs to explain the target
        proportion of variance and their respective proportions of variance explained as:
        (explained variances, PCAs) (1 PCA per row).
    """
    # TODO : find a way to use fittings that are reversed and keep only the conf changes
    if macromols_list is None:
        macromols_list = enm.mols[1:]
    refmol = enm.mol
    if filter is None:
        coords = np.zeros((3*len(refmol.masses), len(macromols_list)+1))
    else:
        coords = np.zeros((3*refmol.get_filtered_n(filter), len(macromols_list) + 1))
    for i, mol in enumerate([refmol] + macromols_list):
        if i > 0:
            if not mol.solved:
                mol.solve()
            fit_to_reference_macromol_DEPRECATED(refmol, mol)
        if filter is None:
            coords[:,i] = mol.get_3n_vector()
        else:
            coords[:, i] = mol.get_filtered_3n_vector(filter)
    # Centering the data
    means = np.zeros((len(coords)))
    for i in range(len(coords)):
        means[i] = np.mean(coords[i])
        coords[i] -= means[i]
    covmat = np.cov(coords)
    eigvals, eigvecsmat = np.linalg.eigh(covmat)
    eigvecs = np.transpose(eigvecsmat)
    s = np.sum(eigvals)
    proportion_variance = eigvals/s
    total_var = 0
    n_significant = None
    for i in range(len(proportion_variance)-1, -1, -1):
        total_var += proportion_variance[i]
        if total_var > variance_to_explain:
            n_significant = len(proportion_variance) - i
            break
    return np.flip(proportion_variance, axis=0)[0:n_significant], np.flip(eigvecs, axis=0)[0:n_significant]


def cumulative_overlap(ref_vector, vectors_list):
    """ Cumulative overlap, as defined in Zimmerman and Jernigan 2014, Eq. 4.
    """
    s = 0
    ref_norm = np.linalg.norm(ref_vector)
    for vec in vectors_list:
        s += np.square(np.absolute(np.dot(ref_vector, vec))/(np.linalg.norm(vec)*ref_norm))
    return np.sqrt(s)


def cumulative_overlap_alignment(reference, target, n_modes, alignment, filter=None):
    conf_change = fit_alignment(reference, target, alignment, filter)
    eigvecs = np.transpose(reference.get_filtered_eigvecs_mat(alignment[0], filter))
    assert len(conf_change) == len(eigvecs[0])
    return cumulative_overlap(conf_change, eigvecs[6:6+n_modes])


def rmsip(ref_vectors, vectors_list):
    """ Root mean square inner product, as defined in Leo-Macis et al. 2005.
    """
    s = 0
    for r in ref_vectors:
        for v in vectors_list:
            s += np.square(np.dot(r, v))
    return np.sqrt(s/len(ref_vectors))


def fit_to_reference_macromol_DEPRECATED(ref, target):
    rmsd, r, transvec = rmsd_kabsch(ref.masscoords, target.masscoords)
    target.rotate(r)
    target.translate_xyz(transvec)
    return rmsd


def write_model(enm, count, fh):
    """ Writes one model to the given filehandle, with model number count.
    """
    fh.write("MODEL     {:>4}\n".format(count))
    enm.write_to_filehandle(fh)
    fh.write("ENDMDL\n")


def motion(enm, mode_index, stepsize, maxrmsd, filename):
    """ Writes a PDB file displaying the motion of a normal mode, using MODEL records.
        Important note: mode_index starts at 1, which is the first nontrivial motion (7th normal mode).
    """
    mode = np.copy(enm.eigvecs[mode_index+5])
    modermsd = rmsd_of_3n_vector(mode)
    mode *= stepsize/modermsd
    nsteps = int(maxrmsd/stepsize)
    count = 1
    offset = 0
    with open(filename, "w") as fh:
        write_model(enm, count, fh)
        for i, sign in enumerate([1, -1, 1, -1]):
            mode *= sign
            if i == 3:
                offset = 1
            for i in range(nsteps - offset):
                enm.translate_3n_vector(mode)
                write_model(enm, count, fh)
                count += 1


def rmsd_of_3n_vector(vec):
    dists = np.zeros((int(len(vec)/3)))
    for i in range(0, len(vec), 3):
        dists[int(i/3)] = vec[i]**2 + vec[i+1]**2 + vec[i+2]**2
    return np.sqrt(np.mean(dists))


def overlap(reference, target, n_modes):
    """ Returns a list of overlaps for each normal mode, going from reference to target. reference and target
        are ENCoM objects, and reference obviously needs to be solved.
    """
    conf_change = fit(reference, target)
    eigvecs = reference.eigvecs
    assert len(conf_change) == len(eigvecs[0])
    overlaps = []
    for i in range(6, n_modes+6):
        vec = np.array(eigvecs[i])
        overlaps.append(np.abs(np.dot(vec, conf_change))/(np.linalg.norm(vec)*np.linalg.norm(conf_change)))
    return overlaps


def overlap_alignment(reference, target, n_modes, alignment, filter=None):
    """ Returns a list of overlaps for each normal mode, going from reference to target. reference and target
        are ENM objects, and reference obviously needs to be solved.
    """
    conf_change = fit_alignment(reference, target, alignment, filter)
    eigvecs = np.transpose(reference.get_filtered_eigvecs_mat(alignment[0], filter))
    assert len(conf_change) == len(eigvecs[0])
    overlaps = []
    n = n_modes + 6
    if n > len(eigvecs):
        n = len(eigvecs)
    for i in range(6, n):
        vec = np.array(eigvecs[i])
        overlaps.append(np.abs(np.dot(vec, conf_change))/(np.linalg.norm(vec)*np.linalg.norm(conf_change)))
    return overlaps


def listify(matNx3):
    """ Takes an N x 3 matrix and makes it into a list (eigenvector-like).
    """
    l = []
    for row in matNx3:
        assert len(row) == 3
        for e in row:
            l.append(e)
    return l


def rmsd_kabsch(q, p):
    """ Kabsch algorithm for RMSD. Returns RMSD, rotation matrix and translation
        vector to optimally superpose p into q. See https://en.wikipedia.org/wiki/Kabsch_algorithm
    """
    assert len(p) == len(q)  # make sure that p and q are Nx3 matrices (x,y,z coordinates of N points)
    for i in range(len(p)):
        assert len(p[i]) == 3
        assert len(q[i]) == 3
    p = clone_nx3(p)
    q = clone_nx3(q)
    p_centroid = get_centroid(p)
    q_centroid = get_centroid(q)
    p_centered = subtract_centroid(p, p_centroid)
    q_centered = subtract_centroid(q, q_centroid)
    h = get_cross_covariance_matrix(p_centered, q_centered)
    u, s, vt = np.linalg.svd(h)
    v = np.transpose(vt)
    ut = np.transpose(u)
    d = np.linalg.det(np.dot(v, ut))
    x = np.zeros((3, 3))
    x[0][0] = 1
    x[1][1] = 1
    x[2][2] = d
    r = np.dot(np.dot(v, x), ut)
    rotated_centered_p = np.dot(p_centered, np.transpose(r))
    translation_vector = q_centroid - p_centroid
    rmsd = np.sqrt(np.mean(np.square(get_distance_vector(q_centered, rotated_centered_p))))
    return rmsd, r, translation_vector


def get_distance_vector(q, p):
    assert len(p) == len(q)
    d = np.zeros((len(p)))
    for i in range(len(p)):
        d[i] = np.linalg.norm(p[i]-q[i])
    return d


def get_cross_covariance_matrix(p, q):
    return np.dot(np.transpose(p), q)


def subtract_centroid(p, centroid):
    new_p = np.zeros((len(p), 3))
    for i in range(len(p)):
        for j in range(3):
            new_p[i][j] = p[i][j] - centroid[j]
    return new_p


def clone_nx3(p):
    new_p = np.zeros((len(p), 3))
    for i in range(len(p)):
        for j in range(3):
            new_p[i][j] = p[i][j]
    return new_p


def get_centroid(p):
    """ Returns the centroid of a set of point p (in the form of an Nx3 matrix).
    """
    n = len(p)
    xyz = np.array([0, 0, 0])
    for i in range(n):
        xyz = xyz + np.array([p[i][0], p[i][1], p[i][2]])
    return xyz/n


def test_citrate_synthase():
    closed_file = "unit-tests/citrate_synthase/closed_clean.pdb"
    open_file = "unit-tests/citrate_synthase/open_clean.pdb"
    closed = ENCoM(closed_file)
    open = ENCoM(open_file, solve=False)
    # rmsd, r, transvec = rmsd_kabsch(closed.get_xyz(), open.get_xyz())
    # print(rmsd, r, transvec)
    # open.rotate(r)
    # open.translate_xyz(transvec)
    # open.write_to_file("unit-tests/citrate_synthase/open_unto_closed.pdb")
    # print(overlap(closed, open, 10))
    # closed.write_to_file("unit-tests/citrate_synthase/closed_after_over.pdb")
    # open.write_to_file("unit-tests/citrate_synthase/open_after_over.pdb")
    print(fit_svd(closed, open, 10))
    print(fit_svd_svd(closed, open, 10))


def test_nmr(enm, kwlist=None):
    """ Returns rmsip, nco.
    """
    print(enm)
    insulin_file = "unit-tests/jernigan_renamed_cleaned/JR03.pdb"
    if kwlist is None:
        insulin = enm(insulin_file)
    else:
        print(kwlist)
        insulin = enm(insulin_file, **kwlist)
    vars, pcas = pca_ensemble(insulin)
    print(rmsip(pcas, insulin.eigvecs[6:16]))
    var_overlap = np.zeros((len(pcas), 2))
    for i in range(len(pcas)):
        var_overlap[i] = vars[i], cumulative_overlap(pcas[i], insulin.eigvecs[6:16])
    # print(var_overlap)
    print(np.sum([x[0]*x[1] for x in var_overlap]))
    # print(vars)
    # print(pcas)


def test_jernigan(filename, enm, kwlist=None):
    print(enm)
    if kwlist is None:
        model = enm(filename)
    else:
        print(kwlist)
        model = enm(filename, **kwlist)
    vals, vecs = pca_ensemble(model, filter={"P"})
    return cumulative_overlap(vecs[0], model.get_n_filtered_eigvecs_orthonorm({"P"}, 20))


def test_entropy(factor=1):
    enc_1caa = ENCoM("unit-tests/delta_s/1caa_trimmed.pdb")
    ref_ent = enc_1caa.compute_vib_entropy_nussinov(factor=factor)
    pdbs = glob.glob("unit-tests/delta_s/*.pdb")
    old_nussinov = np.zeros((3, 4))
    for i, pdb in enumerate(pdbs):
        print(pdb)
        enc = ENCoM(pdb, ignore_hetatms=True)
        initial, nussi = enc.compute_vib_entropy(), enc.compute_vib_entropy_nussinov(factor=factor)
        print(initial, nussi, nussi/ref_ent)
        old_nussinov[0][i] = enc.compute_vib_entropy()
        old_nussinov[1][i] = enc.compute_vib_entropy_nussinov(factor=factor)
        old_nussinov[2][i] = enc.get_n_masses()
    return old_nussinov


def test_entropy_bfacts(factor=1):
    basedir = "unit-tests/filtered_bfact_pdbs"
    with open("/Users/Paracelsus/school/PhD/h20/projet/delta_s/old_vs_nussinov.txt") as f:
        lines = f.readlines()
    codes = []
    for line in lines:
        codes.append(line.split()[0])
    old_nussinov = np.zeros((4, 20))
    for i, code in enumerate(codes):
        if i > 19:
            break
        enc = ENCoM("{0}/{1}.pdb".format(basedir, code), ignore_hetatms=True, use_pickle=True)
        old_nussinov[0][i] = enc.compute_vib_entropy()
        old_nussinov[1][i] = enc.compute_vib_entropy_nussinov(factor=factor)
        old_nussinov[2][i] = enc.compute_vib_enthalpy_nussinov()
        old_nussinov[3][i] = enc.get_n_masses()
    return old_nussinov


def run_mir125a(filename):
    if os.path.isfile(filename[:-4] + ".sig"):
        return
    enc = ENCoM(filename)
    enc.compute_bfactors()
    enc.write_dynamical_signature(filename[:-4] + ".sig")


def test_overlap_alignment():
    closed_fn = "/Users/Paracelsus/school/PhD/a20/projet/citrate_synthase/closed_clean.pdb"
    open_fn = "/Users/Paracelsus/school/PhD/a20/projet/citrate_synthase/open_clean.pdb"
    closed = ENCoM(closed_fn, use_pickle=True)
    open = ENCoM(open_fn, use_pickle=True)
    coeffs = fit_svd(open, closed, 10)
    overlaps = overlap(open, closed, 10)
    # print(open.get_n_masses())
    print(coeffs)
    print(overlaps)
    print(overlaps / coeffs)
    alignment = [x for x in range(1, 437, 2)]
    alignment = [alignment, alignment]
    coeffs_align = fit_svd_alignment(open, closed, 10, alignment)
    overlaps_align = overlap_alignment(open, closed, 10, alignment)
    print(coeffs_align)
    print(overlaps_align)
    print(overlaps_align / coeffs_align)
    all_resis = [x for x in range(437)]
    all_resis = [all_resis, all_resis]
    # print(cumulative_overlap_alignment(closed, open, 3*437-6, all_resis))
    # print(cumulative_overlap_alignment(closed, open, 3*437-6, alignment))
    print(overlap_alignment(closed, open, 3*437-6, alignment))

    overlaps = overlap_alignment(closed, open, 4, alignment)
    print(np.linalg.norm(np.array(overlaps)))

def test_pa_pi(use_pickle=True):
    closed_fn = "/Users/Paracelsus/school/PhD/a20/projet/citrate_synthase/closed_clean.pdb"
    open_fn = "/Users/Paracelsus/school/PhD/a20/projet/citrate_synthase/open_clean.pdb"
    open_trunc_fn = "/Users/Paracelsus/school/PhD/a20/projet/citrate_synthase/open_clean_truncated.pdb"
    closed = ENCoM(closed_fn, use_pickle=use_pickle)
    open = ENCoM(open_fn, use_pickle=use_pickle)
    open_trunc = ENCoM(open_trunc_fn, use_pickle=use_pickle)
    align = [x for x in range(430)]
    alignment = [align, align]

    print(get_pa_pi(closed, open_trunc, alignment=alignment))
    print(get_pa_pi(open_trunc, closed, alignment=alignment))

    print(get_pa_pi(closed, open))
    print(get_pa_pi(open, closed))

def test_pa_pi_2(use_pickle=True):
    closed_fn = "/Users/Paracelsus/school/PhD/a20/projet/citrate_synthase/closed_clean.pdb"
    open_fn = "/Users/Paracelsus/school/PhD/a20/projet/citrate_synthase/open_clean.pdb"
    open_trunc_fn = "/Users/Paracelsus/school/PhD/a20/projet/citrate_synthase/open_clean_truncated.pdb"
    closed = ENCoM(closed_fn, use_pickle=use_pickle)
    open = ENCoM(open_fn, use_pickle=use_pickle)
    open_trunc = ENCoM(open_trunc_fn, use_pickle=use_pickle)
    align = [x for x in range(430)]
    alignment = [align, align]

    print(get_pa_pi(open_trunc, closed, alignment=alignment))
    print(get_pa_pi(closed, open_trunc, alignment=alignment))

    print(get_pa_pi(open, closed))
    print(get_pa_pi(closed, open))

def half_life_compCan():
    d = 2419200000
    24386341640
    n = 0
    for i in range(500):
        print(n)
        n *= 2**(-1/7)
        # n *= .5
        print(n)
        print(0.010530*n)
        n += d
        # n += 7*d


def get_boltzmann_probs(energies, scale_factor):
    probs = np.zeros((len(energies)))
    for i, e in enumerate(energies):
        probs[i] = np.exp(-e/scale_factor)
    probs /= np.sum(probs)
    return probs




def test_boltzmann_squares():
    lambdas = [1, 4, 9, 16]
    lambdas_sqr = [1, 2, 3, 4]
    ref = get_boltzmann_probs(lambdas, 1)
    print(ref)
    for x in range(3280, 3291):
        y = x/10000
        t = get_boltzmann_probs(lambdas_sqr, y)
        print(pearsonr(ref, t))

        print(y, t)



if __name__ == "__main__":
    test_boltzmann_squares()
    # enc = ENCoM("test_medium.pdb")
    # build_conf_ensemble(enc, [7, 8], "test_conf.pdb", step=0.1, max_displacement=0.3)
    # if len(sys.argv) != 2:
    #     raise ValueError("I need 1 arg : pdb (protein) file")
    # pdb_file = sys.argv[1]
    # enc = ENCoM(pdb_file, unique_id=pdb_file)
    # ref = ENCoM("unit-tests/test_medium.pdb", unique_id="unit-tests/test_medium.pdb")
    # targ = ENCoM("unit-tests/test_medium_deformed.pdb", unique_id="unit-tests/test_medium_deformed.pdb")
    # print(fit_svd(ref, targ, 10))
    # print(fit_svd_alignment(ref, targ, 6, [[x for x in range(0, 4)], [x for x in range(0, 4)]]))
    # print(overlap_alignment(ref, targ, 6, [[x for x in range(0, 4)], [x for x in range(0, 4)]]))
    # test_overlap_alignment()
    # half_life_compCan()

    # test_pa_pi()
    # test_pa_pi_2()





    # mir125_dir = "unit-tests/mir125a/1d0022"
    # pdbs = glob.glob(mir125_dir + "/*.pdb")
    # for i in range(len(pdbs)):
    #     run_mir125a(pdbs[i])

    # p = Pool(8)
    # p.map(run_mir125a, pdbs)

    # enc = ENCoM(pdb_file)
    # enc.write_normalized_dynamical_signature("test_dyn.txt")


    # xray_rna = get_codes_new_db(False, False, True, True)
    # parse_new_db(xray_rna)
    # parse_new_db(["1H1K"])

    # parse_filtered_bfacts()
    # test_parsing("unit-tests/filtered_bfact_pdbs/1phb.pdb")




    # test = ENCoM(pdb_file)
    # deformed = ENCoM("unit-tests/test_medium_deformed.pdb", solve=False)
    # vec = np.copy(test.eigvecs[6])
    # for i in range(8):
    #     vec *= 10
    #     print(rmsd_of_3n_vector(vec))
    # n = test.get_n_masses()
    # vec = []
    # for i in range(n):
    #     vec.append([1, 2, 3])
    # test.translate(vec)
    # test.write_to_file("filewrite_test.pdb")

    # test_citrate_synthase()
    # test_nmr(ENCoM, kwlist={'interact_const': 24, 'kr': 2048, 'ktheta': 2048, 'kphi': 2048})



    # # # entropy testing, some weird things are going on!!! ###########################################################
    # b = 10 ** 10
    # p = 10 ** -10
    # old_nuss_bfact = test_entropy_bfacts(factor=10**12)
    # for i in range(11, 15):
    # old_nuss_thermo = test_entropy(factor=10**12)
    # # print(old_nuss_bfact)
    # # print(pearsonr(old_nuss_bfact[0], old_nuss_bfact[1]))
    # old_nuss_bfact[0] = old_nuss_bfact[0] / old_nuss_bfact[2]
    # old_nuss_bfact[1] = old_nuss_bfact[1] / old_nuss_bfact[2]
    # # print(old_nuss_bfact)
    # print(pearsonr(old_nuss_bfact[0], old_nuss_bfact[1]))
    #
    # old_nuss_thermo[0] = old_nuss_thermo[0] / old_nuss_thermo[2]
    # old_nuss_thermo[1] = old_nuss_thermo[1] / old_nuss_thermo[2]
    # print(pearsonr(old_nuss_thermo[0], old_nuss_thermo[1]))
    #
    # np.savetxt("unit-tests/delta_s/bfacts_corr4_newB.txt", np.transpose(old_nuss_bfact), delimiter=',')
    # np.savetxt("unit-tests/delta_s/thermo_corr_newB.txt", np.transpose(old_nuss_thermo), delimiter=',')

    # # # # Extended PD from 2 to 40 (was 2-12 in Jernigan paper) ######################################################
    # jerns = []
    # blacklisted_jerns = {0, 1, 5, 6, 11, 12} # reason for blacklisting : residues named UNK...
    # for i in range(1, 17):
    #     jerns.append("unit-tests/jernigan_renamed_cleaned/JR{:02}.pdb".format(i))
    # print(jerns)
    # results = np.zeros((10, 21))
    # count = 0
    # for i, jern in enumerate(jerns):
    #     if i in blacklisted_jerns:
    #         print("blacklisted...")
    #     else:
    #         print(i+1)
    #         results[count][0] = test_jernigan(jern, ENCoM)
    #         print(results[count][0])
    #         for pd in range(2, 42, 2):
    #             results[count][int(pd/2)] = test_jernigan(jern, ANM, kwlist={'power_dependence': pd})
    #             print(results[count][int(pd/2)])
    #         count += 1
    # with open('jernigan_results_ortho.txt', 'wb') as f:
    #     np.savetxt(f, results, delimiter=',')


    # for cut in range(7, 25):
    #     print(test_nmr(ANM, kwlist={'cut': cut}))


    # for k in range(10):
    #     test_nmr(ENCoM, kwlist={'interact_const': 24, 'kr': 2048, 'ktheta':2048, 'kphi':2048, 'power_dependenceV4':k})

    # test_nmr(ANM)
    # test = np.array([0.29564045844139564, 0.43492658283271873, 0.24394968177600132, 0.13148010177596031, 0.42024919492831947, 0.4147094089006102, 0.10213191265840724, 0.20776748061268779, 0.15718421268408028, 0.025876526844458])
    # print(np.sqrt(np.sum(test**2)))
    # motion(test, 1, .25, 4, "motion.pdb")

    # print(fit_svd(test, deformed, 10))
    # print(fit_svd_svd(test, deformed, 10))
