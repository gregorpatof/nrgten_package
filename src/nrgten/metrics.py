from nrgten.encom import ENCoM
from nrgten.anm import ANM
import numpy as np
import sys
import glob
import os
from scipy.stats import pearsonr
from multiprocessing import Pool, current_process


def get_transit_probs(enm_a, enm_b, gamma=1, alignment=None):
    """Computes the transition 'probabilities' of reaching state B from state A and vice versa.

    Args:
        enm_a (ENM): The ENM object representing state A.
        enm_b (ENM): The ENM object representing state B.
        gamma (float, optional): The Boltzmann scaling factor.
        alignment (list, optional): list of 2 lists of matching indices. Example: The alignment of sequences
                                    enm_a: -XYY--Z and enm_i: ZXYYXXZ would give alignment=[[0,1,2,3],[1,2,3,6]]

    Returns:
        tuple: tuple containing:
            p_a(float) the probability of reaching state A from state B
            p_b(float) the probability of reaching state B from state A
    """
    vals_a = enm_a.eigvals[6:]
    vals_b = enm_b.eigvals[6:]
    if alignment is None:
        assert len(vals_a) == len(vals_b)
    else:
        assert len(alignment[0]) == len(alignment[1])
        assert len(vals_a) >= len(alignment[0])*3-6 and len(vals_b) >= len(alignment[0])*3-6
    pjs_a = np.exp(-1 * np.array(vals_a) / gamma)
    pjs_b = np.exp(-1 * np.array(vals_b) / gamma)
    pjs_a /= np.sum(pjs_a)
    pjs_b /= np.sum(pjs_b)
    if alignment is None:
        overlaps_a = overlap(enm_b, enm_a, len(vals_a))
        overlaps_b = overlap(enm_a, enm_b, len(vals_a))
    else:
        overlaps_a = overlap_alignment(enm_b, enm_a, len(alignment[0]) * 3 - 6, [alignment[1], alignment[0]])
        overlaps_b = overlap_alignment(enm_a, enm_b, len(alignment[0]) * 3 - 6, alignment)
    n = len(overlaps_a)
    p_a = np.sum(np.array([x[0] * x[1] for x in zip(overlaps_a, pjs_a[:n])]))
    p_b = np.sum(np.array([x[0] * x[1] for x in zip(overlaps_b, pjs_b[:n])]))
    return p_a, p_b


def fit(reference, target, filter=None):
    """ Fits the target to the reference with optimal rotation and translation. Returns the vector of remaining
        conformational change.
    """
    ref_xyz = reference.get_filtered_xyz(filter)
    rmsd, r, transvec = rmsd_kabsch(ref_xyz, target.get_filtered_xyz(filter))
    target._rotate(r)
    target._translate_xyz(transvec)
    targ_xyz = target.get_filtered_xyz(filter)
    conf_change = np.array(listify(np.subtract(targ_xyz, ref_xyz)))
    target._translate_xyz(-transvec)
    target._rotate(np.transpose(r))
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
    target._rotate(r)
    target._translate_xyz(transvec)
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
                enm._translate_3n_vector(mode)
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
    return cumulative_overlap(vecs[0], model.get_filtered_eigvecs_mat(filter={"P"}, transpose=False, n_vecs=20, start=6))


def test_entropy(factor=1):
    enc_1caa = ENCoM("unit-tests/delta_s/1caa_trimmed.pdb")
    ref_ent = enc_1caa.compute_vib_entropy(factor=factor)
    pdbs = glob.glob("unit-tests/delta_s/*.pdb")
    old_nussinov = np.zeros((3, 4))
    for i, pdb in enumerate(pdbs):
        print(pdb)
        enc = ENCoM(pdb, ignore_hetatms=True)
        initial, nussi = enc.compute_vib_entropy(), enc.compute_vib_entropy(factor=factor)
        print(initial, nussi, nussi/ref_ent)
        old_nussinov[0][i] = enc.compute_vib_entropy()
        old_nussinov[1][i] = enc.compute_vib_entropy(factor=factor)
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
        old_nussinov[1][i] = enc.compute_vib_entropy(factor=factor)
        old_nussinov[2][i] = enc.compute_vib_enthalpy()
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





if __name__ == "__main__":
    # test_boltzmann_squares()

    open = ENCoM("../../tests/open_clean.pdb")
    closed = ENCoM("../../tests/closed_clean.pdb")
    print(overlap(open, closed, 10))
    print(overlap(closed, open, 10))
    # print(open.compute_bfactors()[:15])

