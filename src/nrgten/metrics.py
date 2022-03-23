from nrgten.encom import ENCoM
from nrgten.anm import ANM
import numpy as np
# import sys
# import glob
# import os
# from scipy.stats import pearsonr
# from multiprocessing import Pool, current_process


def get_transit_probs(enm_a, enm_b, gamma=1, alignment=None):
    """Computes the transition 'probabilities' of reaching state B from state A and vice versa.

    Args:
        enm_a (ENM): The ENM object representing state A.
        enm_b (ENM): The ENM object representing state B.
        gamma (float, optional): The Boltzmann scaling factor.
        alignment (list, optional): list of 2 lists of matching indices. Example: The alignment of sequences
                                    enm_a: -XYY-Z and enm_b: ZXYYXZ would give alignment=[[0,1,2,3],[1,2,3,5]]

    Returns:
        tuple: tuple containing:
            p_a(float): the probability of reaching state A from state B

            p_b(float): the probability of reaching state B from state A
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
        overlaps_a = get_overlaps(enm_b, enm_a, len(vals_a))
        overlaps_b = get_overlaps(enm_a, enm_b, len(vals_a))
    else:
        overlaps_a = get_overlaps(enm_b, enm_a, len(alignment[0]) * 3 - 6, [alignment[1], alignment[0]])
        overlaps_b = get_overlaps(enm_a, enm_b, len(alignment[0]) * 3 - 6, alignment)
    n = len(overlaps_a)
    p_a = np.sum(np.array([x[0] * x[1] for x in zip(overlaps_a, pjs_a[:n])]))
    p_b = np.sum(np.array([x[0] * x[1] for x in zip(overlaps_b, pjs_b[:n])]))
    return p_a, p_b


def fit(reference, target, alignment=None, filter=None):
    """Fits the target to the reference.

    Only the masses that have centers in the filter set are kept (all are kept if filter is None). An alignment can also
    be applied to select corresponding masses. It is applied after filtering.

    Note:
        Only the conformational change vector is returned, and the target ENM object is left untouched. This vector
        does not include the 6 trivial 3-D degrees of freedom (rotational and translational).

    Warning:
        The target object is manipulated during this operation. Could lead to nasty effects if used naively during
        parallel computing.

    Args:
        reference (ENM): The reference ENM object to which the target will be fitted.
        target (ENM): The target ENM object which will be fitted to the reference.
        alignment (list, optional): list of 2 lists of matching indices. Example: The alignment of sequences
                                    enm_a: -XYY-Z and enm_b: ZXYYXZ would give alignment=[[0,1,2,3],[1,2,3,5]]
        filter (set, optional): set containing the names (str) of the center atoms for the masses to keep.

    Returns:
        np.ndarray: a vector representing the conformational change from reference to target
    """
    if alignment is None:
        assert reference.get_n_masses() == target.get_n_masses()
        n = reference.get_n_masses()
        alignment = [[x for x in range(n)], [x for x in range(n)]]
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
    target._rotate(r)
    target._translate_xyz(transvec)

    # new target xyz after translation and rotation
    targ_xyz_before = target.get_filtered_xyz(filter)
    targ_xyz = np.zeros((n, 3))
    for i in range(n):
        targ_xyz[i] = targ_xyz_before[alignment[1][i]]
    conf_change = np.array(listify(np.subtract(targ_xyz, ref_xyz)))
    target._translate_xyz(-transvec)
    target._rotate(np.transpose(r))
    return conf_change


def get_amplitudes_for_fit(reference, target, n_modes, alignment=None, filter=None):
    """This gives the amplitudes to apply to each eigenvector of the target to get as close as possible to the target.

    Args:
        reference (ENM): The reference ENM object to which the target will be fitted.
        target (ENM): The target ENM object which will be fitted to the reference.
        n_modes (int): The number of normal modes for which coefficients will be returned.
        alignment (list, optional): list of 2 lists of matching indices. Example: The alignment of sequences
                                    enm_a: -XYY-Z and enm_b: ZXYYXZ would give alignment=[[0,1,2,3],[1,2,3,5]]
        filter (set, optional): set containing the names (str) of the center atoms for the masses to keep.

    Returns:
        np.ndarray: a vector of amplitudes of length `n_modes`
    """
    if alignment is None:
        assert reference.get_n_masses() == target.get_n_masses()
        n = reference.get_n_masses()
        alignment = [[x for x in range(n)], [x for x in range(n)]]
    conf_change = fit(reference, target, alignment, filter)
    eigvecs = reference.get_filtered_eigvecs_mat(alignment[0], filter)
    u, s, v = np.linalg.svd(eigvecs)
    x = np.dot(np.dot(np.transpose(v), np.diag(s)), np.dot(np.transpose(u), conf_change))
    return x[6:n_modes+6]


def cumulative_overlap(reference, target, n_modes, alignment=None, filter=None):
    """Cumulative overlap, as defined in Zimmerman and Jernigan 2014, Eq. 4.

    This computes the cumulative overlap between the `n_modes` first normal modes of the reference and the
    conformational change going from reference to target. See Eq. 4 from
    `doi.org/10.1261/rna.041269.113 <https://doi.org/10.1261/rna.041269.113>`_

    Args:
        reference (ENM): The reference ENM object to which the target will be fitted.
        target (ENM): The target ENM object which will be fitted to the reference.
        n_modes (int): The number of normal modes for which the cumulative overlap will be returned.
        alignment (list, optional): list of 2 lists of matching indices. Example: The alignment of sequences
                                    enm_a: -XYY-Z and enm_b: ZXYYXZ would give alignment=[[0,1,2,3],[1,2,3,5]]
        filter (set, optional): set containing the names (str) of the center atoms for the masses to keep.

    Returns:
        float: the cumulative overlap, using the `n_modes` first normal modes
    """
    if alignment is None:
        assert reference.get_n_masses() == target.get_n_masses()
        n = reference.get_n_masses()
        alignment = [[x for x in range(n)], [x for x in range(n)]]
    conf_change = fit(reference, target, alignment, filter)
    eigvecs = np.transpose(reference.get_filtered_eigvecs_mat(alignment[0], filter))
    assert len(conf_change) == len(eigvecs[0])
    return _cumulative_overlap_helper(conf_change, eigvecs[6:6+n_modes])


def get_overlaps(reference, target, n_modes, alignment=None, filter=None):
    """Computes the overlaps between the `n_modes` from target and the conformational change.

    The conformational change is the vector of displacement going from `reference` to `target`, without
    translational/rotational degrees of freedom.

    Args:
        reference (ENM): The reference ENM object to which the target will be fitted.
        target (ENM): The target ENM object which will be fitted to the reference.
        n_modes (int): The number of normal modes for which the overlaps will be returned.
        alignment (list, optional): list of 2 lists of matching indices. Example: The alignment of sequences
                                    enm_a: -XYY-Z and enm_b: ZXYYXZ would give alignment=[[0,1,2,3],[1,2,3,5]]
        filter (set, optional): set containing the names (str) of the center atoms for the masses to keep.

    Returns:
        list: The list of overlaps, of length `n_modes`.
    """
    if alignment is None:
        assert reference.get_n_masses() == target.get_n_masses()
        n = reference.get_n_masses()
        alignment = [[x for x in range(n)], [x for x in range(n)]]
    conf_change = fit(reference, target, alignment, filter)
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


def pca_ensemble(enm, macromols_list=None, variance_to_explain=0.99, filter=None):
    """Principal component analysis on an ensemble of structures.

    Args:
        enm (ENM): An ENM object from which the starting structure will be taken. If the macromols_list optional
                   argument is None, this object must also contain at least one other state of the macromolecule (the
                   PDB file must be in NMR format with MODEL and ENDMDL records).
        macromols_list (list, optional): A list of Macromol objects which will be used to compute the PCs.
        variance_to_explain (float, optional): The target proportion of variance explained by the PCs.
        filter (set, optional): set containing the names (str) of the center atoms for the masses to keep.

    Warning:
        The structures other than the starting structures are restored to their initial states at the end but are
        temporarily disturbed during the computation. This could lead to nasty effects if parallel computing is
        attempted without care.

    Returns:
        tuple: tuple containing:
            proportion_variance(numpy.ndarray): the proportion of variance explained by each PC

            components(numpy.ndarray): the PCs in matrix notation, 1 PC per row
    """
    if macromols_list is None:
        macromols_list = enm.mols[1:]
    refmol = enm.mol
    refmol_coords = refmol.get_3n_vector()
    ref_mean = get_mean_xyz(refmol_coords)
    refmol.translate_xyz(ref_mean)

    if filter is None:
        coords = np.zeros((3*len(refmol.masses), len(macromols_list)+1))
    else:
        raise ValueError("filtering not yet supported for PCA ensembles")
        coords = np.zeros((3*refmol.get_filtered_n(filter), len(macromols_list) + 1))
    for i, mol in enumerate([refmol] + macromols_list):
        transvec, r = None, None
        if i > 0:
            if not mol.solved:
                mol.solve()
            rmsd, r, transvec = _fit_to_reference_macromol_DEPRECATED(refmol, mol)
        if filter is None:
            coords[:,i] = mol.get_3n_vector()
        else:
            raise ValueError("filtering not yet supported for PCA ensembles")
            coords[:, i] = mol.get_filtered_3n_vector(filter)
        if i > 0:
            mol.translate_xyz(-transvec)
            mol.rotate(np.transpose(r))
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
    if variance_to_explain == "all":
        return np.flip(proportion_variance, axis=0), np.flip(eigvecs, axis=0)
    total_var = 0
    n_significant = None
    for i in range(len(proportion_variance)-1, -1, -1):
        total_var += proportion_variance[i]
        if total_var > variance_to_explain:
            n_significant = len(proportion_variance) - i
            break
    return np.flip(proportion_variance, axis=0)[0:n_significant], np.flip(eigvecs, axis=0)[0:n_significant]


def get_mean_xyz(vector_3n):
    assert len(vector_3n) % 3 == 0
    mean_xyz = np.zeros(3)
    for offset in [0, 1, 2]:
        for i in range(0, len(vector_3n), 3):
            mean_xyz[offset] += vector_3n[i+offset]
    mean_xyz /= (len(vector_3n)/3)
    return mean_xyz


def get_pcs_no_rot_tran(enm_model, proportion_nrt_variance=0.99):
    assert len(enm_model.mols) > 1
    if enm_model.eigvals is None:
        enm_model.solve()
    variances, pcs = pca_ensemble(enm_model, variance_to_explain="all")
    rot_tran_vecs = enm_model.eigvecs[:6]
    nrt_vars = np.zeros(len(variances))
    nrt_vars_sum = np.zeros(len(variances))
    vals = np.zeros(len(variances))
    for i, pc in enumerate(pcs):
        val = rmsip([pc], rot_tran_vecs)
        # val = rmsip(rot_tran_vecs, [pc])
        vals[i] = val
        nrt_vars[i] = variances[i] * (1 - val)
    nrt_vars /= np.sum(nrt_vars)
    nrt_vars, pcs = reorder_pcs(pcs, nrt_vars)

    n_vecs = None
    for i in range(len(nrt_vars)):
        nrt_vars_sum[i] = np.sum(nrt_vars[:i+1])
        if nrt_vars_sum[i] >= proportion_nrt_variance:
            n_vecs = i+1
            break
    if n_vecs is None:
        raise ValueError("Not able to get to target variance proportion")
    max_vecs = len(pcs)
    n_target_vecs = n_vecs+6
    if n_target_vecs > max_vecs:
        n_target_vecs = max_vecs
    return nrt_vars[:n_vecs], _gram_schmidt(np.append(rot_tran_vecs, pcs, axis=0), n_target_vecs)[6:]


def reorder_pcs(pcs, nrt_vars):
    all_array = np.zeros((len(pcs), len(pcs[0])+1))
    all_array[:, 0] = nrt_vars
    all_array[:, 1:] = pcs
    sorted_array = all_array[np.argsort(-all_array[:, 0])]
    return sorted_array[:, 0], sorted_array[:, 1:]


def _gram_schmidt(vectors, target_n):
    # TODO : apply this to RMSIP to get rid of rot/trans contibution
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
        raise ValueError("Unable to produce {0} orthonormalized vectors".format(target_n))
    return np.array(basis)


def rmsip(ref_vectors, vectors_list):
    """Root mean square inner product, as defined in Leo-Macias et al. 2005.

    Link to the paper: `doi.org/10.1529/biophysj.104.052449 <https://doi.org/10.1529/biophysj.104.052449>`_

    Args:
        ref_vectors (np.ndarray): the reference vectors (PCs), in matrix form with one vector per row.
        vectors_list (np.ndarray): the vectors to test (eigenvectors describing normal modes) in matrix form with one
                                   vector per row.
    Returns:
        float: The root mean square inner product (RMSIP).
    """
    s = 0
    for r in ref_vectors:
        for v in vectors_list:
            s += np.square(np.dot(r, v))
    return np.sqrt(s/len(ref_vectors))


def nco(ref_vectors, ref_variances, vectors_list):
    ref_norms = np.zeros(len(ref_vectors))
    norms_list = np.zeros(len(vectors_list))
    for j, vec in enumerate(vectors_list):
        norms_list[j] = np.linalg.norm(vec)
    nco_result = []
    for i, ref_vec in enumerate(ref_vectors):
        nco_result.append([])
        ref_norms[i] = np.linalg.norm(ref_vec)
        overlaps_sq = np.zeros(len(vectors_list))
        for j, vec in enumerate(vectors_list):
            overlaps_sq[j] = (np.dot(vec, ref_vec)/(norms_list[j]*ref_norms[i]))**2
            nco_result[i].append(ref_variances[i]*np.sqrt(np.sum(overlaps_sq[:j+1])))
    return nco_result


def _fit_to_reference_macromol_DEPRECATED(ref, target):
    rmsd, r, transvec = rmsd_kabsch(ref.masscoords, target.masscoords)
    target.rotate(r)
    target.translate_xyz(transvec)
    return rmsd, r, transvec


def _cumulative_overlap_helper(ref_vector, vectors_list):
    s = 0
    ref_norm = np.linalg.norm(ref_vector)
    for vec in vectors_list:
        s += np.square(np.absolute(np.dot(ref_vector, vec))/(np.linalg.norm(vec)*ref_norm))
    return np.sqrt(s)



def _write_model(enm, count, fh):
    """ Writes one model to the given filehandle, with model number count.
    """
    fh.write("MODEL     {:>4}\n".format(count))
    enm.write_to_filehandle(fh)
    fh.write("ENDMDL\n")


def _motion(enm, mode_index, stepsize, maxrmsd, filename):
    """ Writes a PDB file displaying the motion of a normal mode, using MODEL records.
        Important note: mode_index starts at 1, which is the first nontrivial motion (7th normal mode).
    """
    mode = np.copy(enm.eigvecs[mode_index+5])
    _motion_3n_vector(enm, mode, stepsize, maxrmsd, filename)


def _motion_3n_vector(enm, vector_3n, stepsize, maxrmsd, filename):
    modermsd = _rmsd_of_3n_vector(vector_3n)
    vector_3n *= stepsize / modermsd
    nsteps = int(maxrmsd / stepsize)
    count = 1
    offset = 0
    with open(filename, "w") as fh:
        _write_model(enm, count, fh)
        for i, sign in enumerate([1, -1, 1, -1]):
            vector_3n *= sign
            if i == 3:
                offset = 1
            for i in range(nsteps - offset):
                enm._translate_3n_vector(vector_3n)
                _write_model(enm, count, fh)
                count += 1


def _rmsd_of_3n_vector(vec):
    dists = np.zeros((int(len(vec)/3)))
    for i in range(0, len(vec), 3):
        dists[int(i/3)] = vec[i]**2 + vec[i+1]**2 + vec[i+2]**2
    return np.sqrt(np.mean(dists))


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




if __name__ == "__main__":
    # test_boltzmann_squares()


    open_state = ENCoM("../../tests/open_clean.pdb")
    closed_state = ENCoM("../../tests/closed_clean.pdb")
    print(get_amplitudes_for_fit(open_state, closed_state, 10))
    # print(overlap(open_state, closed_state, 10))
    # print(overlap(closed_state, open_state, 10))
    # print(open_state.compute_bfactors()[:15])

