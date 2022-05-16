Principal components analysis (PCA) and RMSIP
=============================================

PCA
---

When dealing with an ensemble of structures such as those produced by NMR experiments,
a useful metric is to extract the motions apparent from the ensemble
using principal components analysis (PCA). The principal components (PCs) can
then be compared with the low-frequency normal modes to see if they capture the
variance from the ensemble.

NMR structures of single-chain insulin (PDB id 2LWZ) will be used in this example.
To extract the PCs from the ensemble, simply use::

    from nrgten.encom import ENCoM
    from nrgten.metrics import pca_ensemble

    insulin_nmr = ENCoM("2lwz.pdb")
    variances, pcs = pca_ensemble(insulin_nmr, variance_to_explain=0.9)
    print(variances)

This prints::

    [0.51180055 0.2667673  0.08917762 0.06021049]

These values are the proportions of variance explained by each of the 4 PCs.
The reason there are 4 is that this number is sufficient to explain 90% of the
variance within the ensemble.

This is the classical way of computing principal components, however there can be
rotational-translational motions present in these PCs if the ensemble contains
more than 2 structures. To circumvent this, on can use the get_pcs_no_rot_tran function
to get non-rotational-translational PCs (nrt-PCs) and their associated proportions of
non-rotational-translational variance explained::

    from nrgten.encom import ENCoM
    from nrgten.metrics import pca_ensemble, get_pcs_no_rot_tran

    insulin_nmr = ENCoM("2lwz.pdb")
    variances, pcs = pca_ensemble(insulin_nmr, variance_to_explain=0.9)
    print(variances)
    variances_nrt, pcs_nrt = get_pcs_no_rot_tran(insulin_nmr, proportion_nrt_variance=0.9)
    print(variances_nrt)

This prints::

    [0.51180055 0.2667673  0.08917762 0.06021049]
    [0.42474628 0.11339343 0.08538987 0.07681148 0.071926 0.06255958 0.04324297 0.02712513]

We can see that we need more nrt-PCs to explain 90% of the nrt-variance in this case.

RMSIP
-----

The root mean square inner product can be used to measure how well an ensemble
of normal modes captures the same motions as an ensemble of PCs. To compute the
RMSIP between the PCs from the example above and the first 10 normal modes
computed with ENCoM, simpy use::

    from nrgten.encom import ENCoM
    from nrgten.metrics import pca_ensemble, rmsip

    insulin_nmr = ENCoM("2lwz.pdb")
    variances, pcs = pca_ensemble(insulin_nmr, variance_to_explain=0.9)
    print(rmsip(pcs, insulin_nmr.eigvecs[6:16]))

This gives::

    0.47582014355116004

Which means that the 10 normal modes agree with the first 4 PCs to a certain
extent.