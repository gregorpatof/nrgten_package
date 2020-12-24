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

	0.47544106008813264

Which means that the 10 normal modes agree with the first 4 PCs to a certain
extent.