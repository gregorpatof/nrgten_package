Running ENCoM on RNA
====================

ENCoM was recently adapted to work on RNA molecules, using three beads per nucleotide as described
in the following article: `Sequence-sensitive elastic network captures dynamical features necessary for miR-125a maturation <https://doi.org/10.1101/2022.06.09.495567>`_.

The adaptation is included by default in NRGTEN, thus all models will assign three beads per nucleotide
in the input structure. RNA-protein complexes are also supported.

In the nrgten_examples_ repository, the WT and G22U variant of miR-125a are provided, called *mir125a_WT.pdb*
and *mir125a_G22U.pdb* respectively. The Entropic Signature (defined in the above-referenced article)
for each variant can be computed as follows::

    from nrgten.encom import ENCoM
    import numpy as np

    wt = ENCoM("mir125a_WT.pdb")
    g22u = ENCoM("mir125a_G22U.pdb")

    entrosig_wt = wt.compute_bfactors_boltzmann(beta=np.e**2.25)
    entrosig_g22u = g22u.compute_bfactors_boltzmann(beta=np.e**2.25)

The beta value used in this example is the one which led to the best performance (see referenced article).

.. note::
    You can safely ignore such warnings when computing Entropic Signatures: **RuntimeWarning: overflow encountered in float_power**.


.. _nrgten_examples: https://github.com/gregorpatof/nrgten_examples

