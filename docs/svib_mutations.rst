Vibrational entropy and the effect of mutations
===============================================

Among coarse-grained normal mode analysis methods, ENCoM has the unique ability
of being sensitive to the chemical nature of the amino acids, nucleic acids
and/or ligands.

This allows ENCoM to compute the change in vibrational entropy between WT and
mutant forms of the same protein. In the nrgten_examples_ repository, the FimA protein
from E. coli (PDB id 6R74) has had position 103 mutated from an isoleucine to
a tyrosine using MODELLER (`doi.org/10.1002/cpbi.3 <https://doi.org/10.1002/cpbi.3>`_). The difference in vibrational entropy
between the WT and mutant can be computed in this way::

    from nrgten.encom import ENCoM

    wt = ENCoM("6r74.pdb")
    tyr103 = ENCoM("6r74_TYR103.pdb")
    wt_entropy = wt.compute_vib_entropy(beta=1)
    tyr103_entropy = tyr103.compute_vib_entropy(beta=1)
    diff_entropy = wt_entropy - tyr103_entropy

.. _nrgten_examples: https://github.com/gregorpatof/nrgten_examples

The **diff_entropy** variable above contains the difference in vibrational
entropy between the WT and mutant forms of the protein. If we print its value
using::

    print(diff_entropy)

We see that it is around -0.02, which means that the mutant is more
flexible than the WT.


.. note::

    The beta parameter, which is the Boltzmann scaling factor, is set to 1 in
    the above examples because preliminary work has shown this value to give the
    best results when trying to predict experimental measurements of ΔΔG.

.. note::

    Since the ENCoM model is pseudo-physical, the vibrational entropy value does not
    have definite units (but it still represents a measurement of energy per temperature).
    It is however possible to select a beta parameter that
    gives a close match to experimental data, or to use the vibrational entropy
    value in a linear predictor which will match experimental data.


