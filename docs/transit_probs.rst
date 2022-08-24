Transition probabilities between conformational states
======================================================

ENCoM can be used to compute transition probabilities between two conformational
states of the same macromolecule. These probabilities do not represent actual
thermodynamic properties but give an indication of which state is favored by
the computed normal modes. For a detailed explanation of the metric, see the following
paper: `Modelling conformational state dynamics and its role on infection for SARS-CoV-2 Spike protein variants <https://doi.org/10.1101/2020.12.16.423118>`_.

In the nrgten_examples_ repository, the open and closed forms of the citrate
synthase enzyme have been cleaned up from water and solvent molecules and are
called *open_clean.pdb* and *closed_clean.pdb*. The transition probabilities
between the states can be computed in the following way::

	from nrgten.encom import ENCoM
	from nrgten.metrics import get_transit_probs

	open_cs = ENCoM("open_clean.pdb")
	closed_cs = ENCoM("closed_clean.pdb")
	prob_open, prob_closed = get_transit_probs(open_cs, closed_cs, gamma=1)

The **gamma** parameter is a Boltzmann scaling factor. Lower values of gamma
translate to a bigger contribution of the slow-frequency normal modes. The default value is 1,
but you may have to explore different values for your system and fit against experimental data like
described in our paper mentioned above.

If we print the probabilities::

	print("prob_open: {}, prob_closed: {}".format(prob_open, prob_closed))

We get the following values::

	prob_open : 0.28940086631043577, prob_closed : 0.43081399660689834

The aspartic acid in position 239 has been mutated to a lysine in both conformations
using MODELLER (`doi.org/10.1002/cpbi.3 <https://doi.org/10.1002/cpbi.3>`_). We can compute
the same transition probabilities using the mutated forms::

	from nrgten.encom import ENCoM
	from nrgten.metrics import get_transit_probs

	open_cs_mut = ENCoM("open_clean_LYS239.pdb")
	closed_cs_mut = ENCoM("closed_clean_LYS239.pdb")
	prob_open_mut, prob_closed_mut = get_transit_probs(open_cs_mut, closed_cs_mut, gamma=1)
	print("prob_open_mut: {}, prob_closed_mut: {}".format(prob_open_mut, prob_closed_mut))

This gives::

	prob_open_mut : 0.2903252666753628, prob_closed_mut : 0.4304065571363618

So we can see that the mutation slightly favors the open state. Here is the full
documentation for the get_transit_probs method:

.. autofunction:: nrgten.metrics.get_transit_probs
	:noindex:

 
.. _nrgten_examples: https://github.com/gregorpatof/nrgten_examples