Typical Uses
============

All the PDB files used in this guide can be found in the nrgten_examples_ GitHub
repository. If you haven't already, you can clone it with this command::

	git clone https://github.com/gregorpatof/nrgten_examples


.. _nrgten_examples: https://github.com/gregorpatof/nrgten_examples


The ENCoM model is implemented as a class, so anytime you wish to compute some
property of a macromolecule using ENCoM you need to create an ENCoM object::

	from nrgten.encom import ENCoM

	model = ENCoM("test_medium.pdb")

Here are some typical uses of ENCoM:

.. toctree::
   :maxdepth: 2
   
   dynamical_signatures
   conf_ensemble
   svib_mutations
   transit_probs
   rna


