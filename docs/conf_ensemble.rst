Generating conformational ensembles
===================================

ENCoM can be used to generate a conformational ensemble for a protein of
interest. To do so, you simply need to call the build_conf_ensemble() method on
an ENCoM object::

	from nrgten.encom import ENCoM

	model = ENCoM("test_medium.pdb")
	model.build_conf_ensemble([7,8,9], "conf_ensemble.pdb")

The first argument is the list of normal modes to use to generate the ensemble.
In the example above, the first 3 modes are used. Note that 1 corresponds to
the first normal mode, but the first 6 normal modes are trivial
rotational/translational motions. Thus, index 7 is the first non-trivial
normal mode.

The second argument is simply the name of the output for the conformational
ensemble. The ensemble is outputted using the same convention as ensembles of
solutions for NRM experiments, using MODEL and ENDMDL statements to separate
the different conformations.

Here is the detailed documentation for the build_conf_ensemble() method:

.. autofunction:: nrgten.enm.ENM.build_conf_ensemble
	:noindex:
