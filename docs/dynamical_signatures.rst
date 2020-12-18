Dynamical signatures
====================

The dynamical signature of a macromolecule (which is the
predicted b-factors at each position) can be computed and written to file using::

	from nrgten.encom import ENCoM

	model = ENCoM("test_medium.pdb")
	model.write_dynamical_signature("signature.txt")

This :signature.txt: file will contain the name of each mass in the system (in
the case of a protein, one per amino acid) in the first column and the relative
fluctuation of that mass in the second. The higher this value, the more that
mass is 'dynamic'.

If you compute a lot of dynamical signatures and do not want to write to a file
each time, you can also compute it directly within a Python program using::

	dyna_sig = model.compute_bfacts()

In that case, **dyna_sig** is the dynamical signature as a Python list.

