Overlaps and cumulative overlap between normal modes and a conformational change
================================================================================

Overlaps
--------

The overlap metric is a measure of how well a given normal mode captures a given
conformational change. The following code computes the overlaps between the first
10 normal modes computed by ENCoM on the closed state of the citrate synthase
enzyme and the conformational change from closed to open::

	from nrgten.encom import ENCoM
	from nrgten.metrics import get_overlaps

	closed_cs = ENCoM("closed_clean.pdb")
	open_cs = ENCoM("open_clean.pdb", solve=False) # no need to solve the target
	overlaps = get_overlaps(closed_cs, open_cs, 10)
	print(overlaps)

When executed this prints::

[0.22349464198742516, 0.3799027474795493, 0.31226376941324435, 0.11926373811636343, 0.07141121329464464, 0.5434832619441955, 0.2850587863860328, 0.16140383753735463, 0.18518389540695462, 0.13542692494361588]

We can see that modes 2 and 6 have the best overlaps.

Cumulative overlap
------------------

It is also possible to compute the cumulative overlap from a given number of modes.
This gives a number between 0 and 1 describing how well the best combination of
the specified modes can describe the conformational change::

	from nrgten.encom import ENCoM
	from nrgten.metrics import cumulative_overlap

	closed_cs = ENCoM("closed_clean.pdb")
	open_cs = ENCoM("open_clean.pdb")
	print("Cumulative overlap 10 modes, from closed to open: {}".format(cumulative_overlap(closed_cs, open_cs, 10)))
	print("Cumulative overlap 10 modes, from open to closed: {}".format(cumulative_overlap(open_cs, closed_cs, 10)))

Which prints::

	Cumulative overlap 10 modes, from closed to open: 0.8754571996622168
	Cumulative overlap 10 modes, from open to closed: 0.8888323647003489

We can see that the first 10 normal modes computed by ENCoM capture both transitions
between the closed and open states of citrate synthase very well.


