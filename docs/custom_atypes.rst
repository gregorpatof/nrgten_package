Adding custom residues/ligands to ENCoM
=======================================

ENCoM uses the system of 8 atom types from `Sobolev *et al.* <https://doi.org/10.1002/(SICI)1097-0134(199605)25:1%3C120::AID-PROT10%3E3.0.CO;2-M>`_,
which are listed in the following table:

.. csv-table:: Legitimate (1) or illegitimate (0) contacts between the ENCoM atom types
	:file: sobolev_atypes.csv
	:header-rows: 1

Each residue that is to be considered by ENCoM needs to have every heavy atom
assigned to one of these 8 types. This is accomplished by configuration text files
with the **.atypes** extension. The default files already cover the standard amino acids
and nucleotides, in addition to some common modified nucleotides. Here are the
contents of the amino_acids.atomtypes file:

.. include:: ../src/nrgten/config/amino_acids.atomtypes
	:literal:

As you can see, the syntax is very simple. Each residue occupies one line. The residue
name comes first, then the **|** separator following by a list of atom name:atom type
pairs separated by commas.

The special instruction ADD_TO_ABOVE can be used to add atoms that are common to all
residues listed above that line. It is handy for residues such as nucleotides which
have many shared atoms among all species.

The second configuration file required by ENCoM is the mass definition file, another
text file with the **.masses** extension. For
residues for which only one mass per residue is needed, this file is very simple.
Here are the contents of the amino_acids.masses file:

.. include:: ../src/nrgten/config/amino_acids.masses
	:literal:

The **CONNECT** field tells ENCoM which atoms are connected when the residues form
polymers. It allows the automatic inference of covalent connections between residues.
In the case of ligands which do not form polymers, you can put any two atoms
there. When 1 mass per residue is used, **N_MASSES** is always set to 1.

The **CENTER** field defines which atom will be selected as the position of the
mass representing the whole residue. The **NAME** field is the name given to that
mass and is usually the same as the **CENTER** field in the case of 1 mass per residue.

When using multiple masses per residue, the mass definition file becomes a little
more complex. Here are the contents of the ribonucleic_acids.masses file:

.. include:: ../src/nrgten/config/ribonucleic_acids.masses
	:literal:

The **CONNECT_INIT** and **REDEFINE_RESIS** record are optional and must precede the 
**CONNECT** record. They are needed in cases where an atom from a residue
needs to be placed in the residue following it before the residue is divided in
masses. In the above example, the O3' atom is moved up in the chain of residues
to allow for one mass to represent the whole phosphate group. Such movements usually
change the connectivity between residues, so the **CONNECT_INIT** record defines
the connectivity before the redefinition and the **CONNECT** record defines it
after redefinition.

For each mass from *i* to *N*, a **MASS_i** record is needed, followed by the
3 indented fields: **NAME**, **ATOMS** and **CENTER**. Each mass needs a unique
name, a list of the atoms that are part of that mass and the name of the center
atom from which the mass will inherit coordinates.

.. note::
	See the ENM and ENCoM detailed documentation for details on how to pass these configuration
	files to the ENCoM object.



