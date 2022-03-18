import nrgten.parser as parser
import pyvcon.vcontacts_wrapper as vcontacts_wrapper
import os
import numpy as np


def get_macromol_list(pdb_file, atypes_dict, massdefs, verbose=False, skipmass_threshold=2,
                      cov_bond_length=2, ignore_hetatms=False, solve=False, unique_id=None):
    with open(pdb_file) as f:
        lines = f.readlines()
    multimodel_flag = False
    for line in lines:
        if line[:4] == "ATOM":
            break
        if line.strip()[:5] == "MODEL":
            multimodel_flag = True
            break

    # This is the single model (as in X-ray files) case
    if not multimodel_flag:
        return [Macromol(lines, atypes_dict, massdefs, verbose=verbose,
                         skipmass_threshold=skipmass_threshold, cov_bond_length=cov_bond_length,
                         ignore_hetatms=ignore_hetatms, solve=solve, pdb_file=pdb_file, unique_id=unique_id)]

    # This is the multi model (as in NMR files) case
    macromols = []
    start_i, stop_i = None, None
    count = 0
    for i, line in enumerate(lines):
        if line.strip()[:5] == "MODEL":
            assert stop_i is None
            start_i = i
        if line.strip()[:6] == "ENDMDL":
            assert start_i is not None
            stop_i = i
            if unique_id is None:
                mol_id = None
            else:
                mol_id = str(unique_id) + str(count)
            macromols.append(Macromol(lines[start_i:stop_i], atypes_dict, massdefs, verbose=verbose,
                                      skipmass_threshold=skipmass_threshold, cov_bond_length=cov_bond_length,
                                      ignore_hetatms=ignore_hetatms, solve=solve, pdb_file=pdb_file, unique_id=mol_id))
            count += 1
            start_i, stop_i = None, None
    assert start_i is None and stop_i is None
    return macromols

class Macromol:
    """ Class use to represent macromolecules. Infers the connectivity of
        residues from the atomic positions so that residue numbers and chain
        names have no effect on the data structure itself.
    """

    def __init__(self, pdb_lines, atypes_dict, massdefs, verbose=False, skipmass_threshold=2,
                 cov_bond_length=2, ignore_hetatms=False, solve=False, pdb_file=None, unique_id=None):
        self.pdb_file = pdb_file
        self.unique_id = unique_id
        self.dirpath = os.path.dirname(os.path.abspath(__file__))
        assert len(self.dirpath) > 0
        self.verbose = verbose
        self.skipmass_threshold = skipmass_threshold
        self.cov_bond_length = cov_bond_length
        self.atoms, self.resis, self.alt_flag = parser.parse_pdb_from_lines(pdb_lines, atypes_dict,
                                                                            ignore_hetatms=ignore_hetatms)
        self.index_atoms()
        if self.alt_flag:
            self.clear_alt_tags()
        self.atypes_dict, self.massdefs = atypes_dict, massdefs
        self.masses = self.compute_masses_general(self.atypes_dict, self.massdefs)
        self.masscoords = self.make_masscoords()
        self.solved = False
        self.coords_range, self.distmat, self.resi_cubes = None, None, None
        self.connect, self.bends, self.torsions, self.interact_pairs = None, None, None, None
        # TODO : adapt to compute only necessary information for ANM
        if solve:
            self.solve()

    def solve(self):
        self.distmat = self.make_distmat()
        self.resi_cubes = self.make_resi_cubes()
        self.print_verbose(", ".join([r for r in self.resis]))
        self.connect = self.compute_connect_general()
        self.bends = self.find_bends(self.connect)
        self.torsions = self.find_torsions(self.connect)
        self.interact_pairs = self.find_interacting_pairs(self.torsions, self.bends, self.connect)
        self.solved = True

    def clear_info(self):
        self.distmat, self.resi_cubes, self.connect, self.interact_pairs = None, None, None, None
        self.solved = False

    def get_unique_id(self):
        """ Returns the unique id for that macromol object (used for multithreading).
        """
        if self.unique_id is None:
            raise ValueError("No unique_id given for macromolecule from file {0}".format(self.pdb_file))
        return self.unique_id

    def update(self):
        """ Updates the x, y, z coordinates of masses to reflect those of their respective center atoms.
        """
        new_masses = []
        for m in self.masses:
            center_atom = self.resis[m[4]][m[3]]
            x, y, z = center_atom.xyz
            new_masses.append((x, y, z, m[3], m[4], m[5]))
        self.masses = new_masses
        self.masscoords = self.make_masscoords()

    def index_atoms(self):
        for i, a in enumerate(self.atoms):
            a.set_macromol_index(i)

    def make_masscoords(self):
        masscoords = np.zeros((len(self.masses), 3))
        for i, m in enumerate(self.masses):
            masscoords[i] = np.array(m[0:3])
        return masscoords

    def make_distmat(self):
        """ Returns the mass distance matrix.
        """
        n = len(self.masses)
        distmat = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                distmat[i][j] = distmat[j][i] = np.linalg.norm(self.masscoords[i] - self.masscoords[j])
        return distmat

    def make_resi_cubes(self):
        n = len(self.masses)
        inf = float('inf')
        mininf = float('-inf')
        resi_cubes = np.zeros((n, 6))
        delta = self.cov_bond_length / 2 + 0.01
        for i, r in enumerate(self.resis):
            resi = self.resis[r]
            x1, x2, y1, y2, z1, z2 = inf, mininf, inf, mininf, inf, mininf
            for a in resi:
                x, y, z = resi[a].xyz
                if x < x1:
                    x1 = x
                if x > x2:
                    x2 = x
                if y < y1:
                    y1 = y
                if y > y2:
                    y2 = y
                if z < z1:
                    z1 = z
                if z > z2:
                    z2 = z
            resi_cubes[i] = np.array([x1 - delta, x2 + delta, y1 - delta, y2 + delta, z1 - delta, z2 + delta])
        return resi_cubes

    def compute_masses_general(self, atypes_dict, massdefs):
        """ Computes a list of (x, y, z, center, residue, massdef) tuples, and
            computes the connectivity matrix.
        """
        massdefs_set = set()
        for r in massdefs:
            massdefs_set.add(massdefs[r])
        resi_redefinition = False
        for md in massdefs_set:
            if md.redefine_resis is not None:
                resi_redefinition = True
        if resi_redefinition:
            self.redefine_resis(atypes_dict, massdefs)
        masses = []
        newresis = {}
        for r in self.resis.keys():
            resname = r.split('|')[0]
            rd = self.resis[r]
            md = massdefs[resname]
            if md.n_masses == 1:
                c = md.centers[0]
                if c in rd:
                    a = rd[c]
                    k = self.make_mass_resikey(r, md.mass_names[0])
                    x, y, z = a.xyz
                    masses.append((x, y, z, c, k, md))
                    assert k not in newresis
                    newresis[k] = dict()
                    for atom in rd:
                        newresis[k][atom] = rd[atom]
                # TODO : find way of dealing with crappy residue portions (maybe modeRNA after all?)
                else:
                    raise ValueError("could not find center atom {0} in residue {1}, file {2}\n".format(c, r, self.pdb_file))
            else:
                for i in range(len(md.centers)):
                    (atoms, c, name) = md.get_atoms_center_name(i)
                    atoms = atoms.copy()
                    if not c in atoms:
                        print(r)
                        print(atoms)
                        continue
                    atoms.remove(c)
                    newresi = {}
                    for a in atoms:
                        if a in rd:
                            newresi[a] = rd[a]
                    if c in rd:
                        a = rd[c]
                        newresi[c] = a
                        k = self.make_mass_resikey(r, name)
                        x, y, z = a.xyz
                        masses.append((x, y, z, c, k, md))
                        assert k not in newresis
                        newresis[k] = newresi
                    elif len(newresi) <= self.skipmass_threshold:
                        continue
                    else:
                        raise ValueError("could not find center atom {0} in residue {1}\n".format(c, r))
        self.resis = newresis
        return masses

    def compute_connect_general(self):
        """ Computes the connectivity matrix and allows for different types of
            masses. Distance between atoms of <= 2 A is used for covalent bond
            definition.
        """
        n = len(self.masses)
        connect = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                if self.check_connect(i, j):
                    connect[i][j] = connect[j][i] = 1
        return connect

    def check_connect(self, i, j):
        if not self.overlapped_cubes(i, j):
            return False
        m1 = self.masses[i]
        m2 = self.masses[j]
        if m1[5] is not m2[5]:  # Check if mass definition is the same (otherwise, covalent connection is impossible)
            return False
        resi1 = self.resis[m1[4]]
        resi2 = self.resis[m2[4]]
        for key1 in resi1:
            atom1 = resi1[key1]
            for key2 in resi2:
                atom2 = resi2[key2]
                if np.linalg.norm(atom1.xyz - atom2.xyz) <= 2:
                    return True
        return False

    def overlapped_cubes(self, i, j):
        c1 = self.resi_cubes[i]
        c2 = self.resi_cubes[j]
        return not (c1[1] < c2[0] or c2[1] < c1[0] or c1[5] < c2[4] or c2[5] < c1[4] or c1[3] < c2[2] or c2[3] < c1[2])

    def make_mass_resikey(self, ogkey, name):
        sk = ogkey.split('|')
        sk[0] = sk[0] + ".{0}".format(name)
        return '|'.join(sk)

    def redefine_resis(self, atypes_dict, massdefs):
        """ Applies the REDEFINE_RESIS rules found in mass definition files.
            Does two passes: one to remove the atoms transferred, one to add
            them to the residues.
        """
        n = len(self.resis)
        resikeys = [x for x in self.resis.keys()]
        to_add = dict()
        # check all i -> j pairs for potential transfers of atoms
        for i in range(n):
            resi1 = self.resis[resikeys[i]]
            resname1 = resikeys[i].split('|')[0]
            md1 = massdefs[resname1]
            for j in range(n):  # unidirectional so iterate over all j
                if j == i:
                    continue
                if resikeys[j] in to_add:
                    continue
                resi2 = self.resis[resikeys[j]]
                resname2 = resikeys[j].split('|')[0]
                md2 = massdefs[resname2]
                if md1 is md2 and md1.redefine_resis is not None:
                    from_atom = md1.connect_init[0]
                    to_atom = md1.connect_init[1]
                    if from_atom in resi1 and to_atom in resi2 and \
                                    np.linalg.norm(resi1[from_atom].xyz - resi2[to_atom].xyz) <= 2:
                        rr = md1.redefine_resis
                        resikey = None
                        from_resi = None
                        if rr[0] == 1:
                            resikey = resikeys[j]
                            from_resi = resi1
                        else:
                            resikey = resikeys[i]
                            from_resi = resi2
                        to_add[resikey] = []
                        for atom in rr[1]:
                            if not atom in from_resi:
                                raise ValueError("Atom {0} not in residue {1}".format(atom, resikeys[i]))
                            to_add[resikey].append(from_resi[atom])
                            del from_resi[atom]
            # check if transfer atoms present and untransferred
            if md1.redefine_resis is not None:
                for atom in md1.redefine_resis[1]:
                    if atom in resi1:
                        del resi1[atom]
        # put transferred atoms back inside respective residues
        for resi in to_add:
            for atom in to_add[resi]:
                self.resis[resi][atom.name] = atom

    def compute_n_chains(self):
        """ Uses the connectivity matrix to infer the number of chains present
            in the system.
        """
        visited, chains = self.visit_dfs()
        return len(chains)

    def compute_seqs_dfs(self):
        """ Uses the connectivity matrix to get the sequence(s) of masses in the molecule. Chains must be contiguous.
        """
        visited, chains = self.visit_dfs()
        if self.validate_visited(visited):
            seqs = [[]]
            seqs_i = 0
            for i in range(len(self.masses)):
                if i > 0 and visited[i-1] != visited[i]:
                    seqs.append([])
                    seqs_i += 1
                seqs[seqs_i].append(self.masses[i][4].split('|')[0])
            return seqs
        else:
            raise ValueError("Sequences not sequential in file " + self.pdb_file)

    def validate_visited(self, visited):
        """ Returns True if chains are contiguous, False otherwise.
        """
        seen = set()
        last_v = None
        for v in visited:
            if v != last_v:
                if v in seen:
                    return False
                seen.add(v)
            last_v = v
        return True

    def visit_dfs(self):
        """ Creates a list of 'colors' for all the masses in the system using DFS. Same 'color' means the masses are
            part of the same chain. Returns the list, and a set of 'colors' (cardinality = number of chains).
        """
        n = len(self.connect)
        visited = [0] * n
        color = 1
        for i in range(n):
            self.dfs(self.connect, visited, color, i)
            color += 1
        chains = set(visited)
        if 0 in chains:
            raise ValueError("Some nodes were not reached by compute_n_chains")
        return visited, chains

    def dfs(self, connect, visited, color, i):
        """ Simple depth-first search algorithm, used for determining number of
            separate chains.
        """
        for e in self.adjacent_edges(connect, i):
            if visited[e] == 0 or visited[e] > color:
                visited[e] = color
                self.dfs(connect, visited, color, e)

    def adjacent_edges(self, connect, i):
        """ Returns all edges/vertices adjacent to vertex i in connectivity
            matrix connect (undirected graph).
        """
        ae = []
        for j in range(len(connect)):
            if connect[i][j] != 0:
                ae.append(j)
        return ae

    def find_bends(self, connect):
        """ Finds all (i, j, k) triplets such that there is a covalent i-j-k
            connection.
        """
        bends = set()
        n = len(connect)
        for i in range(n):
            for j in self.get_connects(connect, i):
                for k in self.get_connects(connect, j):
                    bend = self.make_bend_tuple(i, j, k)
                    if bend is not None:
                        bends.add(bend)
        return self.trim_bends(bends)

    def trim_bends(self, bends):
        """ Gets rid of duplicate bending angles.
        """
        trimmed = set()
        for b in bends:
            if b[0] > b[2]:
                b = (b[2], b[1], b[0])
            trimmed.add(b)
        return trimmed

    def make_bend_tuple(self, i, j, k):
        s = {i, j, k}
        if len(s) != 3:
            return None
        return (i, j, k)

    def find_torsions(self, connect):
        torsions = set()
        n = len(connect)
        for i in range(n):
            for j in self.get_connects(connect, i):
                for k in self.get_connects(connect, j):
                    for l in self.get_connects(connect, k):
                        torsion = self.make_torsion_tuple(i, j, k, l)
                        if torsion is not None:
                            torsions.add(torsion)
        return self.trim_torsions(torsions)

    def trim_torsions(self, torsions):
        trimmed = set()
        for t in torsions:
            if t[0] > t[3]:
                t = (t[3], t[2], t[1], t[0])
            trimmed.add(t)
        return trimmed

    def make_torsion_tuple(self, i, j, k, l):
        s = {i, j, k, l}
        self.print_verbose(s)
        if len(s) != 4:
            return None
        return i, j, k, l

    def get_connects(self, connect, i):
        connects = []
        for (j, c) in enumerate(connect[i]):
            if c != 0:
                connects.append(j)
        return connects

    def find_interacting_pairs(self, torsions, bends, connect):
        ips = set()
        for t in list(torsions) + list(bends):
            for i in range(len(t)):
                for j in range(i + 1, len(t)):
                    if t[i] < t[j]:
                        ips.add((t[i], t[j]))
                    else:
                        ips.add((t[j], t[i]))
        for i in range(len(connect)):
            for j in range(i + 1, len(connect)):
                if connect[i][j] != 0:
                    ips.add((i, j))
        self.print_verbose(ips)
        return ips

    def is_disconnected(self, i, j):
        if i < j:
            return not ((i, j) in self.interact_pairs)
        else:
            return not ((j, i) in self.interact_pairs)

    def print_connect(self):
        for row in self.connect:
            print(row)

    def is_equal(self, other):
        """
        Args:
            other: another Macromol instance
        Returns: boolean
        """
        assert isinstance(other, Macromol)
        if len(self.atoms) != len(other.atoms):
            return False
        for i in range(len(self.atoms)):
            if not self.atoms[i].is_equal(other.atoms[i]):
                return False
        return True

    @staticmethod
    def pretty_print_interactions(interlist, masses):
        s = ""
        for i in interlist:
            for x in i:
                s = s + masses[x][4] + ", "
            s = s[:-2] + "\n"
        print(s)

    def print_verbose(self, string):
        if self.verbose:
            print(string)

    def rotate(self, rotation_mat):
        """ Performs the rotation described by the rotation matrix on all atoms. Original centroid is preserved.
        """
        centroid = self.get_centroid()
        self.translate_xyz(-centroid)
        all_atom_coords = self.get_allatom_coords()
        rotated_coords = np.dot(all_atom_coords, np.transpose(rotation_mat))
        for atom, vec in zip(self.atoms, rotated_coords):
            atom.set_xyz(vec)
        self.translate_xyz(centroid)

    def get_allatom_coords(self):
        """ Returns Nx3 matrix containing coordinates of atoms, in order of increasing atom number.
        """
        coords = np.zeros((len(self.atoms), 3))
        for i, atom in enumerate(self.atoms):
            coords[i] = atom.xyz
        return coords

    def get_n_atoms(self):
        """ Returns the number of atoms in the macromolecule(s).
        """
        return len(self.atoms)

    def translate_3n_matrix(self, vector):
        """ Translation with full vector represented as Nx3 matrix.
            Can (and most probably will) mess up internal structure.
        """
        masses = self.masses
        resis = self.resis
        assert len(masses) == len(vector)
        for i, m in enumerate(masses):
            vec = vector[i]
            assert len(vec) == 3
            atoms = resis[m[4]]
            for a in atoms:
                atom = atoms[a]
                atom.translate(vector[i])

    def get_3n_vector(self):
        """ Returns the coordinates of the masses as 3n-dimensional (x,y,z, x,y,z, ...) vector.
        """
        vec = np.zeros((3*len(self.masscoords)))
        for i in range(len(self.masscoords)):
            vec[3*i], vec[3*i + 1], vec[3*i + 2] = self.masscoords[i]
        return vec

    def get_filtered_xyz(self, filter):
        """ Returns the xyz coordinates of the masses for which the center's name is in the filter set.
        """
        if filter is None:
            return self.get_xyz()
        return np.array([[a[0], a[1], a[2]] for a in self.masses if a[4].split('|')[0].split('.')[-1] in filter])

    def get_xyz(self):
        """ Returns all xyz coordinates of the masses.
        """
        return np.array([[a[0], a[1], a[2]] for a in self.masses])

    def get_filtered_3n_vector(self, filter):
        """ Same as get_filtered_xyz but returns 3n-dimensional vector.
        """
        coords = self.get_filtered_xyz(filter)
        return np.reshape(coords, (3*len(coords)))

    def get_filtered_n(self, filter):
        return len(self.get_filtered_xyz(filter))

    def translate_3n_vector(self, vector):
        """ Translation with full vector (x,y,z, x,y,z, ...).
            Can (and most probably will) mess up internal structure.
        """
        masses = self.masses
        resis = self.resis
        assert len(masses) * 3 == len(vector)
        for i, m in enumerate(masses):
            atoms = resis[m[4]]
            j = 3 * i
            for a in atoms:
                atom = atoms[a]
                atom.translate([vector[j], vector[j + 1], vector[j + 2]])

    def set_bfactors(self, vector):
        masses = self.masses
        resis = self.resis
        assert len(vector) == len(masses)
        for i, m in enumerate(masses):
            atoms = resis[m[4]]
            for a in atoms:
                atom = atoms[a]
                atom.bfact = vector[i]

    def translate_xyz(self, vector):
        """ Translation with (x, y, z) vector. Preserves internal structure.
        """
        for atom in self.atoms:
            atom.translate(vector)

    def get_centroid(self):
        sum = np.array([0, 0, 0])
        for i in range(len(self.masses)):
            sum = np.add(sum, np.array(self.masses[i][0:3]))
        return sum / len(self.masses)

    def clear_alt_tags(self):
        for a in self.atoms:
            a.alt = ' '

    def write_to_file(self, filename):
        with open(filename, "w") as fh:
            self.write_to_filehandle(fh)

    def write_to_filehandle(self, fh):
        s = self.get_pdb_file_as_string()
        fh.write(s)

    def get_pdb_file_as_string(self):
        # TODO : right now in the case of RNA, not all original atoms are part of resis. Find a workaround.
        s = ""
        resis = self.resis
        for resi in resis:
            atoms = sort_atoms(resis[resi])
            for atom in atoms:
                s += atom.toString() + "\n"
        return s + "END\n"

    def get_surface_dict(self):
        if self.alt_flag:
            self.clear_alt_tags()
        tmp_pdb_file = self.pdb_file + ".tmp"
        with open(tmp_pdb_file, "w") as f:
            self.write_to_filehandle(f)
        sd = vcontacts_wrapper.get_surface_dict(tmp_pdb_file, self.get_n_atoms())
        os.remove(tmp_pdb_file)
        return sd


def sort_atoms(resi):
    adict = dict()
    for a in resi:
        atom = resi[a]
        num = atom.num
        adict[num] = atom
    atoms = []
    for k in sorted(adict.keys()):
        atoms.append(adict[k])
    return atoms
