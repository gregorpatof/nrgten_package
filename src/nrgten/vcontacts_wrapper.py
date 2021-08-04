from pathlib import Path, PurePath
from multiprocessing import Pool
import ctypes
import os
import platform
import subprocess


def get_surface_dict(pdb_str, n_atoms):
    """ Returns a dict of dicts based on atom numbers (input file must have
        unique atom numbers), containing surface area in contact between
        pairs of atoms.
    """
    p = Pool(1)
    results = p.starmap(get_surface_dict_private, zip([pdb_str],[n_atoms]))
    return results[0]
    # return get_surface_dict_private(pdb_str, n_atoms)


def get_surface_dict_private(pdb_str, n_atoms):
    sd = dict()
    atom_nums, atom_names, resi_nums, resi_names, chains, areas, dists = run_vcon(pdb_str, n_atoms)
    n = len(atom_nums)
    i = 0
    last_anum = None
    while i < n: # starting atom is at every multiple of 100, followed by up to 99 contacts
        if atom_nums[i] == -1000000:
            i += 100 - i % 100 # skip to the next starting atom if no more contacts, -1000000 is placeholder
            continue
        anum = atom_nums[i]
        atype = atom_names[5*i:5*i+5].replace(b'\x00', b' ').decode('utf-8').strip()
        resinum = resi_nums[i]
        resiname = resi_names[4*i:4*i+4].replace(b'\x00', b' ').decode('utf-8').strip()
        chain = chains[i]
        key = "{}|{}|{}".format(resiname, resinum, chain)
        if i % 100 == 0:
            if anum == 0: # TODO: find out what wizardry is happening inside vcon for some atoms to have no contacts at all (and remove this unelegant solution)
                i += 1
                continue
            sas = dists[i] # from atoms have -1 in the areas array to distinguish them. SAS is thus in dists.
            last_anum = anum
            if anum in sd:
                raise ValueError("Atom number {0} encountered twice by Vcontacts".format(anum))
            sd[anum] = dict()
        else:
            sd[last_anum][anum] = areas[i]
        i += 1
    return sd

def run_vcon(pdb_str, n_atoms, tried_compiling_once=False):
    """ Runs Vcontacts using ctypes. Returns raw ctypes C arrays of the 7 clolumns outputed by Vcontacts.
        The processing of these happens in get_surface_dict.
    """
    data = ctypes.create_string_buffer(bytes(pdb_str, 'utf-8'), len(pdb_str))
    libpath = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vcontacts'))
    libnames = []
    libnames = libnames + [x for x in libpath.glob("vconlib*.so")]
    libnames = libnames + [x for x in libpath.glob("vconlib*.dll")]
    if len(libnames) > 1:
        raise ValueError("This is libnames: {0}\nand this is libpath: {1}".format(libnames, libpath))
    elif (len(libnames) == 0 or source_was_updated(libnames[0])) and not tried_compiling_once:
        try_to_compile_vcon(libpath)
        return run_vcon(pdb_str, n_atoms, tried_compiling_once=True)

    assert len(libnames) == 1
    libname = libnames[0]
    # libname = "vcontacts/vcon_lib_python.so"
    vcon_lib = ctypes.CDLL(str(libname))

    length = n_atoms * 100

    atom_nums = (ctypes.c_int * length)(-1000000) # -1000000 is a placeholder, never happens as atom number
    atom_names = ctypes.create_string_buffer(bytes("", 'utf8'), length * 5)
    resi_nums = (ctypes.c_int * length)(0)
    resi_names = ctypes.create_string_buffer(bytes("", 'utf8'), length * 4)
    chains = ctypes.create_string_buffer(bytes("", 'utf8'), length)
    areas = (ctypes.c_float * length)(0)
    dists = (ctypes.c_float * length)(0)

    ret = vcon_lib.run_from_python(data, atom_nums, atom_names, resi_nums, resi_names, chains, areas, dists)
    if ret != 0:
        raise ValueError("Error running Vcontacts, run_from_python produced return value {0}".format(ret))
    return atom_nums, atom_names, resi_nums, resi_names, chains, areas, dists


def source_was_updated(libname):
    """Checks if the Vcontacts source is more recent than the library.
    """
    libpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vcontacts')
    lib_time = os.stat(libname).st_mtime
    source_time = None
    if str(libname).endswith(".dll"):
        source_time = os.stat(os.path.join(libpath, 'Vcontacts-v1.2_for_python_windows.c')).st_mtime
    else:
        source_time = os.stat(os.path.join(libpath, 'Vcontacts-v1.2_for_python.c')).st_mtime
    return lib_time < source_time



def try_to_compile_vcon(libpath):
    """ Workaround for compiling the Vcontacts library.

        The reason for using this instead of Extension directly in the setup.py script is that Extension works on
        Unix-like systems but is looking for some type of Cython/Python.h bindings when compiled on Windows (to enable
        importing the extension as a Python module), whereas the Vcontacts library is native C that is used with ctypes
        and does not need to be importable. This function is a simple hack that compiles then library upon the first
        execution of Vcontacts.
    """
    libpath = PurePath(libpath)
    op_system = platform.system()
    args = []
    args2 = None
    if op_system == "Windows":
        args = ['cl.exe', '/LD', str(libpath.joinpath('Vcontacts-v1.2_for_python_windows.c'))]
        args2 = ['cp', 'Vcontacts-v1.2_for_python_windows.dll', str(libpath.joinpath('vconlib.dll'))]
    elif op_system == "Darwin" or op_system == "Linux":
        args = ['gcc', '-shared', '-o', str(libpath.joinpath('vconlib.so')),
                '-fPIC', str(libpath.joinpath('Vcontacts-v1.2_for_python.c'))]
    else:
        raise ValueError("Unrecognized operating system: {0}".format(op_system))
    subprocess.call(args)
    if args2 is not None:
        subprocess.call(args2)






