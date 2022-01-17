from nrgten.encom import ENCoM
from nrgten.anm import ANM
import numpy as np
import glob
import os
import sys
import json

def get_cumulative_prop(filename, model=ENCoM):
    try:
        enc = model(filename)
    except ValueError:
        return []
    entro_list = enc.compute_vib_entropy(beta=1, as_list=True)
    entro_list = entro_list / np.sum(entro_list)
    cumulative_prop = np.zeros(len(entro_list))
    for i in range(len(cumulative_prop)):
        cumulative_prop[i] = np.sum(entro_list[:i + 1])
    return cumulative_prop

def get_n_masses(pdb_filename):
    jsonfile = pdb_filename[:-3] + "seqs.json"
    if not os.path.isfile(jsonfile):
        return 0
    with open(jsonfile) as f:
        seqs = json.load(f)
    n = 0
    for seq in seqs:
        n += len(seq)
    return n

def compute_props():
    pdb_files = glob.glob("/home/shulgin/school/PhD/h20/projet/encom_paper/new_db/raw_pdbs/*.pdb")
    print(len(pdb_files))
    n_masses = [get_n_masses(f) for f in pdb_files]
    selected_pdbs = []
    for i in range(len(n_masses)):
        if n_masses[i] < 300 and n_masses[i] > 0:
            if os.path.isfile(pdb_files[i] + ".vcon"):
                selected_pdbs.append(pdb_files[i])
    print(len(selected_pdbs))
    for filename in selected_pdbs:
        props = get_cumulative_prop(filename)
        if len(props) > 0:
            with open("props.txt", "a") as f:
                f.write("{} [{}]\n".format(filename, ",".join([str(x) for x in props])))
                print(props)

def compute_props_anm():
    with open("props.txt") as f:
        lines = f.readlines()
    for line in lines:
        filename = line.split()[0]
        props = get_cumulative_prop(filename, model=get_anm_cut9)
        with open("props_anm_cut9.txt", "a") as f:
            f.write("{} [{}]\n".format(filename, ",".join([str(x) for x in props])))
        props = get_cumulative_prop(filename, model=get_anm_pd4dot5)
        with open("props_anm_pd4point5.txt", "a") as f:
            f.write("{} [{}]\n".format(filename, ",".join([str(x) for x in props])))

def get_anm_cut9(filename):
    return ANM(filename, cut=9)

def get_anm_pd4dot5(filename):
    return ANM(filename, power_dependence=4.5)


def make_props_df(filename, dfname):
    with open(filename) as f:
        lines = f.readlines()
    props_list = []
    for line in lines:
        props = eval(line.split()[-1])
        props_list.append(props)
    lengths = [len(x) for x in props_list]
    with open(dfname, "w") as f:
        f.write("file_i mode_prop entro_prop\n")
        for i, props in enumerate(props_list):
            n = len(props)
            for j, prop in enumerate(props):
                mode_index = j/(n-1)
                f.write("{} {} {}\n".format(i, mode_index, prop))

if __name__ == "__main__":
    # compute_props_anm()
    make_props_df("props_anm_cut9.txt", "props_anm_cut9.df")
    make_props_df("props_anm_pd4point5.txt", "props_anm_pd4point5.df")

    anm_mod = ANM("/home/shulgin/school/PhD/h20/projet/encom_paper/new_db/raw_pdbs/1X9K.pdb", power_dependence=4.5)
    encom_mod = ENCoM("/home/shulgin/school/PhD/h20/projet/encom_paper/new_db/raw_pdbs/1X9K.pdb")
    print((anm_mod.eigvals[-10:] / anm_mod.eigvals[6])**0.5)
    print((encom_mod.eigvals[-10:] / encom_mod.eigvals[6])**0.5)
