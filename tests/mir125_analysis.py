from nrgten.encom import ENCoM
import numpy as np
import glob
import os
import sys


def compute_all_sigs_all_entro(filenames, sig_folder, one_mass=False):
    for filename in filenames:
        name = filename.split('/')[-1].split('.')[0]
        sig_filename = '/home/shulgin/school/PhD/a21/projet/mir125/{}/{}.sig'.format(sig_folder, name)
        entro_filename = '/home/shulgin/school/PhD/a21/projet/mir125/{}/{}.entro'.format(sig_folder, name)
        if os.path.isfile(sig_filename) and os.path.isfile(entro_filename):
            continue
        enc = ENCoM(filename, one_mass=one_mass)
        enc.write_dynamical_signature(sig_filename)
        entro = enc.compute_vib_entropy()
        with open(entro_filename, 'w') as f:
            f.write("{}\n".format(entro))
        print(name)

def test_bfactors_entro():
    wt = ENCoM("/home/shulgin/school/PhD/a21/projet/mir125/mir125_WT_no_H.pdb")
    entro_matrix = []
    for beta in [0.1, 1, 10]:
        entro_matrix.append(wt.compute_entro_props(beta))
    entro_matrix.append(wt.compute_entro_props(1, use_rigid_rotor=True))
    with open("/home/shulgin/school/PhD/a21/projet/mir125/entro_props.df", "w") as f:
        # header_str = " ".join(["mode_{}".format(i+1) for i in range(len(entro_matrix[0]))])
        f.write("beta rigid_rotor mode_n value prop\n")
        for row, beta in zip(entro_matrix, [0.1, 1, 10, None]):
            out_strs = []
            if beta is None:
                out_str = "-1 TRUE"
            else:
                out_str = "{} FALSE".format(beta)
            total = np.sum(row)
            for i, value in enumerate(row):
                out_strs.append("{} {} {} {}".format(out_str, i+7, value, value/total))
            # f.write("{} {}\n".format(out_str, " ".join([str(x) for x in row])))
            f.write("\n".join(out_strs))
            f.write("\n")

def test_bfactors_detailed_entro():
    wt = ENCoM("/home/shulgin/school/PhD/a21/projet/mir125/mir125_WT_no_H.pdb")
    entro_data = []
    for beta in [0.1, 1, 10]:
        entro_data.append(wt.compute_entro_props_detailed(beta))
    entro_data.append(wt.compute_entro_props_detailed(1, use_rigid_rotor=True))
    with open("/home/shulgin/school/PhD/a21/projet/mir125/entro_props_detailed.df", "w") as f:
        # header_str = " ".join(["mode_{}".format(i+1) for i in range(len(entro_matrix[0]))])
        f.write("beta rigid_rotor mass_n mode_n value prop\n")
        for matrix, beta in zip(entro_data, [0.1, 1, 10, None]):
            for j, row in enumerate(matrix):
                out_strs = []
                if beta is None:
                    out_str = "-1 TRUE {}".format(j+1)
                else:
                    out_str = "{} FALSE {}".format(beta, j+1)
                total = np.sum(row)
                for i, value in enumerate(row):
                    out_strs.append("{} {} {} {}".format(out_str, i+7, value, value/total))
                # f.write("{} {}\n".format(out_str, " ".join([str(x) for x in row])))
                f.write("\n".join(out_strs))
                f.write("\n")





if __name__ == "__main__":
    test_bfactors_detailed_entro()
    raise ValueError()
    assert len(sys.argv) == 3
    sig_folder = sys.argv[1]
    if sig_folder[-1] == '/':
        sig_folder = sig_folder[:-1]
    one_mass = False
    if sys.argv[2] == "one_mass":
        one_mass = True
    mir125_mutants = glob.glob('/home/shulgin/school/PhD/a21/projet/mir125/mutants_mfe/mir125*.pdb')
    print(len(mir125_mutants))
    compute_all_sigs_all_entro(mir125_mutants, sig_folder, one_mass=one_mass)
    # compute_all_sigs_all_entro(mir125_mutants)
    # compute_all_sigs_all_entro_1n(['/home/shulgin/school/PhD/a21/projet/mir125/mutants/mir125_WT.pdb'])



