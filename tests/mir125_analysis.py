from nrgten.encom import ENCoM
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





if __name__ == "__main__":
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



