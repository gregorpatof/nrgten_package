from nrgten.encom import ENCoM
from scipy.stats import pearsonr
import numpy as np


if __name__ == "__main__":
    enc = ENCoM('test_medium.pdb')
    loc = enc.compute_local_signature(beta=np.e**-2, use_entropy=True)
    classic = enc.compute_bfactors()
    print(loc)


