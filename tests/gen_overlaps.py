from nrgten.encom import ENCoM
from nrgten.metrics import overlap

if __name__ == "__main__":
    open = ENCoM("open_clean.pdb")
    # closed = ENCoM("closed_clean.pdb")
    # print(overlap(open, closed, 10))
    # print(overlap(closed, open, 10))
