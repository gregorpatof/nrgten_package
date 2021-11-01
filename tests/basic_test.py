import unittest
import os

from nrgten.encom import ENCoM
from nrgten.metrics import get_overlaps

class MainTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        super(MainTest, self).setUpClass()
        self.dirpath = os.path.dirname(os.path.abspath(__file__))
        self.medium = ENCoM(str(os.path.join(self.dirpath, "test_medium.pdb"))) # small 2-segment protein

        # open and closed states of citrate synthase, cleaned to have exact same number of masses
        self.cs_open = ENCoM(str(os.path.join(self.dirpath, "open_clean.pdb")))
        self.cs_closed = ENCoM(str(os.path.join(self.dirpath, "closed_clean.pdb")))

    def test_entropy(self):
        self.assertAlmostEqual(self.medium.compute_vib_entropy(), 504.8468148072965)

    def test_eigenvalues(self):
        """Ensuring ENCoM consistency

        The passing of this test ensures that the ENCoM model implemented here is the same as in the 2014 Frappier &
        Najmanovich paper (doi: ﻿10.1371/journal.pcbi.1003569)
        """
        expected_first10_eigvals = [ 396.27039649, 720.9888966, 1055.39869454, 1701.78996868, 2317.88448434,
                                     3168.76626567, 3369.20572167, 3698.75019013, 4378.2118962, 5654.46764549]
        for i, v in enumerate(self.medium.eigvals[6:16]):
            self.assertAlmostEqual(v, expected_first10_eigvals[i])

    def test_overlap(self):
        tolerance_big = 0.05 # for some occult reason (probably small numeric errors), overlaps vary from
                             # execution to execution (even on the same system with exact same configuration)
        tolerance_medium = 0.1
        tolerance_small = 0.2

        # expected overlaps
        o_to_c_expected = [0.4786517688583321, 0.22981503940912792, 0.1380999025847963, 0.40367329974466676,
                           0.4350240063735683, 0.3160010739094261, 0.0511350595996429, 0.1633983829218414,
                           0.11572301263394776, 0.039979562471208686]

        c_to_o_expected = [0.25962968038502166, 0.4407398691153137, 0.2659698708967415, 0.1733116045927785,
                           0.4900379124945718, 0.28059316964743347, 0.2709965683414201, 0.03215663281065148,
                           0.15974513867178708, 0.0034342906003420864]


        # computed overlaps
        o_to_c = get_overlaps(self.cs_open, self.cs_closed, 10)
        c_to_o = get_overlaps(self.cs_closed, self.cs_open, 10)

        for expected, observed in zip([o_to_c_expected, c_to_o_expected], [o_to_c, c_to_o]):
            for i in range(10):
                diff = abs(observed[i] - expected[i])
                if expected[i] > 0.3:
                    self.assertLess(diff / expected[i], tolerance_big)
                elif expected[i] > 0.1:
                    self.assertLess(diff / expected[i], tolerance_medium)
                else:
                    self.assertLess(diff / expected[i], tolerance_small)

    def test_bfactors(self):
        tolerance = 0.02 # b-factors also vary a little, but less than overlaps

        expected = [18.076306040878674, 9.937356765617086, 5.731148266626317, 4.5283393865306385, 3.8690932109169642,
                    3.5420565116442946, 4.618542914480897, 4.781751258035752, 3.9879361720104503, 3.740873456359082,
                    4.958843512934868, 5.506093977114312, 4.794620867617188, 4.381345521717479, 6.477148758853096]

        computed = self.cs_open.compute_bfactors()[:15]
        for i in range(15):
            diff = abs(computed[i] - expected[i])
            self.assertLess(diff/expected[i], tolerance)


if __name__ == "__main__":
    unittest.main()
