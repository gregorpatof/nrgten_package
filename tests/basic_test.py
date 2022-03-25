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
        expected_first10_eigvals = [396.2703964932724, 720.9888965951737, 1055.3986945427107, 1701.7899686828996,
                                    2317.884484341096, 3168.7662656736516, 3369.2057216674675, 3698.750190128647,
                                    4378.2118961963315, 5654.467645486391]
        # print(",".join([str(x) for x in self.medium.eigvals[6:16]]))
        for i, v in enumerate(self.medium.eigvals[6:16]):
            self.assertAlmostEqual(v, expected_first10_eigvals[i])

    def test_overlap(self):
        tolerance_big = 0.05 # for some occult reason (probably small numeric errors), overlaps vary from
                             # execution to execution (even on the same system with exact same configuration)
        tolerance_medium = 0.1
        tolerance_small = 0.2

        # expected overlaps
        o_to_c_expected = [0.4786517688565864, 0.22981503941240808, 0.138099902585144, 0.40367329974475447,
                           0.43502400637350247, 0.3160010739095798, 0.051135059599399436, 0.16339838292167744,
                           0.11572301263402335, 0.0399795624712545]

        c_to_o_expected = [0.25962968038520784, 0.44073986911497837, 0.2659698708972389, 0.17331160459258538,
                           0.4900379124944241, 0.2805931696477705, 0.27099656834152835, 0.03215663280918418,
                           0.15974513867174364, 0.003434290600090167]

        # computed overlaps
        o_to_c = get_overlaps(self.cs_open, self.cs_closed, 10)
        c_to_o = get_overlaps(self.cs_closed, self.cs_open, 10)

        # print("[" + ", ".join([str(x) for x in o_to_c]) + "]")
        # print("[" + ", ".join([str(x) for x in c_to_o]) + "]")

        for expected, observed in zip([o_to_c_expected, c_to_o_expected], [o_to_c, c_to_o]):
            for i in range(10):
                self.assertAlmostEqual(observed[i], expected[i])


    def test_bfactors(self):
        tolerance = 0.02 # b-factors also vary a little, but less than overlaps

        expected = [18.076306040880052, 9.937356765617809, 5.731148266626697, 4.528339386531281, 3.8690932109178395,
                    3.5420565116449008, 4.618542914482005, 4.7817512580372155, 3.987936172011557, 3.7408734563598665,
                    4.958843512936163, 5.506093977115766, 4.7946208676183115, 4.381345521718181, 6.4771487588545416]

        computed = self.cs_open.compute_bfactors()[:15]
        # print("[" + ", ".join([str(x) for x in computed]) + "]")
        for i in range(15):
            self.assertAlmostEqual(computed[i], expected[i])
            # diff = abs(computed[i] - expected[i])
            # self.assertLess(diff/expected[i], tolerance)


if __name__ == "__main__":
    unittest.main()
