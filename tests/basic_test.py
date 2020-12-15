import unittest
import os

from nrgten.encom import ENCoM
from nrgten.metrics import overlap

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
        self.assertAlmostEqual(self.medium.compute_vib_entropy(), 504.8449418766389)

    def test_eigenvalues(self):
        """Ensuring ENCoM consistency

        The passing of this test ensures that the ENCoM model implemented here is the same as in the 2014 Frappier &
        Najmanovich paper (doi: ﻿10.1371/journal.pcbi.1003569)
        """
        expected_first10_eigvals = [395.13213997, 705.84751463, 1051.3377265, 1717.77386711, 2334.71773081,
                             3180.41535126, 3367.29667647, 3732.73424169, 4395.60062344, 5646.29249956]
        for i, v in enumerate(self.medium.eigvals[6:16]):
            self.assertAlmostEqual(v, expected_first10_eigvals[i])

    def test_overlap(self):
        tolerance_big = 0.05 # for some occult reason (probably small numeric errors), overlaps vary from
                             # execution to execution (even on the same system with exact same configuration)
        tolerance_medium = 0.1
        tolerance_small = 0.2

        # expected overlaps
        o_to_c_expected = [0.48219475776961046, 0.22445345494422506, 0.13626771118348027, 0.40620100897625844,
                           0.4269350823318777, 0.3252500318424115, 0.050714322164735454, 0.15852890110829515,
                           0.11657022472204998, 0.04085067373729564]

        c_to_o_expected = [0.2572247945186605, 0.4417530982984568, 0.2676782798408168, 0.16980479771422188,
                           0.4857802255735633, 0.2866088336257986, 0.25885548397562513, 0.08276471832051029,
                           0.16543958904525702, 0.007655844355325302]


        # computed overlaps
        o_to_c = overlap(self.cs_open, self.cs_closed, 10)
        c_to_o = overlap(self.cs_closed, self.cs_open, 10)

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
        tolerance = 0.01 # b-factors also vary a little, but less than overlaps

        expected = [17.915181389368097, 9.807507859742454, 5.6361432619585985, 4.478888248023526, 3.8351524693259593,
                    3.503278027646238, 4.584754887889988, 4.75555282481797, 3.949012476056274, 3.690415817549934,
                    4.913395104757805, 5.45099864816634, 4.7258998263711804, 4.325912357234571, 6.385333388587485]
        computed = self.cs_open.compute_bfactors()[:15]
        for i in range(15):
            diff = abs(computed[i] - expected[i])
            self.assertLess(diff/expected[i], tolerance)


if __name__ == "__main__":
    unittest.main()
