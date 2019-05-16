import unittest
import numpy as np
import numpy.testing as npt

import compression


class CompressionTest(unittest.TestCase):
    """Test cases."""
    def setUp(self):
        self.d = 0.4
        self.w = np.array([[0.9, 2.1],
                           [1.9, 0.5],
                           [0.3, 1.1]])

    def test_prune(self):
        gold = np.array([[0.9, 2.1],
                         [1.9, 0.5],
                         [0., 1.1]])
        ans = compression.prune(self.w, self.d)
        self.assertTrue(np.all(gold == ans))

    def test_meld(self):
        gold = np.array([[1., 2.],
                         [2., 0.4],
                         [0.4, 1.]])
        ans = compression.reshape_meldprune(self.w)
        self.assertTrue(np.all(gold == ans))

    def test_meldprune(self):
        gold = np.array([[1., 2.],
                         [2., 0.4],
                         [0.4, 1.]])
        ans = compression.reshape_meldprune(self.w, self.d)
        self.assertTrue(np.all(gold == ans))

    def test_norm_prune(self):
        ans = compression.reshape_norm_prune(self.w, self.d)
        gold = np.array([[0.9, 2.1],
                         [1.9, 0.],
                         [0., 1.1]])
        # npt.assert_equal(gold, ans)
        self.assertTrue(np.all(gold == ans))

    def test_norm_meld(self):
        gold = np.array([[0.9320942, 2.13381307],
                         [1.87036166, 0.42112838],
                         [0.36913373, 1.06338514]])
        ans = compression.reshape_norm_meldprune(self.w)
        npt.assert_almost_equal(ans, gold)

    def test_norm_meldprune(self):
        gold = np.array([[0.9320942, 2.13381307],
                         [1.87036166, 0.],
                         [0., 1.06338514]])
        ans = compression.reshape_norm_meldprune(self.w, self.d)
        npt.assert_almost_equal(ans, gold)


if __name__ == '__main__':
    unittest.main()
