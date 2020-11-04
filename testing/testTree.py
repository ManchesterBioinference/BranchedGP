# Generic libraries
import unittest

import numpy as np

# Branching files
from BranchedGP import BranchingTree as bt


class TestSparseVariational(unittest.TestCase):
    def test(self):
        tree = bt.BinaryBranchingTree(0, 1, fDebug=True)
        trueB = 0.2
        tree.add(None, 1, trueB)
        tree.add(1, 2, trueB + 0.1)
        tree.add(2, 3, trueB + 0.1 + 0.2)
        tree.add(1, 4, trueB + 0.1 + 0.3)
        assert tree.getRoot().idB == 1
        tree.getRoot().val == trueB
        tree.printTree()
        assert tree.findLCAPath(3, 4)[0] == 1
        fm, fmb = tree.GetFunctionBranchTensor()
        assert np.all(fm.shape == (9, 9, 4))
        assert np.all(fmb.shape == (9, 9, 4))


if __name__ == "__main__":
    unittest.main()
