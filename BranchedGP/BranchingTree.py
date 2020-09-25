import numpy as np


def checkIndices(indices, XForKernel, Xtrue):
    '''# Check indices produces XForKernel[] -> same as X (N size) and in addition all rows are taken from X'''
    assert Xtrue.shape[1] == XForKernel.shape[1]  # same number of columns
    Xt = Xtrue[:, 0]  # first column
    Xr = np.zeros(Xt.shape)
    for i, ind in enumerate(indices):
        Xr[i] = XForKernel[ind[0], 0]  # take the first entry
    assert np.allclose(Xr, Xt)

# Utility functions useful to encode/decode function identifiers

def GenFunctionName(branchPoint, branchNumber=0):
    assert branchPoint > 0  # Branch id should be non-zero positive
    assert branchNumber == 0 or branchNumber == 1
    return (branchPoint << 1) + branchNumber  # store in least significant bit function id (0/1)


def GetBranchPtFromFunctionName(functionNumber):
    if(functionNumber <= 0):
        NameError('Function id should be non-zero positive. Got ' + str(functionNumber))
    branchId = (functionNumber >> 1)
    functionId = (functionNumber & 1)  # function id is 0 or 1
    return (branchId, functionId)

#
# Code the implements binary branching tree
#
# Code based on http://www.geeksforgeeks.org/lowest-common-ancestor-binary-tree-set-1/
# and for tree from http://stackoverflow.com/questions/2598437/how-to-implement-a-binary-tree-in-python

# O(n) solution to find LCS of two given values n1 and n2

# A branching tree node


class BranchingPt:
    # Constructor to create a new binary node

    def __init__(self, idB, val):
        self.idB = idB        # id
        self.val = val      # value in x space
        self.left = None
        self.right = None

# A binary tree


class BinaryBranchingTree:

    def __init__(self, lbX, ubX, fDebug=False):
        self.root = None
        self.dictBranch = {}  # dictionary of branching points and their values for quick access
        self.dictFunction = {}
        self.lbX = lbX  # lower bound of X
        self.ubX = ubX  # upper bound of X
        self.fDebug = fDebug  # will print out messages

    def printTree(self):
        if(self.root is not None):
            self._printTree(self.root, 1)

    def _printTree(self, node, ntabs):
        if(node is not None):
            ntabs += 1
            self._printTree(node.left, ntabs)
            print('---------' * ntabs + str(node.idB) + '(' + str(node.val) + ')' +
                  '[' + str(self.dictFunction[node.idB][0]) + ',' + str(self.dictFunction[node.idB][1]) + ']')
            self._printTree(node.right, ntabs)

    def getRoot(self):
        return self.root

    def find(self, idB):
        if(self.root is not None):
            return self._find(idB, self.root)

    def _find(self, idB, node):
        if(idB == node.idB):
            return node
        else:
            n = None
            if(node.left is not None):
                n = self._find(idB, node.left)
                if(n is None and node.right is not None):
                    n = self._find(idB, node.right)
            return n

    # Finds the path from root node to given root of the tree.
    # Stores the path in a list path[], returns true if path
    # exists otherwise false
    def findPath(self, path, idB):
        if self.root is None:
            return False

        return self._findPath(self.root, path, idB)

    def _findPath(self, node, path, idB):
        # Store this node is path vector. The node will be
        # removed if not in path from root to idB
        path.append(node.idB)

        # See if the k is same as root's key
        if node.idB == idB:
            return True

        # Check if k is found in left or right sub-tree
        if ((node.left is not None and self._findPath(node.left, path, idB)) or
                (node.right is not None and self._findPath(node.right, path, idB))):
            return True

        # If not present in subtree rooted with root, remove
        # root from path and return False

        path.pop()
        return False

    def _findNode(self, node, idB):
        # See if the k is same as root's key
        if node.idB == idB:
            return node

        # Check if k is found in left or right sub-tree
        n = None
        if (node.left is not None):
            n = self._findNode(node.left, idB)
        if(n is None):
            # try other child
            if(node.right is not None):
                n = self._findNode(node.right, idB)

        return n

    # Returns path to least common ancestor (LCA) if node n1 , n2 are present in the given
    # branching tree
    def findLCAPath(self, idn1, idn2):
        # To store paths to n1 and n2 fromthe root
        path1 = []
        path2 = []

        # Find paths from root to n1 and root to n2.
        # If either n1 or n2 is not present , return -1
        if (not self.findPath(path1, idn1)):
            raise NameError('Could not locate branch point ' + str(idn1))
        if (not self.findPath(path2, idn2)):
            raise NameError('Could not locate branch point ' + str(idn2))

        # Compare the paths to get the first different value
        i = 0
        while(i < len(path1) and i < len(path2)):
            if path1[i] != path2[i]:
                break
            i += 1
        return path1[:i]

    def add(self, idParent, idB, val):
        # Add branching point
        if(val < self.lbX or val > self.ubX):
            raise NameError('Value of node= ' + str(val) +
                            ' outside bounds [' + str(self.lbX) + ',' + str(self.ubX) + '] ')

        if(self.root is None):
            self.root = BranchingPt(idB, val)
            self.dictBranch[idB] = val
            self.dictFunction[idB] = [GenFunctionName(idB, 0), GenFunctionName(idB, 1)]  # left and right is important
        else:
            node = self.find(idParent)
            if(node is None):
                raise NameError('Could not find parent id ' + str(idParent))
            else:
                # found the node - it's branching value better be less than ours
                if(node.val > val):
                    raise NameError('Trying to add node with greater branch value than parent')
                if(node.left is None):
                    node.left = BranchingPt(idB, val)
                    self.dictBranch[idB] = val
                    self.dictFunction[idB] = [GenFunctionName(idB, 0), GenFunctionName(idB, 1)]
                elif(node.right is None):
                    node.right = BranchingPt(idB, val)
                    self.dictBranch[idB] = val
                    self.dictFunction[idB] = [GenFunctionName(idB, 0), GenFunctionName(idB, 1)]
                else:
                    raise NameError('Trying to add node to node with two children')

    def GetBranchValues(self, idBVector=None):
        # Return all branch values with ids in list idBVecotr
        # return type is list
        if(idBVector is None):
            return list(self.dictBranch.values())
        else:
            return [self.dictBranch[idB] for idB in idBVector]

    def GetNumberOfBranchPts(self):
        return len(self.dictBranch)

    def _GetBranchValuesAsArray(self, idBVector=None):
        # Return all branch values with ids in list idBVecotr
        # return type is numpy 2-D array for ease of use
        return np.atleast_2d(np.array(self.GetBranchValues(idBVector)))

    def GetFunctionPath(self, fid):
        listOfFunctions = [1]  # function 1 always there
        (bid, _) = GetBranchPtFromFunctionName(fid)
        pathBranch = []
        self.findPath(pathBranch, bid)  # find path to that branch
        if(self.fDebug):
            print('path branch' + str(pathBranch))
        if self.root is None:
            return listOfFunctions
        if(len(pathBranch) > 0):
            pathBranch.pop(0)  # is root
            self._findFunctionPath(self.root, pathBranch, listOfFunctions)

        listOfFunctions.append(fid)  # always add myself to the end
        return listOfFunctions

    def _findFunctionPath(self, node, pathBranch, functionPath):
        if(len(pathBranch) == 0):
            # print 'End'
            return

        idBSearch = pathBranch.pop(0)
        if(self.fDebug):
            print('Searching ' + str(idBSearch))
        if(node.left.idB == idBSearch):
            functionPath.append(GenFunctionName(node.idB, 0))  # left is 0
            self._findFunctionPath(node.left, pathBranch, functionPath)
        elif(node.right.idB == idBSearch):
            functionPath.append(GenFunctionName(node.idB, 1))  # right is 1
            self._findFunctionPath(node.right, pathBranch, functionPath)
        else:
            NameError('Could not find child node id  ' + str(idBSearch) + ' for myself  ' + str(node.idB))

    def GetFunctionBranchTensor(self):
        # Create M X M X B tensor that maps function values to branch values
        nb = self.GetNumberOfBranchPts()
        m = 2 * nb + 1  # number of functions
        if(self.fDebug):
            print('Number of branch points ' + str(nb) + 'number of functions' + str(m))

        fm = np.zeros((m, m, nb))  # tensor with branching point ids
        fmb = np.zeros((m, m, nb))  # tensor with branching point values
        fm[:] = np.NAN
        fmb[:] = np.NAN

        for fi in range(m):
            fi = fi + 1
            for fj in range(m):
                fj = fj + 1
                branchpath = []

                if(fi == fj):
                    continue

                bid_i, fBin_i = GetBranchPtFromFunctionName(fi)  # branch parent of fi
                bid_j, fBin_j = GetBranchPtFromFunctionName(fj)  # branch parent of fj
                pathi = set(self.GetFunctionPath(fi))
                pathj = set(self.GetFunctionPath(fj))
                if(pathi <= pathj or pathj <= pathi):
                    # function subset - they are on the same path
                    if(pathi <= pathj):
                        bid = bid_j  # fj is further along
                        fBin = fBin_j
                    else:
                        bid = bid_i
                        fBin = fBin_i

                    if(self.fDebug):
                        print('Path to branch point ' + str(bid))
                    assert self.findPath(branchpath, bid) == True
                    # include all subsequent branch points since they are not visible anyway and will make
                    # matrix well-conditioned
                    node = self._findNode(self.root, bid)  # find the node
                    branchpathFollowing = []
                    if(fBin == 0 and node.left is not None):  # 0 is left
                        self._GetAllNestedBranchPts(node.left, bid, branchpathFollowing)
                    elif(fBin == 1 and node.right is not None):  # 1 is right
                        self._GetAllNestedBranchPts(node.right, bid, branchpathFollowing)
                    else:
                        assert fBin == 0 or fBin == 1

                    if(self.fDebug):
                        print('Following node ids ' + str(branchpathFollowing))
                    branchpath += branchpathFollowing
                else:
                    # not subsets - they are on different baths, use LCA
                    branchpath = self.findLCAPath(bid_i, bid_j)
                    if(self.fDebug):
                        print('LCA path of (' + str(fi) + ',' + str(fj) + ') is ' + str(branchpath))
                    assert len(branchpath) > 0

                branchvalues = self._GetBranchValuesAsArray(branchpath)
                if(self.fDebug):
                    print(
                        'Functions ' +
                        str(fi) +
                        ',' +
                        str(fj) +
                        ' crosspoints ' +
                        str(branchpath) +
                        ', values ' +
                        str(branchvalues))
                #fm[fi-1,fj-1,:-(nb - len(branchpath))]=branchpath
                #fmb[fi-1,fj-1,:-(nb - len(branchpath))]=branchvalues
                fm[fi - 1, fj - 1, :len(branchpath)] = branchpath
                fmb[fi - 1, fj - 1, :len(branchpath)] = branchvalues

        # Check tensor
        # branch ids should form a contiguous set of integers starting at 1
        if(self.fDebug):
            print('Tensor fm' + str(fm))
        fmFlat = np.ravel(fm)
        fmFlat = fmFlat[~np.isnan(fmFlat)]
        expectedBranchPoints = 1 + np.array(list(range(nb)))
        expectedBranchPoints = expectedBranchPoints.astype(float)
        assert set(fmFlat) == set(expectedBranchPoints)

        return (fm, fmb)

    def _GetAllNestedBranchPts(self, node, bid, listBranch):
        # Get all nested branch pts
        # assumes call on correct nested partition (see node)
        # call _findNode first
        if(node.idB != bid):  # dont add myself
            listBranch.append(node.idB)
        if(node.left is not None):
            self._GetAllNestedBranchPts(node.left, bid, listBranch)
        if(node.right is not None):
            self._GetAllNestedBranchPts(node.right, bid, listBranch)

    def GetFunctionIndexList(self, Xin, fReturnXtrue=False):
        ''' Function to return index list  and input array X repeated as many time as each possible function '''
        # limited to one dimensional X for now!
        assert Xin.shape[0] == np.size(Xin)
        df = self.GetFunctionDomains()
        indicesBranch = []
        if(np.any(Xin <= df[0, 0])):
            raise NameError('Value passed in at or less than lower bound ' + str(df[0, 0]))
        if(np.any(Xin > df[-1, 1])):
            raise NameError('Value passed in greater than upper bound ' + str(df[-1, 1]))

        Xtrue = np.zeros((Xin.shape[0], 2), dtype=float)
        Xnew = []
        inew = 0
        for ix, x in enumerate(Xin):
            assignTo = (x > df[:, 0]) & (x <= df[:, 1])        # does where equality is matter check tree search?
            Xtrue[ix, 0] = Xin[ix]
            functionList = list(np.flatnonzero(assignTo))
            Xtrue[ix, 1] = np.random.choice(functionList) + 1  # one based counting
            idx = []
            for f in functionList:
                Xnew.append([x, f + 1])  # could have 1 or 0 based function list - does kernel care?
                idx.append(inew)
                inew = inew + 1
            indicesBranch.append(idx)

        Xnewa = np.array(Xnew)

        checkIndices(indicesBranch, Xnewa[:, 0][:, None], Xin[:, None])

        if(fReturnXtrue):
            return (Xnewa, indicesBranch, Xtrue)
        else:
            return (Xnewa, indicesBranch)

    def GetFunctionDomains(self):
        nb = self.GetNumberOfBranchPts()
        m = 2 * nb + 1  # number of functions
        domainF = np.zeros((m, 2))
        domainF[:] = np.NAN

        if(self.root is None):
            return domainF

        # function 1 is always [lb,b1]
        domainF[0, :] = [self.lbX, self.root.val]

        for fi in range(1, m):
            fi = fi + 1  # functions are 1-based
            # Find branch point
            bid, fBinaryId = GetBranchPtFromFunctionName(fi)
            node = self._findNode(self.root, bid)
            assert node is not None
            assert node.idB == bid
            # function branch point - lower bound is branch point value
            lb = node.val
            if(fBinaryId == 0):  # 0 is left child
                if(node.left is None):
                    ub = self.ubX
                else:
                    ub = node.left.val
            elif(fBinaryId == 1):  # 1 is right child
                if(node.right is None):
                    ub = self.ubX
                else:
                    ub = node.right.val
            else:
                assert fBinaryId == 0 or fBinaryId == 1

            domainF[fi - 1, :] = [lb, ub]
        # check no nans left
        dflat = np.ravel(domainF)
        assert np.any(np.isnan(dflat)) == False
        return domainF
