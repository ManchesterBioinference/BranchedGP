import numpy as np


# Objective function
class objectiveBAndK:
    def __init__(self, mV, fOptimizeLocal):
        '''
        Pass in initial value for branch point, VB models, Y data, pseudotime t
        and OMGP model
        '''
        self.mV = mV
        # Store initial values
        self.likvar = self.mV.likelihood.variance.value  # just fixed nugget
        # Global optimiser: branching location, kernel len, kernel variance - fixed = True
        if(fOptimizeLocal):
            assert self.mV.kern.branchkernelparam.kern.variance.fixed is True
        # Local optimiser: Phi is reset  in UpdateBranchingModel
        assert mV.likelihood.variance.fixed is False
        assert self.mV.logPhi.fixed is False
        self.lenInitial = self.mV.kern.branchkernelparam.kern.lengthscales.value
        self.fOptimizeLocal = fOptimizeLocal

    def f(self, theta):
        # theta is nxp array, return nx1
        n = theta.shape[0]
        VBboundarray = np.ones((n, 1))
        if(self.fOptimizeLocal):
            # BO length scale
            assert theta.shape[1] == 3
        else:
            assert theta.shape[1] == 4  # just branch point, kern var, len, lik var

        for i in range(n):
            # Set initial values for local parameters so each run starts the same
            self.mV.likelihood.variance = self.likvar
            # Set BO'd parameters
            self.mV.UpdateBranchingPoint(np.ones((1, 1)) * theta[i, 0])  # this update phi to initial
            self.mV.kern.branchkernelparam.kern.variance = theta[i, 1]
            self.mV.kern.branchkernelparam.kern.lengthscales = theta[i, 2]
            if not self.fOptimizeLocal:
                self.mV.likelihood.variance = theta[i, 3]
            else:
                self.mV.optimize()
            VBboundarray[i] = self.mV.objectiveFun()  # we wish to minimize!
            if(self.fOptimizeLocal):
                print('objectiveB B=%.2f kervar=%.2f len=%.2f VB=%.3f' %
                      (theta[i, 0], theta[i, 1], theta[i, 2], VBboundarray[i]))
            else:
                print('objectiveB B=%.2f kervar=%.2f len=%.2f likvar=%.2f VB=%.3f' %
                      (theta[i, 0], theta[i, 1], theta[i, 2], theta[i, 3], VBboundarray[i]))

        return VBboundarray
