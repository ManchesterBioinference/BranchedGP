import numpy as np
import GPyOpt
import VBHelperFunctions


# Objective function
class objectiveBAndK:

    def __init__(self, mV):
        '''
        Pass in initial value for branch point, VB models, Y data, pseudotime t
        and OMGP model
        '''
        self.mV = mV
        # Store initial values
        self.likvar = self.mV.likelihood.variance.value  # just fixed nugget
        # Global optimiser: branching location, kernel len, kernel variance - fixed = True
        assert self.mV.kern.branchkernelparam.kern.lengthscales.fixed is True
        assert self.mV.kern.branchkernelparam.kern.variance.fixed is True
        # Local optimiser: Phi is reset  in UpdateBranchingModel
        assert mV.likelihood.variance.fixed is False
        assert self.mV.logPhi.fixed is False

    def f(self, theta):
        # theta is nxp array, return nx1
        n = theta.shape[0]
        VBboundarray = np.ones((n, 1))
        for i in range(n):
            # Set initial values for local parameters so each run starts the same
            self.mV.likelihood.variance = self.likvar
            # Set BO'd parameters
            self.mV.UpdateBranchingPoint(np.ones((1, 1)) * theta[i, 0])  # this update phi to initial
            self.mV.kern.branchkernelparam.kern.variance = theta[i, 1]
            self.mV.kern.branchkernelparam.kern.lengthscales = theta[i, 2]
            self.mV.optimize()
            VBboundarray[i] = self.mV.objectiveFun()  # we wish to minimize!
            print('objectiveB B=%.2f kervar=%.2f len=%.2f VB=%.3f' %
                  (theta[i, 0], theta[i, 1], theta[i, 2], VBboundarray[i]))
        return VBboundarray
