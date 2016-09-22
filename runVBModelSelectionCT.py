from matplotlib import pyplot as plt
from matplotlib import cm
import os
import numpy as np
import time
import GPflow
import cPickle as pickle
import assigngp_dense
import VBHelperFunctions
import GPy
import pods
import tensorflow as tf
from GPclust import OMGP
import BranchingTree as bt
import branch_kernParamGPflow as bk
import GPyOpt
import multiprocessing
import pandas as pd
import sys

#import plotly.plotly as py
#import plotly.graph_objs as go


# Objective function
class objectiveBAndK:
    def __init__(self, Binit, mVBmodel, Ygene, pt, m):
        mVBmodel.kern.branchkernelparam.Bv.fixed = False # we wont optimize so this is fine
        mVBmodel.logPhi.fixed = False # allocations not fixed for GPyOpt because we update them for each branch point

        mVBmodel.likelihood.variance.fixed = False # all kernel parameters optimised
        mVBmodel.kern.branchkernelparam.kern.lengthscales.fixed = False
        mVBmodel.kern.branchkernelparam.kern.variance.fixed = False

        # initial branch point
        mVBmodel.kern.branchkernelparam.Bv = Binit
        VBHelperFunctions.InitialisePhiFromOMGP(mVBmodel, phiOMGP=m.phi, b=Binit, Y=Ygene, pt=pt)  
        # Initialise all model parameters using the OMGP model 
        mVBmodel.likelihood.variance = m.variance.values[0]
        mVBmodel.kern.branchkernelparam.kern.lengthscales = np.max(np.array([m.kern[0].lengthscale.values, m.kern[1].lengthscale.values]))
        mVBmodel.kern.branchkernelparam.kern.variance = np.mean(np.array([m.kern[0].variance.values, m.kern[1].variance.values]))
        mVBmodel._compile()

        self.pt = pt
        self.Ygene = Ygene
        self.mVBmodel = mVBmodel
        self.m = m # OMGP model
        
    def f(self, theta):
        # theta is nxp array, return nx1
        n=theta.shape[0]
        VBboundarray = np.ones((n,1))
        for i in range(n):
            self.mVBmodel.kern.branchkernelparam.Bv = theta[i,0]
            VBHelperFunctions.InitialisePhiFromOMGP(self.mVBmodel, phiOMGP=self.m.phi, b=theta[i,0], Y=self.Ygene, pt=self.pt)  
            self.mVBmodel.likelihood.variance = theta[i,1]
            self.mVBmodel.kern.branchkernelparam.kern.lengthscales = theta[i,2]
            self.mVBmodel.kern.branchkernelparam.kern.variance = theta[i,3]
            self.mVBmodel.kern.linear.variance = theta[i,4]
            self.mVBmodel.kern.constant.variance = theta[i,5]
            
            VBboundarray[i] = -self.mVBmodel.compute_log_likelihood() # we wish to minimize!
            print 'objectiveB B=%.0f likvar=%.0f len=%.0f var=%.0f VB=%.0f'%(theta[i,0], theta[i,1], theta[i,2], theta[i,3], VBboundarray[i] )
        return VBboundarray
    
def plotlyCode(strid, mV, labels):
    py.sign_in('alexis.boukouvalas', 'jr2s81lbdw')
    B = mV.kern.branchkernelparam.Bv._array.flatten()
    assert B.size == 1, 'Code limited to one branch point, got ' + str(B.shape)
    pt = mV.t
    l = np.min(pt)
    u = np.max(pt)
    d = 0 # constraint code to be 1D for now
    colorlist=['rgb(93, 164, 214)', 'rgb(255, 144, 14)',
               'rgb(44, 160, 101)', 'rgb(255, 65, 54)']
    for f in range(1, 4):
        if(f == 1):
            ttest = np.linspace(l, B, 100)[:, None]  # root
        else:
            ttest = np.linspace(B, u, 100)[:, None]
        Xtest = np.hstack((ttest, ttest*0+f))
        mu, var = mV.predict_f(Xtest)
        assert np.all(np.isfinite(mu)), 'All elements should be finite but are ' + str(mu)
        assert np.all(np.isfinite(var)), 'All elements should be finite but are ' + str(var)
        traceMean = go.Scatter(x = ttest, y = mu[:, d],
                               name = 'meanf_'+str(f),
                               line = dict(color = ('rgb(205, 12, 24)'), width = 4))
        dataPlot = [traceMean]
        traceStd1 = go.Scatter(x = ttest, y = mu[:, d] - 2*np.sqrt(var.flatten()),
                               name = 'stdf'+str(f),
                               line = dict(
                                    color = colorlist[f-1],
                                    width = 4,
                                    dash = 'dash') # dash options include 'dash', 'dot', and 'dashdot'
                               )
        traceStd2 = go.Scatter(x = ttest, y = mu[:, d] + 2*np.sqrt(var.flatten()),
                               name = 'stdf'+str(f),
                               line = dict(
                                    color = colorlist[f-1],
                                    width = 4,
                                    dash = 'dash') # dash options include 'dash', 'dot', and 'dashdot'
                               )
        dataPlot.append(traceStd1)
        dataPlot.append(traceStd2)
    
    traceBranching = go.Scatter(x = [B, B], y = [mu[:, d].min(), mu[:, d].max()],
                               name = 'BranchingPT',
                               line = dict(color = colorlist[3], width = 4))
    dataPlot.append(traceBranching)
    
    # Plot Phi and labels
    # Phi is size, labels is color and text
    Phi=FlattenPhi(mV)
    gp_num = 1 # can be 0,1,2 - Plot against this
    labelLegend = np.unique(labels)
    for lab in labelLegend:
        y1 = pt[labels == lab]
        y2 = mV.Y[labels == lab]
        dataPlot.append(go.Scatter(
            x=y1,
            y=y2,
            mode='markers',
            name=lab,
            text=lab,
            marker=dict(
                sizemode='diameter',
                sizeref=0.85,
                size=Phi[:, gp_num],
                showscale=True)
            ))
    py.plot(dataPlot, filename=strid)
            


def plotVBCode(mV,figsizeIn=(20,10),lw=3.,fs=10,labels=None, fPlotPhi=True, fPlotVar=False):
    fig=plt.figure(figsize=figsizeIn)
    B=mV.kern.branchkernelparam.Bv._array.flatten()
    assert B.size == 1, 'Code limited to one branch point, got ' + str(B.shape)
    pt = mV.t
    l = np.min(pt)
    u = np.max(pt)
    d = 0 # constraint code to be 1D for now
    for f in range(1, 4):
        if(f == 1):
            ttest = np.linspace(l, B, 100)[:, None]  # root
        else:
            ttest = np.linspace(B, u, 100)[:, None]
        Xtest = np.hstack((ttest, ttest*0+f))
        mu, var = mV.predict_f(Xtest)
        assert np.all(np.isfinite(mu)), 'All elements should be finite but are ' + str(mu)
        assert np.all(np.isfinite(var)), 'All elements should be finite but are ' + str(var)
        mean, = plt.plot(ttest, mu[:, d], linewidth=lw)
        col = mean.get_color()
        if(fPlotVar):
            plt.plot(ttest.flatten(), mu[:, d] + 2*np.sqrt(var.flatten()), '--', color=col, linewidth=lw)
            plt.plot(ttest, mu[:, d] - 2*np.sqrt(var.flatten()), '--', color=col, linewidth=lw)

    v = plt.axis()
    plt.plot([B, B], v[-2:], '-m',linewidth=lw)
    
    # Plot Phi or labels
    if(fPlotPhi):
        Phi=FlattenPhi(mV)
        gp_num = 1 # can be 0,1,2 - Plot against this
        plt.scatter(pt, mV.Y[:,d], c=Phi[:, gp_num], vmin=0., vmax=1, s=40)
        plt.colorbar(label='GP {} assignment probability'.format(gp_num))
    elif(labels is not None):
        # plot labels
        labelLegend = np.unique(labels)
        with plt.style.context('seaborn-whitegrid'):
            colors = cm.spectral(np.linspace(0, 1, len(labelLegend)))
            for lab,c in zip(labelLegend,colors):
                y1 = pt[labels == lab]
                y2 = mV.Y[labels == lab]
                plt.scatter(y1,y2,label=lab, c=c,s=80)
                plt.text(np.median(y1),np.median(y2),lab, fontsize=45, color='blue')
            plt.legend(loc='upper left')
    return fig
            
def InitModels(pt, XExpanded, Y, fMatern32):
    # code that's a bit crappy - we dont need this
    tree = bt.BinaryBranchingTree(0,90,fDebug=False) # set to true to print debug messages
    tree.add(None,1,10) # single branching point
    (fm, _) = tree.GetFunctionBranchTensor()
    #KbranchVB = bk.BranchKernelParam(GPflow.kernels.RBF(1), fm, BvInitial=np.ones((1,1))) + GPflow.kernels.White(1) 
    if(fMatern32):
        kern = GPflow.kernels.Matern32(1)        
    else:
        kern = GPflow.kernels.RBF(1)

    KbranchVB = bk.BranchKernelParam(kern, fm, BvInitial=np.ones((1,1))) + GPflow.kernels.White(1) + GPflow.kernels.Linear(1) + GPflow.kernels.Constant(1) # other copy of kernel
            
    KbranchVB.branchkernelparam.Bv.fixed = True
    mV = assigngp_dense.AssignGP(pt, XExpanded, Y, KbranchVB)
    mV.kern.white.variance = 1e-6
    mV.kern.white.variance.fixed = True
    mV._compile() # creates objective function
    return mV

def FlattenPhi(mV):
    # return flattened and rounded Phi i.e. N X 3
    phiFlattened = np.zeros((mV.Y.shape[0],3)) # only single branching point
    Phi = np.round(np.exp(mV.logPhi._array),decimals=4)
    iterC=0
    for i,_ in enumerate(mV.t):
        phiFlattened[i,:] = Phi[i,iterC:iterC+3] 
        iterC+=3
    return phiFlattened

def plotGene(t,g,labels):    
    labelLegend = np.unique(labels)
    with plt.style.context('seaborn-whitegrid'):
        colors = cm.spectral(np.linspace(0, 1, len(labelLegend)))
        fig = plt.figure(figsize=(10, 10))
        for lab,c in zip(labelLegend,colors):
            y1 = t[labels == lab]
            y2 = g[labels == lab]
            plt.scatter(y1,y2,label=lab, c=c,s=80)
            plt.text(np.median(y1),np.median(y2),lab, fontsize=45, color='blue')
        plt.legend(loc='upper left')
        return fig


def RunAnalysis(fMatern32, fTestRun, fUsePseudoTime):
    # Get number of cores reserved by the batch system (NSLOTS is automatically set)
    runNode = os.environ.get('runNode')  # will return `None` if a key is not present
    n_cores = os.environ.get('NSLOTS')
    if(n_cores is None):
        n_cores = multiprocessing.cpu_count()
    else:
        n_cores = int(n_cores)
    print("Using", n_cores, "core(s)")

    strDataDir = 'modelfiles/'+runNode+'PerGene_Matern_'+str(fMatern32)+'_PT_'+str(fUsePseudoTime) + '/'
    strFig = strDataDir + 'figs/'
    pyplotPrepend = 'Matern_'+str(fMatern32)+'_PT_'+str(fUsePseudoTime)
    
    print 'Saving figures to ' + str(strFig) + ' data to ' + strDataDir
    
    if not os.path.exists(strDataDir):
        os.makedirs(strDataDir)
    if not os.path.exists(strFig):
        os.makedirs(strFig)
            
    print  'Load gene expression'
    data = pods.datasets.singlecell()
    genes = data['Y']  
    Yall =  genes.values    
    t0=time.time()  
    N = Yall.shape[0]
    labels = data['labels']
    stageCell = np.zeros(N)
    stageN = np.zeros(N)
    for i,l in enumerate(labels):
        stageCell[i] = int(l[:2])
        stageN[i] = np.log2(stageCell[i]) + 1
            
    # GPLVM with capture times
    # Additive kernel        
    ct = stageN-3
    if not fUsePseudoTime:
        pt = ct
    else:
        # estimate pseudotime
        if(fMatern32):
            k1=GPy.kern.Matern32(1,ARD=True,active_dims=[0])
            k2=GPy.kern.Matern32(1,ARD=True,active_dims=[1])
        else:
            k1=GPy.kern.RBF(1,ARD=True,active_dims=[0])
            k2=GPy.kern.RBF(1,ARD=True,active_dims=[1])

        priormean=np.hstack([np.expand_dims(ct,1),np.zeros( (N,1))])
        priorstd= 0.3*np.ones((N,2))
        
        np.random.seed(0)
        # import GPy.util.initialization import initialize_latent
        Xinit, _ = GPy.util.initialization.initialize_latent('PCA', 2, Yall)
        # draw from prior for initial condition? PCA is not a good initialisation because it messes up ordering
        for i in range(N):
            Xinit[i,0] = priormean[i,0] + priorstd[i,0]*np.random.randn(1)
            
        print ct[:30]
        print np.round(Xinit[:30,0])
        t0 = time.time()
        time_kernel = k1+k2
        time_model = GPy.models.BayesianGPLVM(Yall, 2, kernel=time_kernel, X=Xinit )
        
        # specify prior
        for i in range(N):
            time_model.X.mean[i, [0]].set_prior(GPy.priors.Gaussian(ct[i], 1), warning=False)
        
        time_model.likelihood.fix(Yall.std()/100)
        time_model.optimize(messages=0, max_iters=100)
        time_model.likelihood.unfix()
        if not fTestRun:
            time_model.optimize(messages=0)
        print 'Pseudotime inference took %g secs'%(time.time()-t0)
                
        if not fCluster:
            time_model.plot_latent(labels=labels)
            plt.savefig(strFig+'plotPseudotime.png', bbox_inches='tight')

        pt = time_model.X.mean.values[:,0]
        
        print 'Final GPLVM Model state'
        print time_model

    # Initial B        
    l = pt.min() + 1
    u = pt.max() - 1
    Binit = np.ones((1,1))*(l+u)/2
    print 'Pseudotime in range ' + str(l) + ','  + str(u) + '. Initial B=' + str(Binit)
    
    # state to save
    saveDict = {'pt':pt}
    if(fUsePseudoTime):
        saveDict['Xlatent'] = time_model.X.mean.values
    # save state
    pickle.dump( saveDict, open( strDataDir + 'Summary.p', "wb" ) )        
    df = pd.DataFrame(index=genes.columns,columns=['B','LikVariance','Lengthscale','KernelVariance','LinearVariance','LinearConstant','fx'])
    
    # Now do one gene at a time
    genesToLookAt = genes.columns
    if fTestRun:
        genesToLookAt =  ['Id2','Sox2']
        
    for g in genesToLookAt:     
        tf.reset_default_graph()
        print 'Processing gene ' + g
        Yg = genes[g].values[:,None]    

        if(fMatern32):
            komgp1=GPy.kern.Matern32(1) 
            komgp2=GPy.kern.Matern32(1) 
        else:
            komgp1=GPy.kern.RBF(1) 
            komgp2=GPy.kern.RBF(1) 
        
        m = OMGP(pt[:,None], Yg, K=2, variance=0.01, kernels=[komgp1, komgp2], prior_Z='DP') # use a truncated DP with K=2 UNDONE
        m.kern[0].lengthscale = 10*(u-l)
        m.kern[1].lengthscale = 10*(u-l)
        m.optimize(step_length=0.01, maxiter=30)
        
        m.optimize(step_length=0.01, maxiter=200)
        
        if not fCluster:
            fig=plt.figure(figsize=(10,10))
            m.plot()
            fig.savefig(strFig+'Gene'+g+'_OMGP.png', bbox_inches='tight')
            
        print g+': OMGP done. Starting VB model'
        _, phiInitial_invSoftmax, XExpanded = VBHelperFunctions.InitialisePhiFromOMGP(None, phiOMGP=m.phi, b=0, Y=Yg,pt=pt)   
        mV = InitModels(pt,XExpanded,Yg,fMatern32) # also do gene by gene
        
        # Initialise all model parameters using the OMGP model
        # Note that the OMGP model has different kernel hyperparameters for each latent function whereas the branching model
        # has one common set.
        mV.logPhi = phiInitial_invSoftmax # initialise allocations from OMGP
        mV.likelihood.variance = m.variance.values[0]
        # set lengthscale to maximum
        mV.kern.branchkernelparam.kern.lengthscales = np.max(np.array([m.kern[0].lengthscale.values, m.kern[1].lengthscale.values]))
        # set process variance to average
        mV.kern.branchkernelparam.kern.variance = np.mean(np.array([m.kern[0].variance.values, m.kern[1].variance.values]))
        print mV
        
        print g + ' :Bayesian optimisation'
        # 1. Use GPyOpt to learn branching point and kernel hyperparameters.
        # 1. set fixed=False for all parameters except for Phi.fixed=True
        # 1. It's still beneficial to use *VB code* rather than *Jings model* since we integrate out (approximately using VB bound) uncertainty in allocation (Phi).
        # 1. Store all intermediate values visited by GPyOpt?
        # 1. Use Matern 3/2 or 5/2 for both OMGP and our model. Actually different kernels for OMGP and our can make sense as outputs different (potentially)?
        # 1. Effect of Phi on inference of branching point?
        # 1. add assert in VB code that before branching point, allocations probs==1 for 1st function.
        #         
        # --- Optimize both B and K        
        myobj = objectiveBAndK(Binit, mV, Yg, pt, m) # pass in initial point - start at mid-point
        eps = 1e-5
        bounds = [(l,u),(eps,3*Yg.var()), (eps,pt.max()), (eps,3*Yg.var()), [0.1,5], [0.1,5]]  # B, lik var, len, var, lin var, c var
        print g+' Bounds used in optimisation: =' + str(bounds)
        BOobj = GPyOpt.methods.BayesianOptimization(f=myobj.f,  # function to optimize       
                                                    bounds=bounds)              # normalized y                       
        t0=time.time()
        if fTestRun:
            max_iter = 2
            nrestart = 3
            n_cores = 2
        else:
            max_iter = 300       
            nrestart = 30                 
        try:
            BOobj.run_optimization(max_iter,                             # Number of iterations
                               acqu_optimize_method = 'fast_random',        # method to optimize the acq. function
                               acqu_optimize_restarts = nrestart,
                               batch_method='lp',
                               n_inbatch = n_cores,                        # size of the collected batches (= number of cores)
                               eps = 1e-6)                                # secondary stop criteria (apart from the number of iterations) 
        except Exception as inst:
            print g+' ExceptCT: failed with error ' + str(inst)
            print("Unexpected error:", sys.exc_info()[0])
            continue
            
        print g+' :GPyOpt took %g secs using %g cores. ' %(time.time()-t0, n_cores)
        print g + ': Solution found by BO x_opt =  ' + str(BOobj.x_opt) + 'fx_opt = ' + str(BOobj.fx_opt)

        #  best solution
        mV.kern.branchkernelparam.Bv = BOobj.x_opt[0]
        VBHelperFunctions.InitialisePhiFromOMGP(mV, phiOMGP=m.phi, b=BOobj.x_opt[0],Y=Yg,pt=pt)   
        mV.likelihood.variance = BOobj.x_opt[1]
        mV.kern.branchkernelparam.kern.lengthscales = BOobj.x_opt[2]
        mV.kern.branchkernelparam.kern.variance = BOobj.x_opt[3]
        print g+ ': VBBound got %.2f should be %.2f. Best model is: '%(-mV.compute_log_likelihood(), BOobj.fx_opt)
        print mV
        
        # Do the plotting
        if not fCluster:
            fig=plotVBCode(mV, fPlotVar=True)        
            fig.savefig(strFig+'Gene'+g+'_mV_PhiVar.png', bbox_inches='tight')
        
            fig=plotVBCode(mV, labels=labels, fPlotPhi=False, fPlotVar=False)
            fig.savefig(strFig+'Gene'+g+'_mV_LabelsMean.png', bbox_inches='tight')
        else:
            pass
            #plotlyCode(pyplotPrepend+'_Gene'+g+'_mV_afterGPyOpt', mV, labels)

        # Save state
        saveDict[g] = BOobj.x_opt            
        pickle.dump( saveDict, open( strDataDir + 'Summary.p', "wb" ) )
        df.loc[g,:] = np.hstack([BOobj.x_opt,BOobj.fx_opt]) 
        df.to_csv(strDataDir + 'Summary.csv')

        #save BO history so we can reconstruct uncertainty by looking at marginal over B
        np.save(strDataDir + 'BOSummary',np.hstack([BOobj.X,BOobj.Y]))
        
        # Optimise some more to see if it improves fit
        if not fTestRun:
            mV.kern.branchkernelparam.Bv.fixed = True
            t0=time.time()
            try:
                mV.optimize()
            except Exception as inst:
                print g+' ExceptionCT: failed with error. mv Optimize: ' + str(inst)
                print("Unexpected error:", sys.exc_info()[0])
                continue
            finally:
                print g+' :Final optimisation took %g secs. ' %(time.time()-t0)
                print mV
                mV.kern.branchkernelparam.Bv.fixed = False # make sure we save it in free state
                np.save(strDataDir+'Gene'+g+'_VBmodel',mV.get_free_state())
                # to restore state: mVNew.set_state( np.load(str + 'VBmodel.npy'))\n",

                if not fCluster:
                    fig = plotVBCode(mV, labels=labels, fPlotPhi=False, fPlotVar=True)
                    fig.savefig(strFig+'Gene'+g+'_mV_FinalPhiVar.png', bbox_inches='tight')            
                else:
                    pass
                    #plotlyCode(pyplotPrepend+'_Gene'+g+'_mV_afterFinalOpt', mV, labels)
                    
if __name__ == "__main__":    
    fMatern32 = True
    fTestRun = True # fast or full run?
    fUsePseudoTime = True # use capture times or pseudotime?
    RunAnalysis(fMatern32, fTestRun, fUsePseudoTime)