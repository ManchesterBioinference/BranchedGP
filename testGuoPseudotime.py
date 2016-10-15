import guoCommon
import testGuoGPLVMCT
from matplotlib import pyplot as plt

fPlot = True
# Load topslam pt
ptTS, YGPLVM, Ygene, labels, labelLegend, stageCell, stageN = guoCommon.LoadMouseQPCRData()

# Baseline - just use capture time
qsCT, rcCT, rCT = guoCommon.evaluatePseudoTime(stageCell, Ygene, labels, stageCell)
print('=================\nCapture time\n=================\npseudotime rank correlation with capture time %.3f\n' % rcCT,
      'median and 90% CI for roughness\n', qsCT)

# Topslam
qsTS, rcTS, rTS = guoCommon.evaluatePseudoTime(ptTS, Ygene, labels, stageCell)
Ygene.to_csv('data/GuoDataMaxProcessssed.csv')

print('=================\nTopslam\n=================\npseudotime rank correlation with capture time %.3f\n' % rcTS,
      'median and 90% CI for roughness\n', qsTS)

# Load DeLorean time, UNDONE this needs to be rerun on exact same data
# UNDONE

# different GPLVM models with capture time
ptGPLVM2, elapsedTime2, m2 = testGuoGPLVMCT.guoGPLVM_CT(Ygene, stageN, fSphere=True, fFixedZ=True, Q=2,
                                                        priorStd=0.5, xvar=.01, M=60, fInitialiseRandom=False, fMatern=False)
qsGP2, rcGP2, rGP2 = guoCommon.evaluatePseudoTime(ptGPLVM2, Ygene, labels, stageCell)
print('=================\nGPLVM 2-D PT \n=================\npseudotime rank correlation with capture time %.3f\n' % rcGP2,
      'median and 90% CI for roughness\n', qsGP2, 'elapsed time %g secs' % elapsedTime2)


# different GPLVM models with capture time
ptGPLVM, elapsedTime, m = testGuoGPLVMCT.guoGPLVM_CT(Ygene, stageN, fSphere=True, fFixedZ=False, Q=1,
                                                     priorStd=0.5, xvar=.01, M=40, fInitialiseRandom=False, fMatern=False)
qsGP, rcGP, rGP = guoCommon.evaluatePseudoTime(ptGPLVM, Ygene, labels, stageCell)
print('=================\nGPLVM 1-D PT \n=================\npseudotime rank correlation with capture time %.3f\n' % rcGP,
      'median and 90% CI for roughness\n', qsGP, 'elapsed time %g secs' % elapsedTime)

if(fPlot):
    plt.ion()
    guoCommon.plotGene(ptTS, Ygene['Pdgfra'], labels.values)
    plt.title('TopSlam')
    guoCommon.plotGene(ptGPLVM2, Ygene['Pdgfra'], labels.values)
    plt.title('GPLVM 2D')
    guoCommon.plotGene(ptGPLVM, Ygene['Pdgfra'], labels.values)
    plt.title('GPLVM 1D')