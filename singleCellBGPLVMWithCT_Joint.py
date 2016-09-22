# GPLVM with non-identical Gaussian prior
from matplotlib import pyplot as plt
import GPflow
import pods
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.cm as cm
plt.close("all")


def plotGene(t, g, labels):
    plt.ion()
    plt.figure()
    colors = cm.spectral(np.linspace(0, 1, len(np.unique(labels))))
    for lab, c in zip(np.unique(labels), colors):
        y1 = t[labels == lab]
        y2 = g[labels == lab]
        plt.scatter(y1, y2, label=lab, c=c, s=80)
        plt.text(np.median(y1), np.median(y2), lab, fontsize=45, color='blue')
    plt.legend(loc='upper left')


# Load simple dataset
data = pods.datasets.singlecell()
genes = data['Y']
Y = data['Y'].values
Yg = data['Y']['Id2'][:, None]
N = Y.shape[0]
Q = 1  # latent dimensions
numInducing = 30  # number of inducing points

# Use cell stage as prior
labels = data['labels']
stageN = np.zeros((N, 1))
for i, l in enumerate(labels):
    stageN[i] = np.log2(int(l[:2])) + 1
priormean = np.reshape(stageN, (N, Q))
priorstd = 20*np.ones((N, Q))
np.random.seed(0)
print 'Initialise from random'
Xinit = np.random.rand(N, Q)
plotGene(Xinit, Yg, labels)
plt.title('init with prior')
k = GPflow.kernels.RBF(Q)
X_variance = np.random.uniform(0, 10, Xinit.shape)
Z = np.random.permutation(Xinit.copy())[:numInducing]
m = GPflow.gplvm.BayesianGPLVM(X_mean=Xinit.copy(), X_var=X_variance.copy(), Y=Y, kern=k, Z=Z.copy(),
                               X_prior_mean=priormean, X_prior_var=np.square(priorstd))
m.optimize()
# print 'With prior lik='+str(m.compute_log_likelihood())
plotGene(m.X_mean.value, Yg, labels)
plt.title('GPLVM model with prior')
