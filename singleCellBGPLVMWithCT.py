# GPLVM with non-identical Gaussian prior
import GPflow
import pods
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
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
Y = data['Y']['Id2'][:, None]
N = Y.shape[0]
Q = 1  # latent dimensions

# Use cell stage as prior
labels = data['labels']
stageN = np.zeros((N, 1))
for i, l in enumerate(labels):
    stageN[i] = np.log2(int(l[:2])) + 1

priormean = np.reshape(stageN, (N, Q))
priorstd = 1.2 * np.ones((N, Q))

np.random.seed(0)

# Initialise from prior but with inflated variance
Xinit = np.zeros((N, Q))
# Xinit = PCA(n_components=Q).fit_transform(Y)
# print Xinit.shape
# for i in range(N):
#     Xinit[i,0] = priormean[i,0] + 5*priorstd[i,0]*np.random.randn(1)
Xinit = PCA(n_components=Q).fit_transform(Y)

plotGene(stageN, Y, labels)
plt.title('CT Times')

# print 'Capture times and rounded sample from prior - first 30 entries:'
# print stageN[:30]
# print np.round(Xinit[:30,0])

''' Bayesian GPLVM without prior '''
X_variance = np.random.uniform(0, .1, Xinit.shape)
Z = np.random.permutation(Xinit.copy())[:50]
k = GPflow.kernels.RBF(Q)
mnp = GPflow.gplvm.BayesianGPLVM(X_mean=Xinit.copy(), X_var=X_variance.copy(), Y=Y, kern=k, Z=Z.copy())
mnp.optimize()
print('Without prior lik=' + str(mnp.compute_log_likelihood()))

plotGene(Xinit, Y, labels)
plt.title('init no prior')

plotGene(mnp.X_mean.value, Y, labels)
plt.title('GPLVM model without prior')


''' Bayesian GPLVM with prior '''
for i in range(N):
    Xinit[i, 0] = priormean[i, 0] + 5 * priorstd[i, 0] * np.random.randn(1)

plotGene(Xinit, Y, labels)
plt.title('init with prior')
# X_variance = np.random.uniform(0, .1, Xinit.shape)
# Z = np.random.permutation(Xinit.copy())[:40]
k = GPflow.kernels.RBF(Q)
m = GPflow.gplvm.BayesianGPLVM(X_mean=Xinit.copy(), X_var=X_variance.copy(), Y=Y, kern=k, Z=Z.copy(),
                               X_prior_mean=priormean, X_prior_var=np.square(priorstd))
m.optimize()
# print 'With prior lik='+str(m.compute_log_likelihood())
plotGene(m.X_mean.value, Y, labels)
plt.title('GPLVM model with prior')
