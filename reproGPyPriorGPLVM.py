# GPLVM with non-identical Gaussian prior
import GPy
import pods
import numpy as np

# Load simple dataset
data = pods.datasets.singlecell()
genes = data['Y']
Y = data['Y']['Id2'][:, None]
N = Y.shape[0]

# Use cell stage as prior
labels = data['labels']
stageN = np.zeros((N, 1))
for i, l in enumerate(labels):
    stageN[i] = np.log2(int(l[:2])) + 1

priormean = stageN
priorstd = 0.2 * np.ones((N, 1))

np.random.seed(0)

# Initialise from prior
Xinit = np.zeros((N, 1))
for i in range(N):
    Xinit[i, 0] = priormean[i, 0] + priorstd[i, 0] * np.random.randn(1)

print('Capture times and rounded sample from prior - first 30 entries:')
print(stageN[:30])
print(np.round(Xinit[:30, 0]))

# X_prior = GPy.core.parameterization.priors.MultivariateGaussian(priormean, np.diag(priorstd.flatten()))
# m = GPy.models.DPBayesianGPLVM(Y, 1, X_prior,  kernel=GPy.kern.RBF(1), X=Xinit )
# This result in error
# ValueError: too many values to unpack
# because pdinv returns more values than expected

''' Bayesian GPLVM with MAP prior '''
X_variance = np.random.uniform(0, .1, Xinit.shape)
Z = np.random.permutation(Xinit.copy())[:50]
m = GPy.models.BayesianGPLVM(Y, 1, kernel=GPy.kern.RBF(1), X=Xinit, X_variance=X_variance, num_inducing=50, Z=Z)
print('m likelihood with no prior specified ' + str(m.log_likelihood()))
print('m objective with no prior specified ' + str(m.objective_function()))  # does not change
# Set prior
for i in range(N):
    m.X.mean[i, [0]].set_prior(GPy.priors.Gaussian(priormean[i, 0], priorstd[i, 0]), warning=False)
print('m likelihood with prior' + str(m.log_likelihood()))  # does not change
print('m objective with prior ' + str(m.objective_function()))  # does not change

'''
As we can see the prior is only used as MAP prior (see Model.objective_function)
The VB model uses a standard normal for the KL term (see variational.NormalPrior)
This should be generalised to allow any normal prior including a multivariate normal.
'''

m.likelihood.fix(Y.std() / 100)
m.optimize(messages=1, max_iters=100)
m.likelihood.unfix()
m.optimize(messages=1)

# plotting code
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

plotGene(stageN, Y, labels)
plt.title('CT Times')

plotGene(m.X.mean.values, Y, labels)
plt.title('GPLVM model')
