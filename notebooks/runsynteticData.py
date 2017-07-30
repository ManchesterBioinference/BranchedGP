import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

datafile='syntheticdata/synthetic20.csv'
data = pd.read_csv(datafile, index_col=[0])
G = data.shape[1] - 2  # all data - time columns - state column
Y = data.iloc[:, 2:]

plt.ion()
f, ax = plt.subplots(5, 8, figsize=(10, 8))
ax = ax.flatten()
for i in range(G):
    for s in np.unique(data['MonocleState']):
        idxs = s == data['MonocleState']
        ax[i].scatter(data['Time'].loc[idxs], Y.iloc[:, i].loc[idxs])
        ax[i].set_title(Y.columns[i])
        ax[i].set_yticklabels([])
        ax[i].set_xticklabels([])
f.suptitle('Branching genes, location=1.1 indicates no branching')

