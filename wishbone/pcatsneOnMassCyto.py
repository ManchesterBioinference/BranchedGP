# Plotting and miscellaneous imports
import pandas as pd
import numpy as np
import bhtsne
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
# use seaborn plotting style defaults
sns.set()


def make_pca_plot(Data_proj, Wt_df, F_df, colorVlist, strTitle,
                  figsize=(10, 10), ngrid=None, addColorbar=False):
    fig, axes = plot(Data_proj.values, colorVlist, strTitle, figsize=figsize, ngrid=ngrid, addColorbar=addColorbar)
    for ax in axes:
        ax.set_xlabel('PC1 ' + F_df.loc['PC1'].values[0])
        ax.set_ylabel('PC2 ' + F_df.loc['PC2'].values[0])
        # plot PC loadings/weights
        Wt_df_scaled = Wt_df/abs(Wt_df).max().max()   # scaled to get the loadings in (-1,1)
        for stim, wts in Wt_df_scaled.iteritems():  # plot feature by feature
            wts = wts*1.5  # x1.5 to make it more visible
            x = wts['PC1']
            y = wts['PC2']
            ax.plot((0, x), (0, y))
            ax.text(x, y, stim)
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
    return fig, axes


def do_pca(X, n_components=10, centered=True):
    '''
        Return:
            f: ratio of explained variance for each PC
            Wt: PCA loadings for each features
            Y: projected data in PCA space
    '''
    d = X
    n_features = X.shape[1]
    if n_components > n_features:
        n_components = n_features
    if isinstance(X, pd.DataFrame):
        X = X.values
    if centered:
        X = X.astype('float')  # Since X is object
        X = X - X.mean(0)
        X = X/X.std(0)
    pca = PCA(n_components=n_components)
    pca.fit(X)
    fracs = pca.explained_variance_ratio_
    F = [float("{0:.4f}".format(i)) for i in fracs]
    Y = pca.fit_transform(X)  # n_sample, n_redim
    Wt = pca.components_  # n_redim, n_feature
    # construct three DataFrames for later use:
    # PCA loadings; % of variance explained for each PC; projected data on PCs
    Wt_df = pd.DataFrame(Wt, index=['PC' + str(i) for i in np.arange(1, Wt.shape[0]+1)], columns=d.columns.values)
    F_df = pd.DataFrame([str(f*100) + '%' for f in F], index=Wt_df.index, columns=['var. %'])
    Data_proj = pd.DataFrame(Y, index=d.index, columns=['PC'+str(i) for i in np.arange(1, Y.shape[1]+1)])
    return F_df, Wt_df, Data_proj, fracs


def plot(Y, colorVlist, strTitle, figsize=(10, 10), ngrid=None, addColorbar=False):
    if(isinstance(Y, pd.DataFrame)):
        Y = Y.values
    Ys = Y/abs(Y).max().max()  # scaled to get points in (-1,1)
    height = int(2 * np.ceil(len(colorVlist) / 5))
    width = 5
    if(ngrid is not None):
        assert len(ngrid) == 2
        n_rows = ngrid[0]
        n_cols = ngrid[1]
    else:
        n_rows = int(height / 2)
        n_cols = int(width / 2)
        print('Creating nrows=%g ncols=%g' % (n_rows, n_cols))
    fig, axes = plt.subplots(n_rows,  n_cols, figsize=figsize, sharex=True, sharey=True)
    axes = np.array(axes).flatten()
    for i, colorV in enumerate(colorVlist):
        s = axes[i].scatter(Ys[:, 0], Ys[:, 1], s=8, c=colorV, alpha=0.8, cmap=plt.cm.get_cmap('nipy_spectral'))
        if(addColorbar):
            fig.colorbar(s, ax=axes[i])  # does not work with plotly
        axes[i].set_title(strTitle[i])
    return fig, axes

if __name__ == '__main__':
    fDoTsne = False
    d = pd.read_csv('data/masscytof.csv', index_col=0)
    print('data shape\n', d.shape, 'head\n', d.head())
    plt.close('all')
    plt.ion()
    if(fDoTsne):
        # tsne projection - fitting takes about 15 seconds
        tsne = bhtsne.tsne(d, perplexity=30)
        # tsne plot
        plot(tsne, [d.sum(axis=1), d['CD8']], ['tSNE: Cell size', 'tSNE: CD8'])
        # add expression counts for proxy of cell size

    F, Wt, X_proj, fracs = do_pca(d)
    plot(X_proj, [d.sum(axis=1), d['CD8']], ['PCA: Cell size', 'PCA: CD8'])   # add expression counts for proxy of cell size

    plt.figure(figsize=(5, 5))
    plt.plot(np.cumsum(fracs))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    print('PCA var explained\n', F.head(2))
    print('PCA PC loadings\n', Wt.head(2))

    make_pca_plot(X_proj, Wt, F, [d.sum(axis=1), d['CD8']], ['PCA: Cell size', 'PCA: CD8'])
