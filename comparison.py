
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn import metrics
import pandas as pd
import statsmodels.api as sm
from corrplots import partialcorr
from functools import partial
from scipy import stats

import cycluster as cy

__all__ = ['compareClusters',
           'alignClusters',
           'crossCompartmentCorr',
           'pwdistComp',
           'pwdistCompXY',
           'pwdistCompCI',
           'moduleCorrRatio']

def compareClusters(labelsA, labelsB, method='ARI', alignFirst=True, useCommon=False):
    """Requre that labelsA and labelsB have the same index"""
    if useCommon:
        labelsA, labelsB = labelsA.align(labelsB, join='inner')
    assert len(labelsA.index) == len(labelsB.index)
    assert (labelsA.index == labelsB.index).sum() == len(labelsA.index)
    uLabels = np.unique(labelsA)
    assert (uLabels == np.unique(labelsB)).sum() == uLabels.shape[0]

    if alignFirst:
        alignedB = alignClusters(labelsA, labelsB)
    else:
        alignedB = labelsB

    if method == 'ARI':
        s = metrics.adjusted_rand_score(labelsA.values, alignedB.values)
    elif method == 'AMI':
        s = metrics.adjusted_mutual_info_score(labelsA.values, alignedB.values)
    elif method == 'overlap':
        s = np.zeros(uLabels.shape[0])
        for labi, lab in enumerate(uLabels):
            membersA = labelsA.index[labelsA == lab]
            membersB = alignedB.index[alignedB == lab]
            accA = np.sum([1 for cy in membersA if cy in membersB]) / len(membersA)
            accB = np.sum([1 for cy in membersB if cy in membersA]) / len(membersB)
            s[labi] = (accA + accB) / 2

    return s

def _alignClusterMats(matA, matB):
    """Returns a copy of matB with columns shuffled to maximize overlap with matA
    matX is a representation of cluster labels using a sparse\binary np.ndarray
    with labels along the columns"""
    out = matB.copy()

    nCols = matA.shape[1]
    
    swaps = {}
    for colA in range(nCols):
        match = np.argmax([(matA[:, colA] * matB[:, colB]).sum() for colB in range(nCols)])
        swaps.update({match:colA})
    if len(swaps) == nCols:
        """Easy 1:1 matching"""
        for colB, colA in list(swaps.items()):
            out[:, colA] = matB[:, colB]

    """In case the clusters aren't clearly 1:1 then try extra swaps until the optimum is found"""
    niters = 0
    while True:
        swaps = []
        curProd = (matA * out).sum()
        for ai, bi in itertools.product(list(range(nCols)), list(range(nCols))):
            ind = np.arange(nCols)
            ind[ai] = bi
            ind[bi] = ai
            newProd = (matA * out[:, ind]).sum()
            if curProd < newProd:
                swaps.append((ai, bi, newProd))
        if len(swaps) == 0:
            break
        else:
            ai, bi, newProd = swaps[np.argmax([x[2] for x in swaps])]
            ind = np.arange(nCols)
            ind[ai] = bi
            ind[bi] = ai
            out = out[:, ind]
    return out

def _alignSparseDf(dfA, dfB):
    out = _alignClusterMats(dfA.values, dfB.values)
    return pd.DataFrame(out, index = dfB.index, columns = dfB.columns)

def _labels2sparseDf(labels):
    labelCols = np.unique(labels)
    clusterMat = np.zeros((labels.shape[0], labelCols.shape[0]), dtype = np.int32)
    for labi, lab in enumerate(labelCols):
        ind = (labels==lab).values
        clusterMat[ind, labi] = 1
    return pd.DataFrame(clusterMat, index = labels.index, columns = labelCols)

def _sparseDf2labels(sparseDf):
    labels = pd.Series(np.zeros(sparseDf.shape[0]), index = sparseDf.index)
    for clusterCol in sparseDf.columns:
        labels[sparseDf[clusterCol].astype(bool)] = clusterCol
    return labels.astype(np.int32)

def alignClusters(labelsA, labelsB):
    """Returns a copy of labelsB with renamed labels shuffled to maximize overlap with matA"""
    sparseA = _labels2sparseDf(labelsA)
    sparseB = _labels2sparseDf(labelsB)
    outLabelsB = _sparseDf2labels(_alignSparseDf(sparseA, sparseB))
    return outLabelsB

def crossCompartmentCorr(dfA, dfB, method='pearson'):
    """Cytokine correlation for those that are common to both A and B"""
    cyList = np.array([cy for cy in dfA.columns if cy in dfB.columns])
    joinedDf = pd.merge(dfA[cyList], dfB[cyList], suffixes=('_A', '_B'), left_index=True, right_index=True)
    tmpCorr = np.zeros((len(cyList), 3))
    for i, cy in enumerate(cyList):
        tmp = joinedDf[[cy + '_A', cy + '_B']].dropna()
        tmpCorr[i, :2] = partialcorr(tmp[cy + '_A'], tmp[cy + '_B'], method=method)
    sorti = np.argsort(tmpCorr[:, 0])
    tmpCorr = tmpCorr[sorti,:]
    _, tmpCorr[:, 2], _, _ = sm.stats.multipletests(tmpCorr[:, 1], method='fdr_bh')
    return pd.DataFrame(tmpCorr, index=cyList[sorti], columns=['rho', 'pvalue', 'qvalue'])

def pwdistCompXY(dmatA, dmatB):
    """Return unraveled upper triangles of the two distance matrices
    using only common columns.

    Parameters
    ----------
    dmatA, dmatB : pd.DataFrame [nfeatures x nfeatures]
        Symetric pairwise distance matrices for comparison.
        Only common columns will be used for comparison (at least 3).

    Returns
    -------
    vecA, vecN : pd.Series"""

    cyVars = [c for c in dmatA.columns if c in dmatB.columns.tolist()]
    n = len(cyVars)
    vecA = dmatA[cyVars].loc[cyVars].values[np.triu_indices(n, k=1)]
    vecB = dmatB[cyVars].loc[cyVars].values[np.triu_indices(n, k=1)]
    return vecA, vecB

def pwdistComp(dmatA, dmatB, method='spearman', nperms=10000, returnPermutations=False):
    """Compare two pairwise distance matrices
    using a permutation test. Test the null-hypothesis that
    the pairwise distance matrices are uncorrelated.

    Note: comparison is based only on shared columns

    Parameters
    ----------
    dmatA, dmatB : pd.DataFrame [nfeatures x nfeatures]
        Symetric pairwise distance matrices for comparison.
        Only common columns will be used for comparison (at least 3).
    method : str
        Method for comparison: "pearson", "spearman"
    nperms : int
        Number of permutations to compute p-value

    Returns
    -------
    stat : float
        Correlation statistic, rho, of all pairwise distances between cytokines.
    pvalue : float
        Two-sided pvalue testing the null hypothesis that
        the distance matrices of dfA and dfB are uncorrelated
    commonVars : list
        List of the common columns in A and B"""
    
    def corrComp(dmatA, dmatB, method):
        n = dmatB.shape[0]
        if method == 'pearson':
            rho, p = stats.pearsonr(dmatA[np.triu_indices(n, k=1)], dmatB[np.triu_indices(n, k=1)])
        elif method == 'spearman':
            rho, p = stats.spearmanr(dmatA[np.triu_indices(n, k=1)], dmatB[np.triu_indices(n, k=1)])
        else:
            raise ValueError('Must specify method as "pearson" or "spearman"')
        return rho
    
    cyVars = [c for c in dmatA.columns if c in dmatB.columns.tolist()]
    ncols = len(cyVars)

    compFunc = partial(corrComp, method=method)

    dA = dmatA[cyVars].loc[cyVars].values
    dB = dmatB[cyVars].loc[cyVars].values
    
    stat = compFunc(dA, dB)
    permstats = np.zeros(nperms)
    for i in range(nperms):
        """Permutation of common columns"""
        rindA = np.random.permutation(ncols)
        rindB = np.random.permutation(ncols)
        permstats[i] = compFunc(dA[rindA,:][:, rindA], dB[rindB,:][:, rindB])
    pvalue = ((np.abs(permstats) > np.abs(stat)).sum() + 1)/(nperms + 1)

    out = (stat, pvalue, cyVars)
    if returnPermutations:
        return out + (permstats,)
    else:
        return out

def pwdistCompCI(dfA, dfB, dmatFunc=None, alpha=0.05, method='spearman', nstraps=10000, returnBootstraps=False):
    """Compare two pairwise distance matrices
    and compute bootstrap confidence intervals.

    Note: comparison is based only on shared columns

    Parameters
    ----------
    dfA, dfA : pd.DataFrame [nfeatures x nfeatures]
        Symetric pairwise distance matrices for comparison.
        Only common columns will be used for comparison (at least 3).
    method : str
        Method for comparison: "pearson", "spearman"
    nstraps : int
        Number of bootstraps used to compute confidence interval.

    Returns
    -------
    lb : float
        Lower bound of the confidence interval covering (1 - alpha)%
    stat : float
        Correlation statistic, rho, of all pairwise distances between cytokines.
    ub : float
        Upper bound of the confidence interval covering (1 - alpha)%"""
    
    def corrComp(dmatA, dmatB, method):
        n = dmatB.shape[0]
        if method == 'pearson':
            rho, p = stats.pearsonr(dmatA[np.triu_indices(n, k=1)], dmatB[np.triu_indices(n, k=1)])
        elif method == 'spearman':
            rho, p = stats.spearmanr(dmatA[np.triu_indices(n, k=1)], dmatB[np.triu_indices(n, k=1)])
        else:
            raise ValueError('Must specify method as "pearson" or "spearman"')
        return rho

    cyVars = [c for c in dfA.columns if c in dfB.columns.tolist()]
    ncols = len(cyVars)

    compFunc = partial(corrComp, method=method)

    dA = dfA[cyVars]
    dB = dfB[cyVars]
    
    strapped = np.zeros(nstraps)
    for i in range(nstraps):
        if dmatFunc is None:
            tmpA = dA.sample(frac=1, replace=True, axis=0).corr()
            tmpB = dB.sample(frac=1, replace=True, axis=0).corr()
        else:
            tmpA = dmatFunc(dA.sample(frac=1, replace=True, axis=0))
            tmpB = dmatFunc(dB.sample(frac=1, replace=True, axis=0))
        strapped[i] = compFunc(tmpA.values, tmpB.values)
    
    out = tuple(np.percentile(strapped, [100*alpha/2, 50, 100*(1-alpha/2)]))
    if returnBootstraps:
        out += (strapped,)
    return out

def moduleCorrRatio(cyDf, labels, cyVars=None, alpha=0.05, nstraps=10000):
    """Compute all pairwise intra- and inter-module cytokine correlation
    coefficients with their IQRs.

    Additionally compute the intra : inter ratio with 95% CI, where the
    ratio is of signed-pearson correlation coefficients transformed to
    the [0,1] interval with 0 meaning perfect anti-correlation
    and 1 meaning perfect correlation
    
    For ratio, uses a signed Pearson correlation coefficient since this is what is used
    for clustering. The disadvantage is that it can't be described as fractional
    variance, while the upside is that it captures the potential problem with
    forming modules of anti-correlated cytokines.

    Parameters
    ----------
    cyDf : pd.DataFrame [n_participants x n_cytokines]
        Raw or normalized analyte log-concentrations.
    labels : pd.Series
        Module labels for each analyte

    Returns
    -------
    intra : np.ndarray shape (3,)
        Vector containing 25th, 50th and 75th quantiles of all cytokine pairs within the same module.
    inter : np.ndarray shape (3,)
        Vector containing 25th, 50th and 75th quantiles of all cytokine pairs from different modules.
    ratio : np.ndarray shape (3,)
        Vector containing the intra : inter correlation ratio with bootstrap 95% CI or (1 - alpha)%
        [LB, ratio, UB]"""

    def ratioFunc(cyDf, intraMask, interMask):
        """smat is on the [0, 1] interval with 0 meaning perfect anti-correlation and 1 meaning perfect correlation"""
        smat = 1 - cy.corrDmatFunc(cyDf, metric='pearson-signed').values
        return np.nanmean((smat * intraMask).ravel()) / np.nanmean((smat * interMask).ravel())

    if cyVars is None:
        cyVars = cyDf.columns.tolist()

    """corrmat is on the [-1, 1] interval with 1 meaning perfect correlation and -1 meaning perfect anti-correlation"""
    corrmat = cyDf[cyVars].corr()

    intra = []
    inter = []
    intraMask = np.nan * np.zeros(corrmat.shape)
    interMask = np.nan * np.zeros(corrmat.shape)
    for a, b in itertools.combinations(cyVars, 2):
        if not a == b:
            s = corrmat.loc[a, b]
            i, j = cyVars.index(a), cyVars.index(b)
            if labels[a] == labels[b]:
                intra.append(s)
                intraMask[i, j] = 1.
            else:
                inter.append(s)
                interMask[i, j] = 1.

    intra = np.percentile(intra, q=[25, 50, 75])
    inter = np.percentile(inter, q=[25, 50, 75])

    if nstraps is None or nstraps == 0:
        return intra, inter
    
    else:
        rratios = np.zeros(nstraps)
        for strapi in range(nstraps):
            rratios[strapi] = ratioFunc(cyDf[cyVars].sample(frac=1, replace=True, axis=0), intraMask, interMask)
        ratio = np.percentile(rratios, [100*alpha/2, 50, 100*(1-alpha/2)])

        return intra, inter, ratio