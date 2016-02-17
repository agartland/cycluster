from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn import metrics
import pandas as pd
import statsmodels.api as sm
from corrplots import partialcorr
from functools import partial
from scipy import stats

__all__ = ['compareClusters',
           'alignClusters',
           'crossCompartmentCorr',
           'pwdistComp',
           'pwdistCompXY']

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
        match = np.argmax([(matA[:,colA] * matB[:,colB]).sum() for colB in range(nCols)])
        swaps.update({match:colA})
    if len(swaps) == nCols:
        """Easy 1:1 matching"""
        for colB,colA in swaps.items():
            out[:,colA] = matB[:,colB]

    """In case the clusters aren't clearly 1:1 then try extra swaps until the optimum is found"""
    niters = 0
    while True:
        swaps = []
        curProd = (matA * out).sum()
        for ai,bi in itertools.product(range(nCols),range(nCols)):
            ind = np.arange(nCols)
            ind[ai] = bi
            ind[bi] = ai
            newProd = (matA * out[:,ind]).sum()
            if curProd < newProd:
                swaps.append((ai,bi,newProd))
        if len(swaps) == 0:
            break
        else:
            ai,bi,newProd = swaps[np.argmax([x[2] for x in swaps])]
            ind = np.arange(nCols)
            ind[ai] = bi
            ind[bi] = ai
            out = out[:,ind]
    return out

def _alignSparseDf(dfA, dfB):
    out = _alignClusterMats(dfA.values, dfB.values)
    return pd.DataFrame(out, index = dfB.index, columns = dfB.columns)

def _labels2sparseDf(labels):
    labelCols = np.unique(labels)
    clusterMat = np.zeros((labels.shape[0], labelCols.shape[0]), dtype = np.int32)
    for labi,lab in enumerate(labelCols):
        ind = (labels==lab).values
        clusterMat[ind,labi] = 1
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
    joinedDf = pd.merge(dfA[cyList], dfB[cyList], suffixes=('_A','_B'), left_index=True, right_index=True)
    tmpCorr = np.zeros((len(cyList),3))
    for i,cy in enumerate(cyList):
        tmp = joinedDf[[cy + '_A',cy + '_B']].dropna()
        tmpCorr[i,:2] = partialcorr(tmp[cy + '_A'], tmp[cy + '_B'], method=method)
    sorti = np.argsort(tmpCorr[:,0])
    tmpCorr = tmpCorr[sorti,:]
    _, tmpCorr[:,2], _, _ = sm.stats.multipletests(tmpCorr[:,1], method='fdr_bh')
    return pd.DataFrame(tmpCorr, index=cyList[sorti], columns=['rho','pvalue','qvalue'])

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
        permstats[i] = compFunc(dA[rindA,:][:,rindA], dB[rindB,:][:,rindB])
    pvalue = ((np.abs(permstats) > np.abs(stat)).sum() + 1)/(nperms + 1)
    out = (stat, pvalue, cyVars)
    if returnPermutations:
        return out + (permstats,)
    else:
        return out