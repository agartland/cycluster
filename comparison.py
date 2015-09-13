from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn import metrics
import pandas as pd
import statsmodels as sm
from corrplots import partialcorr

__all__ = ['compareClusters',
           'plotClusterOverlap',
           'alignClusters',
           'crossCompartmentCorr']

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

def plotClusterOverlap(labelsA, labelsB, useCommon=False):
    if useCommon:
        labelsA, labelsB = labelsA.align(labelsB, join='inner')
    def _thickness(labelsA, labelsB, a, b):
        indA = labelsA == a
        indB = labelsB == b
        return 2 * (indA & indB).sum()/(indA.sum() + indB.sum())
    
    alignedB = alignClusters(labelsA, labelsB)
    
    yA = np.linspace(10,0,np.unique(labelsA).shape[0])
    yB = np.linspace(10,0,np.unique(labelsB).shape[0])

    axh = plt.gca()
    axh.cla()
    annParams = dict(ha = 'center', va = 'center', size = 'x-large', zorder = 15)
    for ai, a in enumerate(np.unique(labelsA)):
        axh.annotate(s = '%s' % a, xy = (0,yA[ai]), color = 'black', **annParams)
        for bi, b in enumerate(np.unique(alignedB)):
            if ai == 0:
                axh.annotate(s = '%s' % b, xy = (1,yB[bi]), color = 'white', **annParams)
            axh.plot([0,1], [yA[ai], yB[bi]], '-', lw = 20 * _thickness(labelsA, alignedB, a, b), color = 'black', alpha = 0.7, zorder = 5)

    axh.scatter(np.zeros(np.unique(labelsA).shape[0]), yA, s = 1000, color = 'red', zorder = 10)
    axh.scatter(np.ones(np.unique(alignedB).shape[0]), yB, s = 1000, color = 'blue', zorder = 10)
    plt.axis('off')
    plt.draw()

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

def crossCompartmentCorr(dfA, dfB, method = 'pearson', useFDR = False, sigThreshold = 0.05):
    """Plot of cytokine correlation for those that are common to both A and B"""
    commonCy = [cy for cy in dfA.columns if cy in dfB.columns]
    tmpCorr = np.zeros((len(commonCy),2))
    for i,cy in enumerate(commonCy):
        tmpCorr[i,0], tmpCorr[i,1] = partialcorr(dfA[cy], dfB[cy], method = method)
    sorti = np.argsort(tmpCorr[:,0])
    tmpCorr = tmpCorr[sorti,:]

    if useFDR:
        """Use q-value significance threshold"""
        sigInd, qvalues, _, _ = sm.stats.multipletests(tmpCorr[:,1], alpha = sigThreshold, method = 'fdr_bh')
    else:
        """Use p-value significance threshold"""
        sigInd = tmpCorr[:,1] < sigThreshold

    axh = plt.gca()
    axh.cla()
    axh.barh(arange(tmpCorr.shape[0])[~sigInd], tmpCorr[~sigInd,0]**2, color = 'black', align='center')
    axh.barh(arange(tmpCorr.shape[0])[sigInd], tmpCorr[sigInd,0]**2, color = 'red', align='center')
    plt.yticks(range(tmpCorr.shape[0]), np.array([commonCy])[sorti])
    plt.grid(True, axis = 'x')
    plt.xlabel('Cross compartment correlation ($^*R^2$)')
    plt.ylim((-1,tmpCorr.shape[0]))
    plt.xlim((0,1))
    tight_layout()
