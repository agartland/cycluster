from __future__ import division
import scipy.cluster.hierarchy as sch
#import scipy.spatial.distance as distance
from gapstat import computeGapStat
from bootstrap_cluster import bootstrapFeatures, bootstrapObservations
import numpy as np
import pandas as pd
from sklearn.decomposition import KernelPCA, PCA
from sklearn.mixture import GMM, DPGMM
from comparison import _alignClusterMats, alignClusters

__all__ = ['hierClusterFunc',
           'gmmClusterFunc',
           'corrDmatFunc',
           'makeModuleVariables',
           'formReliableClusters',
           'labels2modules',
           'cyclusterClass']

def corrDmatFunc(cyDf, metric = 'pearson-signed', dfunc = None, minN = 30):
    if dfunc is None:
        if metric in ['spearman', 'pearson']:
            """Anti-correlations are also considered as high similarity and will cluster together"""
            dmat = (1 - cyDf.corr(method = metric, min_periods = minN).values**2)
            dmat[np.isnan(dmat)] = 1
        elif metric in ['spearman-signed', 'pearson-signed']:
            """Anti-correlations are considered as dissimilar and will NOT cluster together"""
            dmat = ((1 - cyDf.corr(method = metric.replace('-signed',''), min_periods = minN).values) / 2)
            dmat[np.isnan(dmat)] = 1
        else:
            raise NameError('metric name not recognized')
    else:
        ncols = cyDf.shape[1]
        dmat = np.zeros((ncols, ncols))
        for i in range(ncols):
            for j in range(ncols):
                """Assume distance is symetric"""
                if i <= j:
                    tmpdf = cyDf.iloc[:,[i,j]]
                    tmpdf = tmpdf.dropna()
                    if tmpdf.shape[0] >= minN:
                        d = dfunc(cyDf.iloc[:,i],cyDf.iloc[:,j])
                    else:
                        d = np.nan
                    dmat[i,j] = d
                    dmat[j,i] = d
    return pd.DataFrame(dmat, columns = cyDf.columns, index = cyDf.columns)

def hierClusterFunc(dmatDf, K = 6, method = 'complete', returnLinkageMat = False):
    hclusters = sch.linkage(dmatDf.values, method = method)
    labelsVec = sch.fcluster(hclusters, K, criterion = 'maxclust')
    labels = pd.Series(labelsVec, index = dmatDf.columns)
    if not returnLinkageMat:
        return labels
    else:
        return labels, hclusters

def formReliableClusters(cyDf, dmatFunc, clusterFunc, bootstraps = 500, threshold = 0.5):
    """Use bootstrap_clustering to determine the reliable clusters"""
    clusters = {}
    dmatDf = dmatFunc(cyDf)
    #pwrel, labels = bootstrapFeatures(dmat, clusterFunc, bootstraps = bootstraps)
    pwrelDf, labels = bootstrapObservations(cyDf, dmatFunc, clusterFunc, bootstraps = bootstraps)
    
    dropped = pd.Series(np.zeros(cyDf.shape[1]).astype(bool), index = cyDf.columns)
    for currLab in labels.unique():
        cyMembers = labels.index[labels == currLab].tolist()
        """Step-down: start with all members and discard fringe"""
        for cy in cyMembers:
            meanReliability = (1 - pwrelDf[cy].loc[cyMembers].drop(cy).mean())
            if  meanReliability < threshold:
                dropped[cy] = True
                print 'Excluded %s from cluster %s: mean reliability was %1.1f%%' % (cy, currLab, 100 * meanReliability)
        
        """Consider step-up strategy: start with best and add those that fit"""
    return pwrelDf, labels, dropped

def labels2modules(labels, dropped = None):
    uLabels = np.unique(labels)
    out = {lab:labels.index[labels == lab].tolist() for lab in uLabels}
    if not dropped is None:
        todrop = dropped.index[dropped].tolist()
        for lab in out.keys():
            out[lab] = [cy for cy in out[lab] if not cy in todrop]
            if len(out[lab]) == 0:
                _ = out.pop(lab)

    return out

def makeModuleVariables(cyDf, labels, dropped = None):
    """Define variable for each module by standardizing all the cytokines in the module and taking the mean"""
    if dropped is None:
        dropped = pd.Series(np.zeros((labels.shape[0]), dtype = bool), index = labels.index)
    standardizeFunc = lambda col: (col - np.nanmean(col))/np.nanstd(col)
    out = None
    uLabels = np.unique(labels)
    for lab in uLabels:
        ind = (labels == lab) & (~dropped)
        tmpS = cyDf.loc[:,ind].apply(standardizeFunc, raw = True).mean(axis = 1, skipna=True)
        tmpS.name = 'M%s' % lab
        if out is None:
            out = pd.DataFrame(tmpS)
        else:
            out = out.join(tmpS)
    """Drop clusters that don't have any members"""
    out = out.dropna(axis = 1, how = 'all')
    return out

def gmmClusterFunc(cyDf, dmatFunc, minInclusionProb=0.8, K=6, n_components=4):
    """Use Gaussian Mixture Models to cluster
    The probabilities seem too high.
    Check convergence diagnostics.
    Try to plot the cluster density in 2D."""

    dmatDf = dmatFunc(cyDf)

    """First establish that with these parameters we get the same result everytime using the same data"""
    gmmParams = dict(n_components = K, n_init = 100, n_iter = 100, tol = 1e-6)
    g = GMM(**gmmParams)

    """First, reduce the dimensionality of the data (KPCA also solves the missing data problem by using pairwise corr)"""
    pca = KernelPCA(kernel='precomputed', n_components=n_components)
    gram = 1 - (dmatDf.values / dmatDf.values.max())
    xy = pca.fit_transform(gram)
    redDf = pd.DataFrame(xy, index=dmatDf.index, columns=range(n_components))

    g.fit(redDf.values)
    prob = g.predict_proba(redDf.values)

    """Each cytokine is assigned to the ML cluster, but is dropped if Pr < minInclusion"""
    probDf = pd.DataFrame(prob,index=cyDf.columns, columns=range(K))
    labels = pd.Series(np.argmax(prob, axis=1), index=cyDf.columns)
    dropped = pd.Series(np.max(prob, axis=1) < minInclusionProb, index=cyDf.columns)

    return probDf, labels, dropped

class cyclusterClass(object):
    def __init__(self, studyStr, sampleStr, normed, cyDf, compCommS):
        self.studyStr = studyStr
        self.sampleStr = sampleStr
        self.normed = normed
        self.cyDf = cyDf
        self.compCommS = compCommS
        self.cyVars = cyDf.columns.tolist()

    def clusterCytokines(self, alignLabels=None):
        self.pwrel, self.labels, self.dropped = formReliableClusters(self.cyDf, corrDmatFunc, hierClusterFunc)
        if not alignLabels is None:
            self.labels = alignClusters(alignLabels, self.labels)
        self.modS = labels2modules(self.labels, dropped = self.dropped)
        self.modDf = makeModuleVariables(self.cyDf, self.labels, dropped = self.dropped)
        _,self.Z = hierClusterFunc(self.pwrel, returnLinkageMat = True)
        self.dmatDf = corrDmatFunc(self.cyDf)

    def gmmClusterCytokines(self, alignLabels=None, minInclusionProb=0.8, K=6, n_components=4):
        self.probDf, self.labels, self.dropped = gmmClusterFunc(self.cyDf, corrDmatFunc, minInclusionProb, K, n_components)
        if not alignLabels is None:
            self.labels = alignClusters(alignLabels, self.labels)
        self.modS = labels2modules(self.labels, dropped = self.dropped)
        self.modDf = makeModuleVariables(self.cyDf, self.labels, dropped = self.dropped)
        self.dmatDf = corrDmatFunc(self.cyDf)

    def printModules(self):
        tmp = labels2modules(self.labels, dropped = None)
        for m in tmp.keys():
            print 'M%d' % m
            for c in sorted(tmp[m]):
                if self.dropped[c]:
                    print '*',
                print c
            print

    @property
    def name(self):
        return '%s_%s_%s_' % (self.studyStr, self.sampleStr, 'normed' if self.normed else 'raw')
    @property
    def withMean(self):
        return self.cyDf.join(self.compCommS)
    @property
    def modWithMean(self):
        return self.modDf.join(self.compCommS)
