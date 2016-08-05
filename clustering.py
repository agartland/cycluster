from __future__ import division
import scipy.cluster.hierarchy as sch
from bootstrap_cluster import bootstrapFeatures, bootstrapObservations
import numpy as np
import pandas as pd
from functools import partial
from comparison import _alignClusterMats, alignClusters
from preprocessing import partialCorrNormalize
from copy import deepcopy

from corrplots import partialcorr

import statsmodels.api as sm

__all__ = ['hierClusterFunc',
           'corrDmatFunc',
           'makeModuleVariables',
           'formReliableClusters',
           'labels2modules',
           'cyclusterClass',
           'meanCorr',
           'silhouette']

def corrDmatFunc(cyDf, metric='pearson-signed', dfunc=None, minN=30):
    if dfunc is None:
        if metric in ['spearman', 'pearson']:
            """Anti-correlations are also considered as high similarity and will cluster together"""
            dmat = (1 - np.abs(cyDf.corr(method=metric, min_periods=minN).values))
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

def hierClusterFunc(dmatDf, K=6, method='complete', returnLinkageMat=False):
    hclusters = sch.linkage(dmatDf.values, method = method)
    labelsVec = sch.fcluster(hclusters, K, criterion = 'maxclust')
    labels = pd.Series(labelsVec, index = dmatDf.columns)
    if not returnLinkageMat:
        return labels
    else:
        return labels, hclusters

def formReliableClusters(cyDf, dmatFunc, clusterFunc, bootstraps=500, threshold=0.5):
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
                strTuple = (cy, cyDf.sampleStr, 'N' if cyDf.normed else '', currLab, 100 * meanReliability)
                print 'Excluded %s from cluster %s %sM%s: mean reliability was %1.1f%%' % strTuple
        
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

def makeModuleVariables(cyDf, labels, sampleStr='M', dropped=None):
    """Define variable for each module by standardizing all the cytokines in the
    module and taking the mean. Can be applied to a stacked df with multiple timepoints.
    Standardization will be performed across all data.
    Each module is also standardized.

    Parameters
    ----------
    cyDf : pd.DataFrame [n x cytokines]
        Contains columns for making the module.
        May include additional columns than included in labels or dropped.
    labels : pd.Series [index: cytokines]
        Series indicating cluster labels with index containing cytokine vars in cyDf
    dropped : pd.Series [index: cytokines]
        Series indicating if a cytokine (index) should be dropped when making the module

    Returns
    -------
    out : pd.DataFrame [n x modules]
        Modules as columns, one row for every row in cyDf"""

    if dropped is None:
        dropped = pd.Series(np.zeros((labels.shape[0]), dtype = bool), index = labels.index)
    standardizeFunc = lambda col: (col - np.nanmean(col))/np.nanstd(col)
    out = None
    uLabels = np.unique(labels)
    for lab in uLabels:
        members = labels.index[(labels == lab) & (~dropped)]
        tmpS = cyDf.loc[:,members].apply(standardizeFunc, raw = True).mean(axis = 1, skipna=True)
        tmpS.name = '%s%s' % (sampleStr,lab)
        if out is None:
            out = pd.DataFrame(tmpS)
        else:
            out = out.join(tmpS)
    out = out.apply(standardizeFunc)
    return out

def meanCorr(cyDf, meanVar, cyList=None, method='pearson'):
    """Each cytokine's correlation with the mean."""
    if cyList is None:
        cyList = np.array([c for c in cyDf.columns if not c == meanVar])
    cyList = np.asarray(cyList)

    tmpCorr = np.zeros((len(cyList),3))
    for i,s in enumerate(cyList):
        tmpCorr[i,:2] = partialcorr(cyDf[s], cyDf[meanVar], method=method)
    sorti = np.argsort(tmpCorr[:,0])
    tmpCorr = tmpCorr[sorti,:]
    _, tmpCorr[:,2], _, _ = sm.stats.multipletests(tmpCorr[:,1], alpha=0.2, method='fdr_bh')
    return pd.DataFrame(tmpCorr, index=cyList[sorti], columns=['rho','pvalue','qvalue'])

def silhouette(dmatDf, labels):
    """Compute the silhouette of every analyte."""
    def oneSilhouette(cy):
        modInd = labels == labels[cy]
        a = dmatDf.loc[cy, modInd].sum()

        b = None
        for lab in labels.unique():
            if not lab == labels[cy]:
                tmp = dmatDf.loc[cy, labels==lab].sum()
                if b is None or tmp < b:
                    b = tmp
        s = (b - a)/max(b,a)
        return s
    return labels.index.map(oneSilhouette)

class cyclusterClass(object):
    def __init__(self, studyStr, sampleStr, normed, rCyDf, compCommVars=None):
        self.studyStr = studyStr
        self.sampleStr = sampleStr
        self.normed = normed
        self.cyVars = rCyDf.columns.tolist()
        self.rCyDf = rCyDf.copy()

        self.nCyDf, self.normModels = partialCorrNormalize(rCyDf, compCommVars=compCommVars, meanVar='Mean')
        self.meanS = self.nCyDf['Mean']
        self.nCyDf = self.nCyDf[self.cyVars]
        if normed:
            self.cyDf = self.nCyDf
        else:
            self.cyDf = self.rCyDf
        
        self.cyDf.sampleStr = sampleStr
        self.cyDf.normed = normed

    def applyModules(self, target):
        """Use modules from target for computing module values.

        Parameters
        ----------
        target : cyclusterClass"""

        self.pwrel = target.pwrel
        self.Z = target.Z
        self.dmatDf = target.dmatDf
        self.labels = target.labels
        self.dropped = target.dropped
        self.sampleStr = target.sampleStr
        
        self.modS = labels2modules(self.labels, dropped=self.dropped)
        self.modDf = makeModuleVariables(self.cyDf, self.labels, sampleStr=self.sampleStr, dropped=self.dropped)
        if self.normed:
            self.rModDf = makeModuleVariables(self.rCyDf, self.labels, sampleStr=self.sampleStr, dropped=self.dropped)
        else:
            self.rModDf = self.modDf

    def clusterCytokines(self, K=6, alignLabels=None, labelMap=None):
        self.pwrel, self.labels, self.dropped = formReliableClusters(self.cyDf, corrDmatFunc, partial(hierClusterFunc, K=K), threshold=0)
        if not labelMap is None:
            self.labels = self.labels.map(labelMap)
        if not alignLabels is None:
            self.labels = alignClusters(alignLabels, self.labels)
        self.modS = labels2modules(self.labels, dropped=self.dropped)
        self.modDf = makeModuleVariables(self.cyDf, self.labels, sampleStr=self.sampleStr, dropped=self.dropped)
        if self.normed:
            self.rModDf = makeModuleVariables(self.rCyDf, self.labels, sampleStr=self.sampleStr, dropped=self.dropped)
        else:
            self.rModDf = self.modDf
        _,self.Z = hierClusterFunc(self.pwrel, returnLinkageMat=True)
        self.dmatDf = corrDmatFunc(self.cyDf)

    def printModules(self, modules=None):
        tmp = labels2modules(self.labels, dropped=None)
        for m in tmp.keys():
            mStr = '%s%d' % (self.sampleStr,m)
            if modules is None or mStr == modules or mStr in modules:
                print mStr
                for c in sorted(tmp[m]):
                    if self.dropped[c]:
                        print '*',
                    print c
                print
    def modMembers(self,modStr):
        return self.modS[int(modStr[-1])]
    def meanICD(self, dmat='dmat', dropped=None):
        """Compute mean intra-cluster distance using either dmatDf or pwrel"""
        def _micd(df, labels):
            """Should this be weighted by the size of each cluster? Yes."""
            count = 0
            tot = 0
            for lab in np.unique(labels):
                members = labels.index[labels == lab]
                tmp = df[members].loc[members].values.flatten()
                count += len(tmp)
                tot += tmp.sum()
            return tot/count
        
        if dropped is None:
            tmpLabels = labels
        else:
            tmpLabels = labels.loc[~self.dropped]

        if dmat == 'dmat':
            return _micd(self.dmatDf, self.tmpLabels)
        elif dmat == 'pwrel':
            return _micd(self.pwrel, self.tmpLabels)
        else:
            raise IndexError('Value for dmat not understood (%s)' % dmat)

    def pwrelStats(self):
        """Return the mean and standard deviation of values from self.pwrel
        for all non-identical cytokines. This is representative of
        how reliable the clusters are overall. Returns mean of (1 - pwrel)"""
        vec = 1 - self.pwrel.values[np.triu_indices_from(self.pwrel, k=1)].ravel()
        return vec.mean(), vec.std()

    def randCycluster(self):
        """Return a copy of self with shuffled rows, destroying covariation
        among cytokines. Requires that each column be shuffled, independently."""
        out = deepcopy(self)
        N = out.rCyDf.shape[0]
        
        for cy in out.cyVars:
            vals = out.rCyDf[cy].values
            nonnanInd = ~np.isnan(vals)
            nonnan = vals[nonnanInd]
            rind = np.random.permutation(nonnan.shape[0])
            nonnan = nonnan[rind]
            vals[nonnanInd] = nonnan
            out.rCyDf.loc[:, cy] = vals

            vals = out.nCyDf[cy].values
            nonnan = vals[nonnanInd]
            nonnan = nonnan[rind]
            vals[nonnanInd] = nonnan
            out.nCyDf.loc[:, cy] = vals
        return out

    @property
    def name(self):
        return '%s_%s_%s_' % (self.studyStr, self.sampleStr, 'normed' if self.normed else 'raw')
    @property
    def withMean(self):
        return self.cyDf.join(self.meanS)
    @property
    def modWithMean(self):
        return self.modDf.join(self.meanS)
