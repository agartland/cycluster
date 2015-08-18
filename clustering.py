import scipy.cluster.hierarchy as sch
import itertools
import palettable

from corrplots import *
from hclusterplot import *
from gapstat import *
from bootstrap_cluster import *

'''
method = 'complete' #single, centroid (UPGMC), complete, average, ward
metric = 'pearson-signed'
minPatients = 30
K = 6
tiss = 'ETT'
dmatFunc = partial(computeDMat, metric = metric, minN = minPatients)
clusterFunc = partial(clusterFuncK, K = K, method = method)

"""Use the gap statistic to determine the number of clusters"""
for normStr,tmpDf in zip(['normed','raw'],(ndf,df)):
    for tiss in ['ETT','Serum']:
        gs = computeGapStat(tmpDf[cyVars[tiss]], dmatFunc, partial(clusterFuncK,method = method), maxK = 15, bootstraps = 20)
        figure(50, figsize = (10,8.5))
        plotGapStat(*gs)
        figure(50).savefig(DATA_PATH + 'RandolphFlu/figures/PIC_%s_%s_gap_statistic.png' % (tiss,normStr))

"""Use bootstrap_clustering to determine the reliable clusters in PIC"""
modulesPICN = {}
modulesPIC = {}
pwrelPICN = {}
pwrelPIC = {}
for tiss in ['ETT','Serum']: 
    pwrelPICN[tiss], modulesPICN[tiss] = findReliableClusters(ndf, tiss, cyVars, dmatFunc, clusterFunc, 'PIC_normed')
    pwrelPIC[tiss], modulesPIC[tiss] = findReliableClusters(df, tiss, cyVars, dmatFunc, clusterFunc, 'PIC')

modndf = makeModuleVariables(ndf, modulesPICN)
moddf = makeModuleVariables(df, modulesPICN)

corrPlotsByCluster(ndf, modulesPICN,'PIC_normed')
corrPlotsByCluster(df, modulesPIC,'PIC')

for tiss in ['ETT','Serum']:
    plotModuleEmbedding(ndf, tiss, modulesPICN, dmatFunc, 'PIC_normed')
    plotModuleEmbedding(df, tiss, modulesPIC, dmatFunc, 'PIC')
'''

def clusterFuncK(dmat, K = 6, method = 'complete'):
    hclusters = sch.linkage(dmat, method = method)
    labels = sch.fcluster(hclusters, K, criterion = 'maxclust')
    return labels

def findReliableClusters(df, tiss, cyVars, dmatFunc, clusterFunc, studyStr, bootstraps=500, threshold=0.5):
    """Use bootstrap_clustering to determine the reliable clusters in PIC"""
    clusters = {}
    dmat = dmatFunc(df[cyVars[tiss]])
    #pwrel, labels = bootstrapFeatures(dmat, clusterFunc, bootstraps = bootstraps)
    pwrel, labels = bootstrapObservations(df[cyVars[tiss]], dmatFunc, clusterFunc, bootstraps = bootstraps)
    pwrel = pd.DataFrame(pwrel, index = cyVars[tiss], columns = cyVars[tiss])
    labels = pd.Series(labels, index = cyVars[tiss])

    pwrelPlot = pwrel.copy()

    for currLab in labels.unique():
        cyMembers = labels.index[labels == currLab]
        """Step-down: start with all members and discard fringe"""
        clusters[currLab] = cyMembers.tolist()
        for cy in cyMembers:
            if pwrel[cy].loc[cyMembers].drop(cy).mean() > threshold:
                clusters[currLab].remove(cy)
                pwrelPlot = pwrelPlot.rename_axis({cy: '*' + cy}, axis = 1)
                print 'Excluded %s from %s cluster %s: mean reliability is %1.0f%%' % (cy,tiss,currLab,100*(1-pwrel[cy].loc[cyMembers].drop(cy).mean()))
        
        """Step-up strategy: start with best and add those that fit"""
        '''
        clusters[currLab] = []
        """Best is the cytokine among the cluster that is closest to all other members"""
        bestCy = pwrel[cyMembers].loc[cyMembers].sum(axis=0).argmin()
        prntTuple = (bestCy,tiss,currLab,100*(1-(pwrel[cyMembers].loc[cyMembers].mean(axis=0).min())), (100*(1-pwrel[cyMembers].loc[cyMembers].mean(axis=0))).tolist())
        print '%s is best from %s cluster %s: mean reliability is %1.0f%% (%s)' % prntTuple
        clusters[currLab].append(bestCy)
        for cy in cyMembers.drop(bestCy):
            currMembers = clusters[currLab]
            if (pwrel[cy].loc[currMembers]<threshold).any():
                clusters[currLab].append(cy)
            else:
                print 'Excluded %s from %s cluster %s: max reliability is %1.0f%%' % (cy,tiss,currLab,100*(1-(pwrel[cy].loc[currMembers]).min()))
        '''
        if len(clusters[currLab]) <= 1:
            print 'Dropped cluster %s %s' % (tiss,currLab)
            popped = clusters.pop(currLab)

    figure(641, figsize = (15.5, 9.5))
    colInds = plotHColCluster(df = None, col_dmat = pwrelPlot, col_labels = labels, titleStr = studyStr)
    figure(641).savefig(DATA_PATH + 'RandolphFlu/figures/%s_%s_cluster_reliability.png' % (studyStr, tiss))
    return pwrel, clusters