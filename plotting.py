from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import palettable
import tsne
from sklearn.decomposition import KernelPCA, PCA
from sklearn.manifold import TSNE
import tsne
import itertools
from functools import partial
from myboxplot import manyboxplots

from matplotlib.gridspec import GridSpec
from matplotlib import cm
import scipy.cluster.hierarchy as sch

from corrplots import validPairwiseCounts, partialcorr, combocorrplot, crosscorr, heatmap
from hclusterplot import plotBicluster
import statsmodels.api as sm
from scipy import stats

import sklearn

import seaborn as sns
sns.set(style = 'darkgrid', palette = 'muted', font_scale = 1.75)

from cycluster import labels2modules, makeModuleVariables, meanCorr
from cycluster.comparison import *

__all__ = ['plotModuleEmbedding',
           'plotModuleCorr',
           'plotInterModuleCorr',
           'cyBoxPlots',
           'logisticRegressionBars',
           'logisticRegressionResults',
           'plotMeanCorr',
           'outcomeBoxplot',
           'plotROC',
           'plotInterModuleCorr'
           'plotClusterOverlap',
           'plotCrossCompartmentHeatmap',
           'plotCrossCompartmentBoxplot',
           'plotCrossCompartmentBars',
           'plotHierClust']

def plotModuleEmbedding(dmatDf, labels, dropped=None, method='kpca', plotLabels=True, plotDims=[0,1], weights=None):
    """Embed cytokine correlation matrix to visualize cytokine clusters"""
    uLabels = np.unique(labels).tolist()
    n_components = max(plotDims) + 1
    dmat = dmatDf.values
    
    if method == 'kpca':
        """By using KernelPCA for dimensionality reduction we don't need to impute missing values"""
        pca = KernelPCA(kernel='precomputed', n_components=n_components)
        gram = 1 - (dmat / dmat.max())
        xy = pca.fit_transform(gram)
    elif method == 'tsne':
        xy = tsne.run_tsne(dmat)
    elif method == 'sklearn-tsne':
        tsneObj = TSNE(n_components=n_components, metric='precomputed', random_state=0)
        xy = tsneObj.fit_transform(dmat)

    colors = palettable.colorbrewer.get_map('Set1', 'qualitative', len(uLabels)).mpl_colors
    figh = plt.gcf()
    figh.clf()
    axh = figh.add_axes([0.03,0.03,0.94,0.94])
    axh.axis('off')
    figh.set_facecolor('white')
    annotationParams = dict(xytext=(0,5), textcoords='offset points', size='medium')
    for cyi,cy in enumerate(dmatDf.columns):
        if not dropped is None and dropped[cy]:
            cyLab = '*' + cy
            alpha = 0.3
        else:
            cyLab = cy
            alpha = 0.8

        if plotLabels:
            axh.annotate(cyLab, xy=(xy[cyi,plotDims[0]], xy[cyi,plotDims[1]]), **annotationParams)
        col = colors[uLabels.index(labels[cyi])]
        if weights is None:
            s = 100
        else:
            s = weights[cy] * 200 + 10
        axh.scatter(xy[cyi,plotDims[0]], xy[cyi,plotDims[1]], marker='o', s=s, alpha=alpha, c=col)
    plt.draw()

def plotModuleCorr(cyDf, labels, plotLabel, sampleStr='M', dropped=None, compCommVar=None):
    """Make a corr plot for a module."""
    modDf = makeModuleVariables(cyDf[labels.index], labels, dropped=dropped)
    modVar = '%s%s' % (sampleStr, plotLabel)
    cyVars = labels2modules(labels, dropped = None)[plotLabel]
    if not compCommVar is None:
        cyVars.append(compCommVar)
    tmpDf = cyDf[cyVars].join(modDf[modVar]).copy()

    """Rename dropped columns with an asterisk but leave them on the plot"""
    if not dropped is None:
        tmpDf.columns = np.array([c + '*' if c in dropped and dropped[c] else c for c in tmpDf.columns])

    figh = plt.gcf()
    figh.clf()
    combocorrplot(tmpDf, method = 'pearson')
    axh = plt.gca()
    axh.annotate('Module %s%s' % (sampleStr, plotLabel), xy=(0.5,0.99), xycoords='figure fraction', va = 'top', ha='center')

def plotInterModuleCorr(cyDf, labels, dropped = None, compCommVar = None):
    """Make a plot showing inter-module correlation"""
    modDf = makeModuleVariables(cyDf[labels.index], labels, dropped = dropped)
    modVars = modDf.columns.tolist()
    if not compCommVar is None:
        modDf = modDf.join(cyDf[compCommVar])
        modVars += [compCommVar]

    figh = plt.gcf()
    figh.clf()
    combocorrplot(modDf[modVars], method = 'pearson')

def cyBoxPlots(cyDf, ptidDf=None, hue=None, unLog=True):
    """Boxplots of cytokines sorted by median"""
    def sortFunc(cyDf, c):
        tmp = cyDf[c].dropna()
        if tmp.shape[0] == 0:
            return 0
        else:
            return np.median(tmp)
    sortedCy = sorted(cyDf.columns, key=partial(sortFunc,cyDf), reverse=True)
    plt.clf()
    if ptidDf is None or hue is None:
        sns.boxplot(cyDf, order=sortedCy)
    else:
        tmp = cyDf.stack().reset_index().set_index('PTID').join(ptidDf)
        if set(ptidDf[hue].unique()) == set([0,1]):
            tmp[hue] = tmp[hue].replace({1:'Yes',0:'No'})
        sns.boxplot(x='level_1', y=0, data=tmp, hue=hue, order=sortedCy)

    plt.xticks(rotation=90)
    plt.xlabel('')
    if unLog:
        plt.ylabel('Analyte concentration (pg/mL)')
        plt.yticks(np.arange(8)-1,['$10^{%d}$' % i for i in np.arange(8)-1])
    else:
        plt.ylabel('Analyte level (log-scale)')
    plt.tight_layout()

def logisticRegressionResults(df, outcome, predictors, adj=[]):
    k = len(predictors)
    assoc = np.zeros((k,6))
    params = []
    pvalues = []
    for i,predc in enumerate(predictors):
        tmp = df[[outcome, predc] + adj].dropna()
        exogVars = list(set([predc] + adj))
        model = sm.GLM(endog=tmp[outcome].astype(float), exog=sm.add_constant(tmp[exogVars].astype(float)), family=sm.families.Binomial())
        try:
            res = model.fit()
            assoc[i, 0] = np.exp(res.params[predc])
            assoc[i, 3] = res.pvalues[predc]
            assoc[i, 1:3] = np.exp(res.conf_int().loc[predc])
            params.append(res.params.to_dict())
            pvalues.append(res.pvalues.to_dict())
        except sm.tools.sm_exceptions.PerfectSeparationError:
            assoc[i, 0] = np.nan
            assoc[i, 3] = 0
            assoc[i, 1:3] = [np.nan, np.nan]
            params.append({k:np.nan for k in [predc] + adj})
            pvalues.append({k:np.nan for k in [predc] + adj})
            print 'PerfectSeparationError: %s with %s' % (predc, outcome)
    outDf = pd.DataFrame(assoc[:,:4], index=predictors, columns=['OR','LL','UL','pvalue'])
    outDf['params'] = params
    outDf['pvalues']= pvalues
    return outDf

def logisticRegressionBars(df, outcome, predictors, adj = [], useFDR = False, sigThreshold = 0.05, printPQ = False):
    """Forest plot of each predictor association with binary outcome."""
    """OR, LL, UL, p, ranksum-Z, p"""
    k = len(predictors)
    assoc = np.zeros((k,6))
    for i,predc in enumerate(predictors):
        tmp = df[[outcome, predc] + adj].dropna()
        model = sm.GLM(endog = tmp[outcome].astype(float), exog = sm.add_constant(tmp[[predc] + adj]), family = sm.families.Binomial())
        try:
            res = model.fit()
            assoc[i, 0] = np.exp(res.params[predc])
            assoc[i, 3] = res.pvalues[predc]
            assoc[i, 1:3] = np.exp(res.conf_int().loc[predc])
        except sm.tools.sm_exceptions.PerfectSeparationError:
            assoc[i, 0] = 0
            assoc[i, 3] = 0
            assoc[i, 1:3] = [0,0]
            print 'PerfectSeparationError: %s' % predc

        z, pvalue = stats.ranksums(tmp[predc].loc[tmp[outcome] == 1], tmp[predc].loc[tmp[outcome] == 0])
        assoc[i, 4] = z
        assoc[i, 5] = pvalue
        
    if useFDR:
        """Use q-value significance threshold"""
        sigInd, qvalues, _, _ = sm.stats.multipletests(assoc[:,3], alpha = sigThreshold, method = 'fdr_bh')
    else:
        """Use p-value significance threshold"""
        sigInd = assoc[:,3] < sigThreshold

    figh = plt.gcf()
    figh.clf()
    if printPQ:
        pqh = figh.add_axes([0.70, 0.1, 0.20, 0.80], frameon=False)
        axh = figh.add_axes([0.1, 0.1, 0.70, 0.80], frameon=True)
    else:
        axh = figh.add_subplot(111)
    axh.barh(bottom = np.arange(k)[~sigInd], left = assoc[~sigInd,1], width = assoc[~sigInd,2] - assoc[~sigInd,1], color = 'black', align = 'center')
    axh.barh(bottom = np.arange(k)[sigInd], left = assoc[sigInd,1], width = assoc[sigInd,2] - assoc[sigInd,1], color = 'black', align = 'center')
    axh.scatter(assoc[sigInd,0], np.arange(k)[sigInd],s = 100, color = 'red', zorder = 10)
    axh.scatter(assoc[~sigInd,0], np.arange(k)[~sigInd],s = 100, color = 'white', zorder = 10)
    axh.plot([1,1],[-1,k],'k-', lw = 2)
    plt.yticks(range(k), predictors)
    plt.grid(True, axis = 'x')
    plt.xlabel('Association with %s (odds-ratio)' % outcome)
    plt.ylim((k, -1))
    yl = plt.ylim()
    xl = plt.xlim()
    plt.xlim((0, xl[1]))
    if printPQ:
        annParams = dict(size='medium', weight='bold', color='black', ha='left', va='center')
        if useFDR:
            values = qvalues
            sigStr = 'q = %1.2g'
        else:
            values = assoc[:,3]
            sigStr = 'p = %1.2g'
        for i,v in enumerate(values):
            pqh.annotate(sigStr % v, xy=(0.1,i), **annParams)
        pqh.set_ylim(yl)
        pqh.set_xlim(-1,1)
        pqh.set_xticks(())
        pqh.set_yticks(())
    #plt.tight_layout()
    plt.show()

def plotMeanCorr(cyDf, meanVar, cyList=None, method='pearson'):
    """Plot of each cytokine's correlation with the mean."""
    corrDf = meanCorr(cyDf, meanVar, cyList, method=method)
    """Use p-value significance threshold"""
    sigInd = (corrDf.pvalue < 0.05).values
    n = corrDf.shape[0]
    plt.clf()
    plt.barh(np.arange(n)[~sigInd], corrDf.rho.loc[~sigInd], color='black', align='center')
    plt.barh(np.arange(n)[sigInd], corrDf.rho.loc[sigInd], color='red', align='center')
    plt.yticks(range(n), corrDf.index)
    plt.grid(True, axis='x')
    plt.xlabel('Correlation between\ncytokines and the "complete-common" mean ($\\rho$)')
    plt.plot([0,0],[-1,n],'k-',lw=1)
    plt.ylim((-1, n))
    plt.xlim((-1,1))
    plt.tight_layout()

def plotCrossCompartmentBars(cyDfA, cyDfB, method='pearson'):
    corrDf = crossCompartmentCorr(cyDfA, cyDfB, method=method)
    ncy = corrDf.shape[0]
    sigInd = (corrDf['pvalue'] < 0.05).values
    
    axh = plt.gca()
    axh.cla()
    axh.barh(bottom=np.arange(ncy)[~sigInd], width=corrDf.rho.loc[~sigInd], color='black', align='center')
    axh.barh(bottom=np.arange(ncy)[sigInd], width=corrDf.rho.loc[sigInd], color='red', align='center')
    plt.yticks(range(ncy), corrDf.index)
    plt.grid(True, axis='x')
    plt.xlabel('Cross compartment correlation ($\\rho$)')
    plt.ylim((-1,ncy))
    plt.xlim((-1,1))
    plt.tight_layout()

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

def plotCrossCompartmentHeatmap(cyDfA, cyDfB, n_clusters=4):
    rho,pvalue,qvalue = crosscorr(cyDfA[sorted(cyDfA.columns)], cyDfB[sorted(cyDfB.columns)])
    if n_clusters is None:
        heatmap(rho, vmin=-1, vmax=1)   
    else:
        rho_sorted = plotBicluster(rho, n_clusters=n_clusters)

def plotCrossCompartmentBoxplot(cyDfA, cyDfB):
    rho,pvalue,qvalue = crosscorr(cyDfA[sorted(cyDfA.columns)], cyDfB[sorted(cyDfB.columns)])
        
    s = [rho.loc[i,j] for i,j in itertools.product(rho.index, rho.columns) if i == j]
    d = [rho.loc[i,j] for i,j in itertools.product(rho.index, rho.columns) if i != j]
    a = pd.DataFrame({'Group':['Same']*len(s) + ['Different']*len(d), '$\\rho$':s+d})
    
    plt.clf()
    sns.boxplot(x='Group', y='$\\rho$', data=a)
    sns.stripplot(x='Group', y='$\\rho$', data=a, jitter=True)
    plt.xlabel('')
    plt.ylim((-1,1))
    plt.tight_layout()

def outcomeBoxplot(cyDf, cyVar, outcomeVar, printP=True, axh=None):
    if axh is None:
        axh = plt.gca()
    axh.cla()
    sns.boxplot(y=cyVar, x=outcomeVar, data=cyDf, ax=axh, order=[0,1])
    sns.stripplot(y=cyVar, x=outcomeVar, data=cyDf, jitter=True, ax=axh, order=[0,1])
    plt.xticks([0,1], ['False', 'True'])
    if printP:
        tmp = cyDf[[cyVar, outcomeVar]].dropna()
        z, pvalue = stats.ranksums(tmp[cyVar].loc[tmp[outcomeVar] == 1], tmp[cyVar].loc[tmp[outcomeVar] == 0])
        annParams = dict(textcoords='offset points', xytext=(0,-5), ha='center', va='top', color='black', weight='bold', size='medium')
        plt.annotate('p = %1.3g' % pvalue, xy=(0.5,plt.ylim()[1]), **annParams)
    plt.show()

def plotROC(cyDf, cyVarList, outcomeVar, n_folds=5):
    """Predict outcome with each cyVar and plot ROC for each, in a cross validation framework."""
    cyDf = cyDf.dropna()
    cv = sklearn.cross_validation.KFold(n=cyDf.shape[0], n_folds=n_folds, shuffle=True, random_state=110820)

    plt.clf()
    for cvars in cyVarList:
        cvarStr = ' + '.join(cvars)
        mean_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.zeros(mean_fpr.shape[0])
        all_tpr = []
        counter = 0
        for i, (trainInd, testInd) in enumerate(cv):
            trainDf = cyDf[[outcomeVar] + cvars].iloc[trainInd]
            testDf = cyDf[[outcomeVar] + cvars].iloc[testInd]

            model = sm.GLM(endog = trainDf[outcomeVar].astype(float), exog = sm.add_constant(trainDf[cvars]), family = sm.families.Binomial())
            try:
                outcomePred = model.fit().predict(sm.add_constant(testDf[cvars]))
                
                fpr, tpr, thresholds = sklearn.metrics.roc_curve(testDf[outcomeVar].values, outcomePred)
                mean_tpr += np.interp(mean_fpr, fpr, tpr)
                counter += 1
            except (ValueError, sm.tools.sm_exceptions.PerfectSeparationError):
                print 'PerfectSeparationError: %s, %s (skipping this train/test split)' % (cvarStr,outcomeVar)
        if counter == n_folds:
            mean_tpr /= counter
            mean_auc = sklearn.metrics.auc(mean_fpr, mean_tpr)
            mean_tpr[0], mean_tpr[-1] = 0,1
            plt.plot(mean_fpr, mean_tpr, lw=2, label='%s (AUC = %0.2f)' % (cvarStr, mean_auc))
        else:
            print 'ROC: did not finish all folds (%d of %d)' % (counter, n_folds)
            plt.plot([0, 1], [0, 1], lw=2, label='%s (AUC = %0.2f)' % (cvarStr, 0.5))


    plt.plot([0, 1], [0, 1], '--', color='gray', label='Luck')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for %s' % outcomeVar)
    plt.legend(loc="lower right")
    plt.show()

def cyNHeatmap(cyDf):
    """Heatmap showing number of data points for each potential pairwise comparison of cytokines"""
    plt.clf()
    pwCounts = validPairwiseCounts(cyDf)
    heatmap(pwCounts, cmap=cm.gray, edgecolors='w', labelSize='medium')
    plt.tight_layout()

def _colors2labels(labels, setStr = 'Set3', cmap = None):
    """Return pd.Series of colors based on labels"""
    if cmap is None:
        N = max(3,min(12,len(np.unique(labels))))
        cmap = palettable.colorbrewer.get_map(setStr,'Qualitative',N).mpl_colors
    cmapLookup = {k:col for k,col in zip(sorted(np.unique(labels)),itertools.cycle(cmap))}
    return labels.map(cmapLookup.get)

def _clean_axis(ax):
    """Remove ticks, tick labels, and frame from axis"""
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(False)
    ax.set_axis_bgcolor('white')

def plotHierClust(dmatDf, Z, labels=None, titleStr=None, vRange=None, tickSz='small', cmap=None, cmapLabel=''):
    """Display a hierarchical clustering result."""
    if vRange is None:
        vmin = np.min(np.ravel(dmatDf.values))
        vmax = np.max(np.ravel(dmatDf.values))
    else:
        vmin,vmax = vRange
    
    if cmap is None:
        if vmin < 0 and vmax > 0 and vmax <= 1 and vmin >= -1:
            cmap = cm.RdBu_r
        else:
            cmap = cm.YlOrRd

    fig = plt.gcf()
    fig.clf()

    if labels is None:
        denAX = fig.add_subplot(GridSpec(1,1,left=0.05,bottom=0.05,right=0.15,top=0.85)[0,0])
        heatmapAX = fig.add_subplot(GridSpec(1,1,left=0.16,bottom=0.05,right=0.78,top=0.85)[0,0])
        scale_cbAX = fig.add_subplot(GridSpec(1,1,left=0.87,bottom=0.05,right=0.93,top=0.85)[0,0])
    else:
        denAX = fig.add_subplot(GridSpec(1,1,left=0.05,bottom=0.05,right=0.15,top=0.85)[0,0])
        cbAX = fig.add_subplot(GridSpec(1,1,left=0.16,bottom=0.05,right=0.19,top=0.85)[0,0])
        heatmapAX = fig.add_subplot(GridSpec(1,1,left=0.2,bottom=0.05,right=0.78,top=0.85)[0,0])
        scale_cbAX = fig.add_subplot(GridSpec(1,1,left=0.87,bottom=0.05,right=0.93,top=0.85)[0,0])

    my_norm = mpl.colors.Normalize(vmin = vmin, vmax = vmax)

    """Dendrogaram along the rows"""
    plt.sca(denAX)
    denD = sch.dendrogram(Z, color_threshold=np.inf, orientation='right')
    colInd = denD['leaves']
    _clean_axis(denAX)

    if not labels is None:
        cbSE = _colors2labels(labels)
        axi = cbAX.imshow([[x] for x in cbSE.iloc[colInd].values],interpolation='nearest',aspect='auto',origin='lower')
        
        _clean_axis(cbAX)

    """Heatmap plot"""
    axi = heatmapAX.imshow(dmatDf.values[colInd,:][:,colInd],interpolation='nearest',aspect='auto',origin='lower',norm=my_norm,cmap=cmap)
    _clean_axis(heatmapAX)

    """Column tick labels along the rows"""
    if tickSz is None:
        heatmapAX.set_yticks(())
        heatmapAX.set_xticks(())
    else:
        heatmapAX.set_yticks(np.arange(dmatDf.shape[1]))
        heatmapAX.yaxis.set_ticks_position('right')
        heatmapAX.set_yticklabels(dmatDf.columns[colInd],fontsize=tickSz,fontname='Consolas')

        """Column tick labels"""
        heatmapAX.set_xticks(np.arange(dmatDf.shape[1]))
        heatmapAX.xaxis.set_ticks_position('top')
        xlabelsL = heatmapAX.set_xticklabels(dmatDf.columns[colInd],fontsize=tickSz,rotation=90,fontname='Consolas')

        """Remove the tick lines"""
        for l in heatmapAX.get_xticklines() + heatmapAX.get_yticklines(): 
            l.set_markersize(0)

    """Add a colorbar"""
    cb = fig.colorbar(axi,scale_cbAX) # note that we could pass the norm explicitly with norm=my_norm
    cb.set_label(cmapLabel)
    """Make colorbar labels smaller"""
    for t in cb.ax.yaxis.get_ticklabels():
        t.set_fontsize('small')

    """Add title as xaxis label"""
    if not titleStr is None:
        heatmapAX.set_xlabel(titleStr,size='x-large')

