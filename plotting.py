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

from corrplots import validPairwiseCounts, partialcorr,combocorrplot
import statsmodels.api as sm
from scipy import stats

import sklearn

import seaborn as sns
sns.set(style = 'darkgrid', palette = 'muted', font_scale = 1.75)

from cycluster import labels2modules, makeModuleVariables

__all__ = ['plotModuleEmbedding',
           'plotModuleCorr',
           'plotInterModuleCorr',
           'cyBoxPlots',
           'logisticRegressionBars',
           'plotMeanCorr',
           'outcomeBoxplot',
           'plotROC',
           'plotInterModuleCorr']

def plotModuleEmbedding(dmatDf, labels, dropped=None, method='kpca', plotLabels=True, plotDims=[0,1]):
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
    annotationParams = dict(xytext=(0,5), textcoords='offset points', size='x-small')
    for cyi,cy in enumerate(dmatDf.columns):
        if not dropped is None and dropped[cy]:
            cyLab = '*' + cy
            alpha = 0.3
        else:
            cyLab = cy
            alpha = 0.8

        if plotLabels:
            axh.annotate(cyLab, xy = (xy[cyi,plotDims[0]], xy[cyi,plotDims[1]]), **annotationParams)
        col = colors[uLabels.index(labels[cyi])]
        axh.scatter(xy[cyi,plotDims[0]], xy[cyi,plotDims[1]], marker='o', s=100, alpha=alpha, c=col)
    plt.draw()

def plotModuleCorr(cyDf, labels, plotLabel, dropped = None, compCommVar = None):
    """Make a corr plot for a module."""
    modDf = makeModuleVariables(cyDf[labels.index], labels, dropped = dropped)
    modVar = 'M%s' % plotLabel
    cyVars = labels2modules(labels, dropped = None)[plotLabel]
    if not compCommVar is None:
        cyVars.append(compCommVar)
    tmpDf = cyDf[cyVars].join(modDf[modVar]).copy()

    """Rename dropped columns with an asterisk but leave them on the plot"""
    tmpDf.columns = np.array([c + '*' if c in dropped and dropped[c] else c for c in tmpDf.columns])

    figh = plt.gcf()
    figh.clf()
    combocorrplot(tmpDf, method = 'pearson')
    axh = plt.gca()
    axh.annotate('Module M%s' % (plotLabel), xy=(0.5,0.99), xycoords='figure fraction', va = 'top', ha='center')

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

def cyBoxPlots(cyDf, basefile, vRange=None,):
    """Boxplots of cytokines sorted by median"""
    def sortFunc(df, c):
        tmp = df[c].dropna()
        if tmp.shape[0] == 0:
            return 0
        else:
            return np.median(tmp)
    
    figh = plt.gcf()
    k = 17
    sortedCy = sorted(cyDf.columns, key = partial(sortFunc,cyDf), reverse = True)
    n = len(sortedCy)
    for i in np.arange(np.ceil(n/k)):
        figh.clf()
        axh = plt.subplot2grid((5,1), (0,0), rowspan = 4)
        #axh.set_yscale('log')
        #sns.violinplot(plotDf, order = sortedCy[int(i*k) : int(i*k+k)], ax = axh, alpha = 0.7, inner = 'points')
        manyboxplots(cyDf, cols = sortedCy[int(i*k) : int(i*k+k)], axh = axh, alpha = 0.7, vRange = vRange, xRot = 90, violin = False)
        plt.ylabel('Concentration (log-pg/mL)')
        plt.title('Cytokines (page %d)' % (i+1))
        figh.savefig('%s_%02d.png' % (basefile,i))

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

def plotMeanCorr(df, meanVar):
    """Plot of each cytokine's correlation with the mean."""
    cyList = [c for c in df.columns if not c == meanVar]

    tmpCorr = np.zeros((len(cyList),2))
    for i,s in enumerate(cyList):
        tmpCorr[i,0], tmpCorr[i,1] = partialcorr(df[s], df[meanVar], method = 'pearson')
    sorti = np.argsort(tmpCorr[:,0])
    tmpCorr = tmpCorr[sorti,:]

    """Use q-value significance threshold"""
    sigInd, qvalues, _, _ = sm.stats.multipletests(tmpCorr[:,1], alpha = 0.2, method = 'fdr_bh')
    """Use p-value significance threshold"""
    sigInd = tmpCorr[:,1] < 0.05

    plt.clf()
    plt.barh(np.arange(tmpCorr.shape[0])[~sigInd], tmpCorr[~sigInd,0]**2, color = 'black', align='center')
    plt.barh(np.arange(tmpCorr.shape[0])[sigInd], tmpCorr[sigInd,0]**2, color = 'red', align='center')
    plt.yticks(range(tmpCorr.shape[0]), np.array(cyList)[sorti])
    plt.grid(True, axis = 'x')
    plt.xlabel('Correlation between\ncytokines and the "complete-common" mean ($^*R^2$)')
    plt.ylim((-1, tmpCorr.shape[0]))
    plt.xlim((0,1))
    plt.tight_layout()

def outcomeBoxplot(cyDf, cyVar, outcomeVar, printP=True):
    figh = plt.gcf()
    plt.clf()
    axh = plt.subplot(111)
    sns.boxplot(y = cyVar, x = outcomeVar, data = cyDf, ax = axh, order  = [0,1])
    sns.stripplot(y = cyVar, x = outcomeVar, data = cyDf, jitter = True, ax = axh, order  = [0,1])
    plt.xticks([0,1], ['False', 'True'])
    if printP:
        tmp = cyDf[[cyVar, outcomeVar]].dropna()
        z, pvalue = stats.ranksums(tmp[cyVar].loc[tmp[outcomeVar] == 1], tmp[cyVar].loc[tmp[outcomeVar] == 0])
        annParams = dict(textcoords='offset points', xytext=(0,-5), ha='center', va='top', color='black', weight='bold', size='medium')
        plt.annotate('p = %1.3g' % pvalue, xy=(0.5,plt.ylim()[1]), **annParams)
    plt.show()

def plotROC(cyDf, cyVars, outcomeVar, n_folds=5):
    """Predict outcome with each cyVar and plot ROC for each, in a cross validation framework."""
    cyDf = cyDf.dropna()
    cv = sklearn.cross_validation.KFold(n=cyDf.shape[0], n_folds=n_folds, shuffle=True, random_state=110820)

    plt.clf()
    for cvar in cyVars:
        mean_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.zeros(mean_fpr.shape[0])
        all_tpr = []
        counter = 0
        for i, (trainInd, testInd) in enumerate(cv):
            trainDf = cyDf[[outcomeVar, cvar]].iloc[trainInd]
            testDf = cyDf[[outcomeVar, cvar]].iloc[testInd]
            model = sm.GLM(endog = trainDf[outcomeVar].astype(float), exog = sm.add_constant(trainDf[cvar]), family = sm.families.Binomial())
            try:
                outcomePred = model.fit().predict(sm.add_constant(testDf[cvar]))
                
                fpr, tpr, thresholds = sklearn.metrics.roc_curve(testDf[outcomeVar].values, outcomePred)
                mean_tpr += np.interp(mean_fpr, fpr, tpr)
                counter += 1
            except (ValueError, sm.tools.sm_exceptions.PerfectSeparationError):
                print 'PerfectSeparationError: %s, %s (skipping this train/test split)' % (cvar,outcomeVar)
        if counter == n_folds:
            mean_tpr /= counter
            mean_auc = sklearn.metrics.auc(mean_fpr, mean_tpr)
            mean_tpr[0], mean_tpr[-1] = 0,1
            plt.plot(mean_fpr, mean_tpr, lw=2, label='%s (AUC = %0.2f)' % (cvar, mean_auc))
        else:
            print 'ROC: did not finish all folds (%d of %d)' % (counter, n_folds)
            plt.plot([0, 1], [0, 1], lw=2, label='%s (AUC = %0.2f)' % (cvar, 0.5))


    plt.plot([0, 1], [0, 1], '--', color='gray', label='Luck')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for %s' % outcomeVar)
    plt.legend(loc="lower right")
    plt.show()

def _cyNHeatmap(df, cyDict, cySets, studyStr):
    """Heatmap showing number of data points for each potential pairwise comparison of cytokines"""
    figure(2,figsize=(15,11.8))
    for cySet in cySets:
        pwCounts = validPairwiseCounts(df,cyDict[cySet])
        heatmap(pwCounts, cmap = cm.gray, edgecolors = 'w', labelSize = 'medium')
        tight_layout()
        figure(2).savefig(DATA_PATH + 'RandolphFlu/figures/%s_num_cy_%s.png' % (studyStr,cySet))

    figure(3,figsize=(22,10))
    pwCounts = validPairwiseCounts(df,cyDict['All'])
    heatmap(pwCounts, cmap = cm.gray, edgecolors='w', labelSize='small')
    tight_layout()
    figure(3).savefig(DATA_PATH + 'RandolphFlu/figures/%s_num_cy_All.png' % studyStr)

def _plotClusterNetwork(df, labels):
    """WORK IN PROGRESS"""
    metric = 'pearson-signed'
    minPatients = 30
    K = 6
    tiss = 'ETT'
    dmatFunc = partial(computeDMat, metric = metric, minN = minPatients)
    dmat = dmatFunc(fillMissing(ndf[cyVars[tiss]], cyVars[tiss]))
    dt = [('rho', float)]
    A = np.matrix([[(dmat[i,j],) for j in range(dmat.shape[1])] for i in range(dmat.shape[0])], dtype = dt)
    g = nx.from_numpy_matrix(A)
    g = nx.relabel_nodes(g, {i:col.split(' ')[0] for i,col in enumerate(cyVars[tiss])})

    drawParams = dict(font_size = 9,
                      font_weight = 'bold',
                      with_labels = True,
                      node_size = 500)

    """This is a threshld on DISTANCE"""
    threshold = 0.2
    edgelist = [(u,v) for u,v,data in g.edges_iter(data = True) if data['rho'] < threshold]

    figure(2)
    clf()
    pos = nx.graphviz_layout(g, prog = 'neato')
    nx.draw(g, pos = pos, edgelist = edgelist, **drawParams)


    g = nx.Graph()
    """Add a node for each unique value in each column with name: col_value"""
    for col in df.columns:
        for val in df[col].unique():
            freq = (df[col]==val).sum()/df.shape[0]
            g.add_node((col,val),freq=freq)
    """Add edges for each unique pair of values
    with edgewidth proportional to frequency of pairing"""
    for col1,col2 in itertools.combinations(df.columns,2):
        for val1,val2 in itertools.product(df[col1].unique(),df[col2].unique()):
            w = ((df[col1]==val1) & (df[col2]==val2)).sum()
            if w>0:
                dat = dict(weight = w/df.shape[0])
                dat['pvalue'] = pvalueArr[edgeKeys.index(((col1,val1),(col2,val2)))]
                dat['qvalue'] = qvalueArr[edgeKeys.index(((col1,val1),(col2,val2)))]
                g.add_edge((col1,val1),(col2,val2),**dat)


    """Compute attributes of edges and nodes"""
    edgewidth = array([d['weight'] for n1,n2,d in g.edges(data=True)])
    nodesize = array([d['freq'] for n,d in g.nodes(data=True)])

    nColors = min(max(len(df.columns),3),9)
    colors = brewer2mpl.get_map('Set1','Qualitative',nColors).mpl_colors
    cmap = {c:color for c,color in zip(df.columns, itertools.cycle(colors))}
    nodecolors = [cmap[n[0]] for n in g.nodes()]
    if layout == 'twopi':
        """If using this layout specify the most common node as the root"""
        freq = {n:d['freq'] for n,d in g.nodes(data=True)}
        pos = nx.graphviz_layout(g,prog=layout, root=max(freq.keys(),key=freq.get))
    else:
        pos = nx.graphviz_layout(g,prog=layout)

    """Use either matplotlib or plot.ly to plot the network"""
    if mode == 'mpl':
        clf()
        figh=gcf()
        axh=figh.add_axes([0.04,0.04,0.92,0.92])
        axh.axis('off')
        figh.set_facecolor('white')

        #nx.draw_networkx_edges(g,pos,alpha=0.5,width=sznorm(edgewidth,mn=0.5,mx=10), edge_color='k')
        #nx.draw_networkx_nodes(g,pos,node_size=sznorm(nodesize,mn=500,mx=5000),node_color=nodecolors,alpha=1)
        ew = szscale(edgewidth,mn=wRange[0],mx=wRange[1])

        for es,e in zip(ew,g.edges_iter()):
            x1,y1=pos[e[0]]
            x2,y2=pos[e[1]]
            props = dict(color='black',alpha=0.4,zorder=1)
            if testSig and g[e[0]][e[1]]['qvalue'] < testSig:
                props['color']='orange'
                props['alpha']=0.8
            plot([x1,x2],[y1,y2],'-',lw=es,**props)

        scatter(x=[pos[s][0] for s in g.nodes()],
                y=[pos[s][1] for s in g.nodes()],
                s=szscale(nodesize,mn=sRange[0],mx=sRange[1]), #Units for scatter is (size in points)**2
                c=nodecolors,
                alpha=1,zorder=2)
        for n in g.nodes():
            annotate(n[1],
                    xy=pos[n],
                    fontname='Consolas',
                    size='medium',
                    weight='bold',
                    color='black',
                    va='center',
                    ha='center')
        colorLegend(labels=df.columns,colors = [c for x,c in zip(df.columns,colors)],loc=0)
        title(titleStr)

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