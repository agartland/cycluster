import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import palettable
import tsne
from sklearn.decomposition import KernelPCA, PCA
from sklearn.manifold import TSNE
import tsne
from functools import partial
from myboxplot import manyboxplots
from corrplots import combocorrplot

from corrplots import validPairwiseCounts, partialcorr
import statsmodels.api as sm
from scipy import stats

import seaborn as sns
sns.set(style = 'darkgrid', palette = 'muted', font_scale = 1.75)

from . import labels2modules, makeModuleVariables

__all__ = ['plotModuleEmbedding',
           'plotModuleCorr',
           'cytokineBoxPlots',
           'logisticRegressionBars',
           'plotMeanCorr',
           'outcomeBoxplots']

def plotModuleEmbedding(dmatDf, labels, dropped = None, method = 'tsne', plotLabels = True):
    """Embed cytokine correlation matrix to visualize cytokine clusters"""
    uLabels = np.unique(labels).tolist()

    dmat = dmatDf.values
    
    if method == 'kpca':
        """By using KernelPCA for dimensionality reduction we don't need to impute missing values"""
        pca = KernelPCA(kernel='precomputed')
        gram = 1 - (dmat / dmat.max())
        xy = pca.fit_transform(gram)
    elif method == 'tsne':
        xy = tsne.run_tsne(dmat)
    elif method == 'sklearn-tsne':
        tsneObj = TSNE(n_components=2, metric='precomputed', random_state = 0)
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
            axh.annotate(cyLab, xy = (xy[cyi,0], xy[cyi,1]), **annotationParams)
        col = colors[uLabels.index(labels[cyi])]
        axh.scatter(xy[cyi,0], xy[cyi,1], marker = 'o', s = 100, alpha = alpha, c = col)
    plt.draw()

def plotModuleCorr(cyDf, labels, plotLabel, dropped = None, compCommVar = None):
    """Make a corr plot for a module."""
    modDf = makeModuleVariables(cyDf[labels.index], labels, dropped = dropped)
    modVar = 'M%s' % plotLabel
    cyVars = labels2modules(labels, dropped = dropped)[plotLabel]
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

def cyBoxPlots(cyDf, vRange, basefile):
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

def logisticRegressionBars(df, outcome, predictors, adj = [], useFDR = False, sigThreshold = 0.05):
    """Forest plot of each predictor association with binary outcome."""
    """OR, LL, UL, p, ranksum-Z, p"""
    k = len(predictors)
    assoc = np.zeros((k,6))
    for i,predc in enumerate(predictors):
        tmp = df[[outcome, predc]].dropna()
        model = sm.GLM(endog = df[outcome].astype(float), exog = sm.add_constant(df[[predc] + adj]), missing = 'drop', family = sm.families.Binomial())
        res = model.fit()
        assoc[i, 0] = np.exp(res.params[predc])
        assoc[i, 3] = res.pvalues[predc]
        assoc[i, 1:3] = np.exp(res.conf_int().loc[predc])
        z, pvalue = stats.ranksums(tmp[predc].loc[tmp[outcome] == 1], tmp[predc].loc[df[outcome] == 0].dropna())
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
    xl = plt.xlim()
    plt.xlim((0.05, xl[1]))
    plt.tight_layout()
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

def outcomeBoxplots(cyDf, cyVar, outcomeVar):
    figh = plt.gcf()
    plt.clf()
    axh = plt.subplot(111)
    sns.boxplot(y = cyVar, x = outcomeVar, data = cyDf, ax = axh, order  = [0,1])
    sns.stripplot(y = cyVar, x = outcomeVar, data = cyDf, jitter = True, ax = axh, order  = [0,1])
    plt.xticks([0,1], ['False', 'True'])
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