import networkx as nx
import seaborn as sns

from corrplots import validPairwiseCounts, partialcorr
from myboxplot import myboxplot, manyboxplots
import statsmodels.api as sm
from sklearn.decomposition import KernelPCA, PCA

import pandas as pd
import numpy as np

def cyBoxPlots(plotDf, sortDf, cyDict, cySets, vRange, fn):
    """Boxplots of all cytokines, organized by source and sorted by median"""
    def sortFunc(df,c):
        tmp = df[c].dropna()
        if tmp.shape[0] == 0:
            return 0
        else:
            return median(tmp)
    figure(1,figsize=(17,9))
    axh = subplot(111)
    k = 17
    for cySet in cySets:
        sortedCy = sorted(cyDict[cySet], key = partial(sortFunc,sortDf), reverse=True)
        n = len(sortedCy)
        for i in arange(ceil(n/k)):
            clf()
            axh = subplot2grid((5,1), (0,0), rowspan=4)
            #axh.set_yscale('log')
            #sns.violinplot(plotDf, order = sortedCy[int(i*k) : int(i*k+k)], ax = axh, alpha = 0.7, inner = 'points')
            manyboxplots(plotDf, cols = sortedCy[int(i*k) : int(i*k+k)], axh = axh, alpha=0.7, vRange=vRange, xRot=90, violin=False)
            ylabel('Concentration (pg/mL)')
            title(cySet + ' Analytes (page %d)' % (i+1))
            figure(1).savefig(DATA_PATH + 'RandolphFlu/figures/%s_%s_%02d.png' % (fn,cySet,i))

def cyNHeatmap(df, cyDict, cySets, studyStr):
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
def compCommCorr(df, cyDict, tiss, compCommVec, studyStr):
    """Plot of each cytokine's correlation with the mean."""
    tmpCorr = zeros((len(cyDict[tiss]),2))
    for i,s in enumerate(cyDict[tiss]):
        tmpCorr[i,0], tmpCorr[i,1] = partialcorr(df[s], compCommVec, method = 'spearman')
    sorti = argsort(tmpCorr[:,0])
    tmpCorr = tmpCorr[sorti,:]

    """Use q-value significance threshold"""
    sigInd, qvalues, _, _ = sm.stats.multipletests(tmpCorr[:,1], alpha = 0.2, method = 'fdr_bh')
    """Use p-value significance threshold"""
    sigInd = tmpCorr[:,1] < 0.05

    figure(211, figsize = (10,11.8))
    clf()
    barh(arange(tmpCorr.shape[0])[~sigInd], tmpCorr[~sigInd,0]**2, color = 'black', align='center')
    barh(arange(tmpCorr.shape[0])[sigInd], tmpCorr[sigInd,0]**2, color = 'red', align='center')
    yticks(range(tmpCorr.shape[0]), array([cy.split(' ')[0] for cy in cyDict[tiss]])[sorti])
    grid(True, axis = 'x')
    xlabel('Correlation between\n%s cytokines and the "complete-common" mean ($^*R^2$)' % (tiss))
    ylim((-1,tmpCorr.shape[0]))
    xlim((0,1))
    tight_layout()
    figure(211).savefig(DATA_PATH + 'RandolphFlu/figures/%s_%s_mean_corr.png' % (studyStr,tiss))

def corrPlotsByCluster(df, clusters, studyStr):
    """Make a corr plot for each cluster"""
    modDf = makeModuleVariables(df, clusters)
    figure(642, figsize=(23,11))
    for tiss in clusters.keys():
        for c in clusters[tiss].keys():
            modVar = 'Module%d %s' % (c,tiss)
            clf()
            combocorrplot(df[clusters[tiss][c]].join(modDf[modVar]), method = 'spearman')
            annotate('%s: Cluster %s' % (tiss,c),xy=(0.5,0.99), xycoords='figure fraction', va = 'top', ha='center')
            figure(642).savefig(DATA_PATH + 'RandolphFlu/figures/%s_%s_cluster_%s.png' % (studyStr,tiss,c))

def plotModuleEmbedding(df, tiss, modules, dmatFunc, studyStr, plotLabels = True):
    """Embed cytokine correlation matrix to visualize cytokine clusters"""
    clustLookup = {cy:n for n,v in modules[tiss].items() for cy in v}
    allCy = clustLookup.keys()
    moduleNames = modules[tiss].keys()

    dmat = dmatFunc(df[allCy])

    """By using KernelPCA for dimensionality reduction we don't need to impute missing values"""
    kpca = KernelPCA(kernel='precomputed')
    gram = 1 - (dmat / dmat.max())
    xy = kpca.fit_transform(gram)

    colors = palettable.colorbrewer.get_map('Set1', 'qualitative', len(modules[tiss])).mpl_colors
    figh = figure(10)
    clf()
    axh = figh.add_axes([0.03,0.03,0.94,0.94])
    axh.axis('off')
    figh.set_facecolor('white')
    for cyi,cy in enumerate(allCy):
        if plotLabels:
            annotate(' '.join(cy.split(' ')[:-1]), xy=(xy[cyi,0],xy[cyi,1]), xytext=(0,5), textcoords='offset points', size='x-small')
        col = colors[moduleNames.index(clustLookup[cy])]
        #col = ['gray']*50
        scatter(xy[cyi,0],xy[cyi,1],marker='o', s = 100, alpha = 0.8, c = col)
    annotate('%s %s' % (studyStr,tiss),xy=(0.5,0.99),xycoords = 'figure fraction', ha='center',va='top',size='x-large')
    figure(10).savefig(DATA_PATH + 'RandolphFlu/figures/%s_%s_module_embed_kpca.png' % (studyStr,tiss))

def plotEmbedding(df, cyVars, tiss, dmatFunc, studyStr, plotLabels = True):
    """Embed cytokine correlation matrix to visualize cytokine clusters"""
    dmat = dmatFunc(df[cyVars[tiss]])

    """By using KernelPCA for dimensionality reduction we don't need to impute missing values"""
    kpca = KernelPCA(kernel='precomputed')
    gram = 1 - (dmat / dmat.max())
    xy = kpca.fit_transform(gram)

    figh = figure(10)
    clf()
    axh = figh.add_axes([0.03,0.03,0.94,0.94])
    axh.axis('off')
    figh.set_facecolor('white')
    scatter(xy[:,0],xy[:,1], s=100, alpha = 0.8, c = 'gray')
    if plotLabels:
        for cyi,cy in enumerate(cyVars[tiss]):
            annotate(' '.join(cy.split(' ')[:-1]), xy=(xy[cyi,0],xy[cyi,1]), xytext=(0,5), textcoords='offset points', size='x-small')
    annotate('%s %s' % (studyStr,tiss),xy=(0.5,0.99),xycoords = 'figure fraction', ha='center',va='top',size='x-large')
    figure(10).savefig(DATA_PATH + 'RandolphFlu/figures/%s_%s_module_embed_kpca.png' % (studyStr,tiss))


def plotClusterEmbedding(df, labels, plotLabels = True, dmatFunc = None):
    """Embed cytokine correlation matrix to visualize cytokine clusters"""
    uLabels = list(unique(labels))

    if not dmatFunc is None:
        """By using KernelPCA for dimensionality reduction we don't need to impute missing values"""
        dmat = dmatFunc(df)
        pca = KernelPCA(kernel='precomputed')
        gram = 1 - (dmat / dmat.max())
        xy = pca.fit_transform(gram)
    else:
        pca = PCA(n_components = 2)
        xy = pca.fit_transform(df.T)


    colors = palettable.colorbrewer.get_map('Set1', 'qualitative', len(uLabels)).mpl_colors
    figh = gcf()
    clf()
    axh = figh.add_axes([0.03,0.03,0.94,0.94])
    axh.axis('off')
    figh.set_facecolor('white')
    annotationParams = dict(xytext=(0,5), textcoords='offset points', size='x-small')
    for cyi,cy in enumerate(df.columns):
        if plotLabels:
            annotate(' '.join(cy.split(' ')[:-1]), xy=(xy[cyi,0],xy[cyi,1]), **annotationParams)
        col = colors[uLabels.index(labels[cyi])]
        scatter(xy[cyi,0], xy[cyi,1], marker = 'o', s = 100, alpha = 0.8, c = col)
    return pca

def plotClusterNetwork(df, labels):
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