def crossCompartmentCorr(df, cyDict, cySets, studyStr, useFDR = False):
    """Plot of cytokine correlation for those that are common to both serum and ETT"""
    tmpCorr = zeros((len(cyDict[cySets[0]]),2))
    for i,(s,e) in enumerate(zip(cyDict[cySets[0]],cyDict[cySets[1]])):
        tmpCorr[i,0], tmpCorr[i,1] = partialcorr(df[s], df[e], method = 'spearman')
        #print s,e,tmpCorr[i,:]
        #print 'scatterfit(log(ndf["%s"]),log(ndf["%s"]),method="spearman")' % (s,e)
    sorti = argsort(tmpCorr[:,0])
    tmpCorr = tmpCorr[sorti,:]

    if useFDR:
        """Use q-value significance threshold"""
        sigInd, qvalues, _, _ = sm.stats.multipletests(tmpCorr[:,1], alpha = 0.2, method = 'fdr_bh')
    else:
        """Use p-value significance threshold"""
        sigInd = tmpCorr[:,1] < 0.05

    figure(111, figsize = (10,11.8))
    clf()
    barh(arange(tmpCorr.shape[0])[~sigInd], tmpCorr[~sigInd,0]**2, color = 'black', align='center')
    barh(arange(tmpCorr.shape[0])[sigInd], tmpCorr[sigInd,0]**2, color = 'red', align='center')
    yticks(range(tmpCorr.shape[0]), array([' '.join(cy.split(' ')[:-1]) for cy in cyDict[cySets[0]]])[sorti])
    grid(True, axis = 'x')
    xlabel('Correlation between\n%s and %s ($^*R^2$)' % (cySets[0],cySets[1]))
    ylim((-1,tmpCorr.shape[0]))
    xlim((0,1))
    tight_layout()
    figure(111).savefig(DATA_PATH + 'RandolphFlu/figures/%s_paired_corr.png' % studyStr)

def computeClusterOverlap(labelsA, labelsB):
    uA = unique(labelsA)
    uB = unique(labelsB)
    s = zeros((len(uA),len(uB)))
    for ai,a in enumerate(uA):
        for bi,b in enumerate(uB):
            indA = labelsA == a
            indB = labelsB == b
            s[ai,bi] = 2 * (indA & indB).sum()/(indA.sum() + indB.sum())
    sA = sorted(uA, key = lambda a: (labelsA == a).sum(), reverse = True)
    #sB = sorted(uB)
    sB = sorted(uB, key = lambda b: find(sA == uA[argmax(s[:,find(uB==b)])])[0])
    for ai,a in enumerate(sA):
        for bi,b in enumerate(sB):
            indA = labelsA == a
            indB = labelsB == b
            s[ai,bi] = 2 * (indA & indB).sum()/(indA.sum() + indB.sum())
    return sA, sB, s

def plotClusterOverlap(sA, sB, s):
    yA = linspace(10,0,len(sA))
    yB = linspace(10,0,len(sB))
    for ai, a in enumerate(sA):
        annotate(s = '%s' % a, xy = (0,yA[ai]),ha = 'center', va = 'center', size = 'x-large', zorder = 15)
        for bi, b in enumerate(sB):
            if ai == 0:
                annotate(s = '%s' % b, xy = (1,yB[bi]),ha = 'center', va = 'center', size = 'x-large', zorder = 15, color='white')
            plot([0,1],[yA[ai], yB[bi]], '-', lw = 20 * s[ai,bi], color='black', alpha = 0.7, zorder = 1)
    scatter(zeros(len(sA)), yA, s = 1000, color = 'red', zorder = 10)
    scatter(ones(len(sB)), yB, s = 1000, color = 'blue', zorder = 10)
    axis('off')

def alignClusters(predA,predB):
    """Returns a copy of predB with columns shuffled to maximize overlap with predA"""
    nCols = predA.shape[1]
    swap = {}
    for colA in range(nCols):
        match = argmax([(predA[:,colA] * predB[:,colB]).sum() for colB in range(nCols)])
        swap.update({match:colA})

    out = predB.copy()
    for colB,colA in swap.items():
        out[:,colA] = predB[:,colB]
    
    """In case the clusters aren't clearly 1:1 then try extra swaps until the optimum is found"""
    niters = 0
    while True:
        swaps = []
        curProd = (predA * out).sum()
        for ai,bi in itertools.product(range(nCols),range(nCols)):
            ind = arange(nCols)
            ind[ai] = bi
            ind[bi] = ai
            newProd = (predA * out[:,ind]).sum()
            if curProd < newProd:
                swaps.append((ai,bi,newProd))
        if len(swaps) == 0:
            break
        else:
            ai,bi,newProd = swaps[argmax([x[2] for x in swaps])]
            ind = arange(nCols)
            ind[ai] = bi
            ind[bi] = ai
            out = out[:,ind]
    return out