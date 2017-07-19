import pandas as pd
import cycluster as cy
import os.path as op
import numpy as np
import palettable
from custom_legends import colorLegend
import seaborn as sns
from hclusterplot import *

sns.set_context('paper')

dataFilename = op.join(DATA_PATH, '170615_LEGENDplex_ADAMTS4_DB.csv')

"""A long df has one analyte measurement per row"""
longDf = pd.read_csv(dataFilename)

longDf.loc[:,'ptid'] = ['%s-%d-%d' % c for c in zip(longDf.genotype, longDf['sample'], longDf['dpi'])]

"""Print table of sample count"""
print(longDf.loc[longDf.cytokine=='mcp1'].groupby(['genotype', 'dpi'])['ptid'].count())

"""Identify primary day for clustering"""
df = longDf.set_index(['ptid', 'dpi','cytokine'])['log10_conc'].unstack(['cytokine','dpi'])
#plt.plot([0, 3, 6, 9, 12], df['ifng'].values.T, '-o')

"""A wide df has one sample per row (analyte measurements across the columns)"""
# dayDf = longDf.loc[longDf.dpi == 9]
dayDf = longDf.loc[longDf.dpi.isin([3, 6, 9])]

tmp = dayDf.pivot_table(index='ptid', columns='cytokine', values='log10_conc')
noVar = tmp.columns[np.isclose(tmp.std(), 0)].tolist()
naCols = tmp.columns[(~tmp.isnull()).sum() < 5].tolist() + ['il21', 'il9']
keepCols = [c for c in tmp.columns if not c in (noVar + naCols)]

def _prepCyDf(dayDf, keepCols, K=3, normed=False):
    dayDf = dayDf.pivot_table(index='ptid', columns='cytokine', values='log10_conc')[keepCols]
    """By setting normed=True the data our normalized based on correlation with mean analyte concentration"""
    rcyc = cy.cyclusterClass(studyStr='ADAMTS', sampleStr='LUNG', normed=normed, rCyDf=dayDf)
    rcyc.clusterCytokines(K=K, metric='spearman-signed', minN=0)
    rcyc.printModules()
    return rcyc

rcyc = _prepCyDf(dayDf, keepCols, normed=True)
wt = _prepCyDf(dayDf.loc[dayDf.genotype == 'WT'], keepCols, normed=True)
ko = _prepCyDf(dayDf.loc[dayDf.genotype == 'KO'], keepCols, normed=True)

"""Now you can use attributes in nserum for plots and testing: cyDf, modDf, dmatDf, etc."""
plt.figure(41, figsize=(15.5, 9.5))
colInds = plotHColCluster(rcyc.cyDf,
                          method='complete',
                          metric='pearson-signed',
                          col_labels=rcyc.labels,
                          col_dmat=rcyc.dmatDf,
                          tickSz='large',
                          vRange=(0,1))

plt.figure(43, figsize = (15.5, 9.5))
colInds = cy.plotting.plotHierClust(1 - rcyc.pwrel,
                               rcyc.Z,
                               labels=rcyc.labels,
                               titleStr='Pairwise reliability (%s)' % rcyc.name,
                               vRange=(0, 1),
                               tickSz='large')

plt.figure(901, figsize=(13, 9.7))
cy.plotting.plotModuleEmbedding(rcyc.dmatDf, rcyc.labels, method='kpca', txtSize='large')
colors = palettable.colorbrewer.get_map('Set1', 'qualitative', len(np.unique(rcyc.labels))).mpl_colors
colorLegend(colors, ['%s%1.0f' % (rcyc.sampleStr, i) for i in np.unique(rcyc.labels)], loc='lower left')


"""df here should have one column per module and the genotype column"""
ptidDf = longDf[['ptid', 'sample', 'genotype', 'dpi']].drop_duplicates().set_index('ptid')
df = rcyc.modDf.join(ptidDf)

ind = df.genotype == 'WT'
col = 'SERUM1'
stats.ranksums(df[col].loc[ind], df[col].loc[~ind])

