

"""Create modules of co-signaling cytokines for analysis of multiplexed cytokine data (e.g. Luminex)"""

"""Common data types:
   - rawDf: pd.DataFrame with cytokines as columns and sample IDs along the index
   - cyDf: same as rawDf, but with missing values filled (if needed), log-transformed and possibly normalized
   - dmatDf: pd.DataFrame representation of pairwise distance matrix of cytokines (index and columns of cytokines)
   - pwrelDf: pd.DataFrame of pairwise cluster reliability (as a distance) from a bootstrap (index and columns of cytokines)
   - dmatFunc: function that makes a distance matrix from cyDf
   - clusterFunc: function that returns cluster labels, based on a distance matrix
   - labels: pd.Series containing cluster labels, indexed by cytokine
   - dropped: pd.Series containing an indicator about whether a cytokine is dropped from the modules for poor reliability
   - modules: dict of lists of cytokines
   - modDf: pd.DataFrame of summarized module variables
   - compComm: list of complete-common cytokines for computing the mean for each sample
"""

from . import preprocessing
from .clustering import *
from gapstat import computeGapStat
from bootstrap_cluster import bootstrapFeatures, bootstrapObservations
from . import plotting
from . import comparison

__all__ = ['preprocessing',
           'computeGapStat',
           'plotting',
           'comparison']