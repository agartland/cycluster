import pandas as pd
from os.path import join as opj

__all__ = ['writeModules']

def writeModules(cy, folder):
    """rawDf: pd.DataFrame with cytokines as columns and sample IDs along the index"""
    cy.rCyDf.to_csv(opj(folder, '{name}_raw_log-conc.csv'.format(name=cy.name)))

    """cyDf: same as rawDf, but with missing values filled (if needed), log-transformed and possibly normalized"""
    cy.cyDf.to_csv(opj(folder, '{name}_normalized_conc.csv'.format(name=cy.name)))

    """dmatDf: pd.DataFrame representation of pairwise distance matrix of cytokines (index and columns of cytokines)"""
    cy.dmatDf.to_csv(opj(folder, '{name}_dmat.csv'.format(name=cy.name)))
    
    """pwrelDf: pd.DataFrame of pairwise cluster reliability (as a distance) from a bootstrap (index and columns of cytokines)"""
    cy.pwrel.to_csv(opj(folder, '{name}_pwrel_dmat.csv'.format(name=cy.name)))

    """labels: pd.Series containing cluster labels, indexed by cytokine"""
    cy.labels.to_csv(opj(folder, '{name}_cluster_labels.csv'.format(name=cy.name)))

    """modDf: pd.DataFrame of summarized module variables"""
    cy.modDf.to_csv(opj(folder, '{name}_normalized_modules.csv'.format(name=cy.name)))

    cy.rModDf.to_csv(opj(folder, '{name}_raw_modules.csv'.format(name=cy.name)))
