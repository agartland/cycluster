import pandas as pd
import numpy as np
import statsmodels.api as sm
from functools import partial
import sklearn

__all__ = ['transformCytokines',
           'enforceSensitivity',
           'normalizeLevels',
           'meanSubNormalize',
           'partialCorrNormalize',
           'convertLevel',
           'standardizedMean',
           'imputeNA']

def meanSubNormalize(cyDf, cyVars=None, compCommVars=None, meanVar=None):
    """Normalize cytokine columns by the log-mean for each patient, within each compartment.
    The point is that if cytokine concentrations are generally high for one sample or another,
    this might dominate the covariation of cytokines across patients (both within/across compartments).

    We subtract off the mean since the "overall inflamation" level
    that we are adjusting for would probably be on the fold-change concentration scale.
    (additive on the log-concentration scale)"""
    def _normFuncSub(vec):
        out = vec - muVec
        return out

    if cyVars is None:
        cyVars = cyDf.columns
    if meanVar is None:
        meanVar = 'Mean'
    if compCommVars is None:
        cyDf.columns

    """No standardizing cytokines before taking the mean (need units to stay in log-concentration)"""
    muVec = cyDf[compCommVars].mean(axis=1)
    
    ndf = cyDf.copy()
    ndf.loc[:, cyVars] = ndf[cyVars].apply(_normFuncSub, axis = 1)
    ndf.loc[:, meanVar] = muVec
    return ndf

def partialCorrNormalize(cyDf, cyVars=None, compCommVars=None, meanVar=None, returnModels=True):
    """Computes residuals in-place after regressing each cytokine on the mean cytokine level
    Correlations among residuals are partial correlations, adjusting for meanVar

    Parameters
    ----------
    cyDf : pd.DataFrame
        Log-transformed cytokine data with cytokine columns and rows per patient/timepoint
    cyVars : list
        Cytokine columns in cyDf that will be normalized and included in the returned df
        (default: all columns in cyDf)
    compCommVars : list
        Cytokine columns used for computing the mean level for each row.
        (default: all columns in cyDf)
    meanVar : str
        Name of the cytokine mean column added to the df
        (default: "Mean")
    returnModels : bool
        If True, return the fitted statsmodels.GLM objects for transforming additional timepoints.

    Returns
    -------
    nCyDf : pd.DataFrame
        Residuals after regressing each cyVar on the mean cytokine level for each row.
    models : pd.Series
        If returnModels is True,
        result object from the regression for each cytokine (index), that can be used
        to normalize additional timepoints."""

    def _meanCorrModel(colVec):
        if colVec.isnull().all():
            result = None
        else:
            model = sm.GLM(endog=colVec, exog=sm.add_constant(muVec), missing='drop')
            try:
                result = model.fit()
            except sm.tools.sm_exceptions.PerfectSeparationError:
                prnt = (colVec.name, colVec.isnull().sum(), colVec.shape[0])
                print('PerfectSeparationError with column "%s": %d of %d values are Nan' % prnt)
                result = None
        return result
    def _meanCorrResiduals(colVec):
        result = _meanCorrModel(colVec)
        if result is None:
            return colVec
        else:
            return colVec - result.predict(sm.add_constant(muVec))

    if cyVars is None:
        cyVars = cyDf.columns
    if meanVar is None:
        meanVar = 'Mean'
    if compCommVars is None:
        compCommVars = cyVars

    """Standardize each cytokine before taking the mean.
    Ensures equal "weighting" between cytokines when computing the mean level."""
    muVec = standardizedMean(cyDf, vars=compCommVars)
    
    models = cyDf.loc[:, cyVars].apply(_meanCorrModel, axis=0)

    ndf = cyDf.copy()
    ndf.loc[:, cyVars] = ndf.loc[:, cyVars].apply(_meanCorrResiduals, axis=0)
    ndf.loc[:, meanVar] = muVec

    if returnModels:
        return ndf, models
    else:
        return ndf

def standardizedMean(df, vars=None):
    if vars is None:
        vars = df.columns
    muS = df[vars].apply(lambda cy: (cy - cy.mean()) / cy.std(), axis=0).mean(axis=1)
    muS.name = 'Mean'
    return muS

def imputeNA(df, method='mean', dropThresh=0.75):
    """Drop rows (PTIDs) that have fewer than 90% of their cytokines"""
    outDf = df.dropna(axis=0, thresh=np.round(df.shape[1] * dropThresh)).copy()
    if method == 'resample':
        for col in outDf.columns:
            naInd = outDf[col].isnull()
            outDf.loc[naInd, col] = outDf.loc[~naInd, col].sample(naInd.sum(), replace=True).values
    elif method == 'mean':
        for col in outDf.columns:
            naInd = outDf[col].isnull()
            outDf.loc[naInd, col] = outDf.loc[~naInd, col].mean()
    elif method == 'predict':
        naInds = []
        for col in outDf.columns:
            naInd = outDf[col].isnull()
            outDf.loc[naInd, col] = outDf.loc[~naInd, col].mean()
            naInds.append(naInd)
        for naInd,col in zip(naInds, outDf.columns):
            if naInd.sum() > 0:
                otherCols = [c for c in outDf.columns if not c == col]
                mod = sklearn.linear_model.LinearRegression().fit(outDf[otherCols], outDf[col])
                outDf.loc[naInd, col] = mod.predict(outDf.loc[naInd, otherCols])
    return outDf

def convertLevel(mn, mx, val, mask = False, verbose = False):
    """Map function for cleaning and censoring cytokine values

    Remove ">" and "," characters, while setting minimum/maximum detection level
    and coverting to floats
    
    If mask is True, return a sentinel value for the mask (instead of a converted value)
    0 : minimum sensitivity/NS (no signal)
    0.5 : OK value
    1 : maximum/saturated
    -1 : NA/ND (not done)"""
    
    if isinstance(val, str):
        val = val.replace(',', '').replace('>', '')
    
    try:
        out = float(val)
    except ValueError:
        if val == 'NS':
            out = mn
        elif val == 'ND':
            out = np.nan
        elif val.find('N/A') >= 0:
            out = np.nan
        else:
            raise BaseException("Unexpected value: %s" % val)

    if not mx is None:
        if out >= mx:
            if verbose:
                print('Max truncation: %1.2f to %1.2f' % (out, mx))
            out = min(out, mx)
            if mask:
                return 1
    if not mn is None:
        if out <= mn:
            if verbose:
                print('Min truncation %1.2f to %1.2f' % (out, mn))
            out = max(out, mn)
            if mask:
                return 0
    if mask:
        if np.isnan(out):
            return -1
        else:
            return 0.5
    return out

def enforceSensitivity(cyDf, sensitivityS, truncateLevels=True, inplace=True):
    """Perform truncation based on reported Sensitivity and ">XXXX" values?"""
    if not inplace:
        cyDf = cyDf.copy()

    maskDf = cyDf.copy()
    for col in cyDf.columns:
        mn = sensitivityS[col]
        if not cyDf[col].dtype == dtype('float64'):
            gtPresent = (cyDf[col].str.contains('>') == True)
        if any(gtPresent):
            mx = cyDf.loc[gtPresent, col].map(partial(_convertLevel, 0, None)).max()
        else:
            mx = None
        maskDf.loc[:, col] = maskDf[col].map(partial(convertLevel, mn, mx, mask = True))
        if truncateLevels:
            cyDf.loc[:, col] = cyDf[col].map(partial(convertLevel, mn, mx))
        else:
            cyDf.loc[:, col] = cyDf[col].map(partial(convertLevel, mn, None))
    return cyDf, maskDf

def tranformCytokines(cyDf, maskDf=None, performLog=True, halfLOD=True, discardCensored=False, inplace=True):
    if maskDf is None:
        maskDf = pd.DataFrame(np.zeros(cyDf.shape), index=cyDf.index, columns=cyDf.columns)

    if not inplace:
        cyDf = cyDf.copy()
    
    if halfLOD:
        """Replace left-censored values with the LOD/2"""
        tmpView = cyDf.values
        tmpView[maskDf.values == 0] = tmpView[maskDf.values == 0]/2

    if performLog:
        """Take the log of all cytokine concentrations"""
        cyDf = np.log10(cyDf)
    if discardCensored:
        """Force censored values to be Nan"""
        tmpView = cyDf.values
        tmpView[(maskDf.values == 1) | (maskDf.values == 0)] = np.nan

    return cyDf