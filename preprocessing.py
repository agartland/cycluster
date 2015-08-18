import pandas as pd
import numpy as np
import statsmodels as sm
from functools import partial

__all__ = ['transformCytokines',
           'enforceSensitivity',
           'fillMissing',
           'normalizeLevels',
           'meanSubNormalize',
           'partialCorrNormalize']

def meanSubNormalize(cyDf, cyVars = None, compCommVars = None, meanVar = None):
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
        meanVar = 'compComm'
    if compCommVars is None:
        cyDf.columns

    """No standardizing cytokines before taking the mean (need units to stay in log-concentration)"""
    muVec = cyDf[compCommVars].mean(axis=1)
    
    ndf = cyDf.copy()
    ndf.loc[:, cyVars] = ndf[cyVars].apply(_normFuncSub, axis = 1)
    ndf.loc[:, meanVar] = muVec
    return ndf

def partialCorrNormalize(cyDf, cyVars = None, compCommVars = None, meanVar = None):
    """Computes residuals in-place after regressing each cytokine on the mean cytokine level
    Correlations among residuals are partial correlations, adjusting for meanVar"""
    def _meanCorrResiduals(colVec):
        model = sm.GLM(endog = colVec, exog = sm.add_constant(muVec), missing = 'drop')
        result = model.fit()
        return colVec - result.predict(sm.add_constant(muVec))

    if cyVars is None:
        cyVars = cyDf.columns
    if meanVar is None:
        meanVar = 'compComm'
    if compCommVars is None:
        cyDf.columns

    """Standardize each cytokine before taking the mean"""
    muVec = cyDf[compCommVars].apply(lambda cy: (cy - cy.mean()) / cy.std(), axis = 0).mean(axis=1)
    
    ndf = cyDf.copy()
    ndf.loc[:,cyVars] = ndf.loc[:,cyVars].apply(_meanCorrResiduals, axis = 0)
    ndf.loc[:, meanVar] = muVec
    return ndf

def fillMissing(df):
    """Drop rows (PTIDs) that have fewer than 90% of their cytokines"""
    out = df.dropna(axis=0, thresh=round(df.shape[1] * 0.9)).copy()
    for c in df.columns:
        naind = out[c].isnull()
        plugs = np.random.permutation(out[c].loc[~naind].values)[:naind.sum()]
        out.loc[naind, c] = plugs
    return out

def _convertLevel(mn, mx, val, mask = False):
    """Map function for cleaning and censoring cytokine values

    Remove ">" and "," characters, while setting minimum/maximum detection level
    and coverting to floats
    
    If mask is True, return a sentinel value for the mask (instead of a converted value)
    0 : minimum sensitivity/NS (no signal)
    0.5 : OK value
    1 : maximum/saturated
    -1 : NA/ND (not done)"""
    
    if isinstance(val, basestring):
        val = val.replace(',','').replace('>','')
    
    try:
        out = float(val)
    except ValueError:
        if val == 'NS':
            out = mn
        elif val == 'ND':
            out = nan
        elif val.find('N/A') >= 0:
            out = nan
        else:
            raise BaseException, "Unexpected value: %s" % val

    if not mx is None:
        if out >= mx:
            print 'Max truncation: %1.2f to %1.2f' % (out,mx)
            out = min(out,mx)
            if mask:
                return 1
    if not mn is None:
        if out <= mn:
            print 'Min truncation %1.2f to %1.2f' % (out,mn)
            out = max(out,mn)
            if mask:
                return 0
    if mask:
        if isnan(out):
            return -1
        else:
            return 0.5
    return out

def enforceSensitivity(cyDf, sensitivityS, truncateLevels = True, inplace = True):
    """Perform truncation based on reported Sensitivity and ">XXXX" values?"""
    if not inplace:
        cyDf = cyDf.copy()

    maskDf = cyDf.copy()
    for col in cyDf.columns:
        mn = sensitivityS[col]
        if not cyDf[col].dtype == dtype('float64'):
            gtPresent = (cyDf[col].str.contains('>') == True)
        if any(gtPresent):
            mx = cyDf.loc[gtPresent,col].map(partial(_convertLevel,0,None)).max()
        else:
            mx = None
        maskDf.loc[:,col] = maskDf[col].map(partial(convertLevel, mn, mx, mask = True))
        if truncateLevels:
            cyDf.loc[:,col] = cyDf[col].map(partial(convertLevel, mn, mx))
        else:
            cyDf.loc[:,col] = cyDf[col].map(partial(convertLevel, mn, None))
    return cyDf, maskDf

def tranformCytokines(cyDf, maskDf, performLog = True, halfLOD = True, inplace = True):
    if not inplace:
        cyDf = cyDf.copy()
    
    if hlafLOD:
        """Replace left-censored values with the LOD/2"""
        tmpView = cyDf.values
        tmpView[maskDf.values == 0] = tmpView[maskDf.values == 0]/2

    if performLog:
        """Take the log of all cytokine concentrations"""
        cyDf = np.log(cyDf)
    return cyDf