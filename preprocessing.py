from scipy import stats

import pandas as pd
import numpy as np

def convertLevel(mn, mx, val, mask = False, ):
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


"""Perform truncation based on reported Sensitivity and ">XXXX" values?"""
def censoring():
    truncateLevels = True
    maskDf = df.copy()
    for col in cyVars['All']:
        mn = cyVarsDf.Sensitivity.loc[col]
        if not df[col].dtype == dtype('float64'):
            gtPresent = df[col].str.contains('>') == True
        if any(gtPresent):
            mx = df.loc[gtPresent,col].map(partial(convertLevel,0,None)).max()
        else:
            mx = None
        maskDf[col] = maskDf[col].map(partial(convertLevel, mn, mx, mask = True))
        if truncateLevels:
            df[col] = df[col].map(partial(convertLevel, mn, mx))
        else:
            df[col] = df[col].map(partial(convertLevel, mn, None))

def preprocess():
    """Replace left-censored values with the LOD/2"""
    tmpView = df[cyVars['All']].values
    tmpView[maskDf[cyVars['All']].values == 0] = tmpView[maskDf[cyVars['All']].values == 0]/2

    """Take the log of all cytokine concentrations"""
    df.loc[:,cyVars['All']] = log(df.loc[:,cyVars['All']])


def makeModuleVariables(df, modules):
    """Define variable for each module by standardizing all the cytokines in the module and taking the mean"""
    standardizeFunc = lambda col: (col - nanmean(col))/nanstd(col)
    out = None
    for tiss in modules.keys():
        for n in modules[tiss].keys():
            tmpS = df[modules[tiss][n]].apply(standardizeFunc, raw = True).mean(axis=1)
            tmpS.name = 'Module%s %s' % (n,tiss)
            if out is None:
                out = pd.DataFrame(tmpS)
            else:
                out = out.join(tmpS)
    return out

def fillMissing(df, cyVars):
    """Drop rows (PTIDs) that have fewer than 90% of their cytokines"""
    out = df[cyVars].dropna(axis=0, thresh=round(len(cyVars) * 0.9)).copy()
    for c in cyVars:
        naind = out[c].isnull()
        plugs = permutation(out[c].loc[~naind].values)[:naind.sum()]
        out.loc[naind, c] = plugs
    return out



def define_complete_common():
    """Complete-common cytokines: those cytokines for which we have data in all patients and all compartments (22)"""
    compN = {'ETT':(~admitDf[cyVars['ETT']].isnull()).sum(axis=0).max(),
             'Serum':(~admitDf[cyVars['Serum']].isnull()).sum(axis=0).max(),
             'Plasma':(~cyDf[cyVars9['Plasma']].isnull()).sum(axis=0).max(),
             'NW':(~cyDf[cyVars9['NW']].isnull()).sum(axis=0).max()}
    compComm = {}
    compComm['base'] = [c for c in common['base'] if all([(~admitDf[c + ' ' + s].isnull()).sum() == compN[s] for s in ['ETT','Serum']] + [(~cyDf[c + ' ' + s].isnull()).sum() == compN[s] for s in ['Plasma','NW']])]
    for tiss in ['NW', 'Plasma', 'ETT', 'Serum']:
        compComm[tiss] = [c + ' ' + tiss for c in compComm['base']]

    """Take the mean across the 'complete-common' cytokines:
    those cytokines for which we have data in all patients and all compartments (22)"""

def meanSubNormalize(df, cyVars, compCommVars, meanVar):
    """Normalize cytokine columns by the log-mean for each patient, within each compartment.
    The point is that if cytokine concentrations are generally high for one sample or another,
    this might dominate the covariation of cytokines across patients (both within/across compartments).

    We subtract off the mean since the "overall inflamation" level
    that we are adjusting for would probably be on the fold-change concentration scale.
    (additive on the log-concentration scale)"""
    def _normFuncSub(vec):
        out = vec - muVec
        return out

    """No standardizing cytokines before taking the mean (need units to stay in log-concentration)"""
    muVec = df[compCommVars].mean(axis=1)
    
    ndf = df.copy()
    ndf.loc[:, cyVars] = ndf[cyVars].apply(_normFuncSub, axis = 1)
    ndf.loc[:, meanVar] = muVec
    return ndf

def partialCorrNormalize(df, cyVars, compCommVars, meanVar):
    """Computes residuals in-place after regressing each cytokine on the mean cytokine level
    Correlations among residuals are partial correlations, adjusting for meanVar"""
    def _meanCorrResiduals(colVec):
        model = sm.GLM(endog = colVec, exog = sm.add_constant(muVec), missing = 'drop')
        result = model.fit()
        return colVec - result.predict(sm.add_constant(muVec))

    """Standardize each cytokine before taking the mean"""
    muVec = df[compCommVars].apply(lambda cy: (cy - cy.mean()) / cy.std(), axis = 0).mean(axis=1)
    
    ndf = df.copy()
    ndf.loc[:,cyVars] = ndf.loc[:,cyVars].apply(_meanCorrResiduals, axis = 0)
    ndf.loc[:, meanVar] = muVec
    return ndf

'''
"""PIC"""
nAdmitDf = meanSubNormalize(admitDf, cyVars['ETT'], compComm['ETT'], 'compComm ETT')
nAdmitDf = meanSubNormalize(nAdmitDf, cyVars['Serum'], compComm['Serum'], 'compComm Serum')

"""FLU09"""
nCyDf = meanSubNormalize(cyDf, cyVars9['NW'], compComm['NW'], 'compComm NW')
nCyDf = meanSubNormalize(nCyDf, cyVars9['Plasma'], compComm['Plasma'], 'compComm Plasma')


"""PIC"""
nAdmitDf = partialCorrNormalize(admitDf, cyVars['ETT'], compComm['ETT'], 'compComm ETT')
nAdmitDf = partialCorrNormalize(nAdmitDf, cyVars['Serum'], compComm['Serum'], 'compComm Serum')

"""FLU09"""
nCyDf = partialCorrNormalize(cyDf, cyVars9['NW'], compComm['NW'], 'compComm NW')
nCyDf = partialCorrNormalize(nCyDf, cyVars9['Plasma'], compComm['Plasma'], 'compComm Plasma')
'''