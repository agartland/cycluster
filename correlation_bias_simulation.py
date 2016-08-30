"""Is it possible that the normalization is subject to a negative correlation bias
that could be leading to two possible false discoveries:
    (1) Anti-correlation between analytes that are actually independent
    (2) Inverse associations when in fact there is no association (e.g. BS4)

Set up a simulation with 5 analytes, X_i: 4 are positively correlated with each other
and associated with a binary variable, Y. The last is either not correlated with anything
or is inversely correlated with the other X_i and Y.

If we instead normalized by dividing by the mean, would that induce a bias?
How does negative correlation bias depend on sample size? It seems like having a
low number of variables of which most are positively correlated, might be a situation
in which negative correlation bias with the other variable could be observed."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from functools import partial

from corrplots import *

N = 100
k = 5
rho = 0.7

C = rho * np.ones((k, k))
C[diag_indices(k)] = 1.

np.random.seed(110820)
d = np.random.multivariate_normal(np.zeros(k), C, 100)
d[:, 0] = np.random.randn(N)

plt.figure(1)
combocorrplot(pd.DataFrame(d), method='pearson')

def meanNormalize(d):
    muVec = np.mean(d, axis=1)[:, None]
    return d / muVec

def partialCorrNormalize(d):
    def _meanCorrModel(colVec):
        model = sm.GLM(endog=colVec, exog=sm.add_constant(muVec))
        result = model.fit()
        return result
    def _meanCorrResiduals(colVec):
        result = _meanCorrModel(colVec)
        return colVec - result.predict(sm.add_constant(muVec))

    """Standardize each variable before taking the mean.
    Ensures equal "weighting" of variables when computing the mean level."""
    muVec = np.mean((d - np.mean(d, axis=0)[None, :]) / np.std(d, axis=0)[None, :], axis=1)
    
    # models = [_meanCorrModel(d[:,coli]) for coli in range(d.shape[1])]

    nd = d.copy()
    for coli in range(d.shape[1]):
        nd[:, coli] = _meanCorrResiduals(d[:, coli])    
