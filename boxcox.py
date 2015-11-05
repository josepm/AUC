__author__ = 'josep'

"""
# ##############################################################
# ################## boxcox transformations ####################
# ##############################################################
"""
import numpy as np
import sys
from scipy.optimize import minimize_scalar

def boxcox_transform(lambdax, X):
    """
    Performs a box-cox transformation to data vector X.
    WARNING: elements of X should be all positive!
    """
    if not isinstance(X, np.ndarray):
        print 'boxcox_opt: invalid type'
        sys.exit(0)

    if any(X <= 0):
        print 'Non-positive value(s) in data'
        sys.exit(0)
    return np.log(X) if abs(lambdax) < 1.0e-5 else (np.power(X, lambdax) - 1.0) / lambdax

def boxcox_inv(lambdax, X):
    return np.exp(X) if abs(lambdax) < 1.0e-5 else np.power(1.0 + lambdax * X, 1.0 / lambdax)

def boxcox_loglik(lambdax, X):
    """
    Computes the log-likelihood function for a transformed vector.
    """
    if not isinstance(X, np.ndarray):
        print 'boxcox_opt: invalid type'
        sys.exit(0)

    n = len(X)
    if n == 0:
        print 'no data'
        sys.exit(0)
    Xtrans = boxcox_transform(lambdax, X)
    meanX = np.mean(Xtrans)
    S = np.sum(np.power(Xtrans - meanX, 2.0))
    S1= (-n / 2.0) * np.log(S / n)
    S2 = (lambdax - 1.0) * np.sum(np.log(X))
    return S1 + S2

def boxcox_opt(X):
    """
    optimal lambda parameter for data set X
    :param X: np array
    :return:
    """
    if not isinstance(X, np.ndarray):
        print 'boxcox_opt: invalid type'
        sys.exit(0)

    def f(lambdax, X):
        return -boxcox_loglik(lambdax, X)
    res = minimize_scalar(f, args=(X,))
    return res.x

