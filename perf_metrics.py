__author__ = 'josep'

import pandas as pd
import numpy as np
import sys
import numba


def np_auc(arr, mvt=False):
    """
    compute the auc for the recall-precision curve
    uses numpy and no pandas.
    See AUC docs in ranking folder
    :param
    arr: an nX1 or nX2 numpy array
            typically the result of pos_arr = pos_df.values.astype(np.intp) where pos_df is a df with cols position, value (optional) and size (optional).
            position = arr[:, 0] contains the positions of the products sold. The position array can have duplicated values. Positions are are counted from 1, not from 0
            value = arr[:, 1] contains the weights for each position. If not there, its is assume to be all 1's. Otherwise it could be the revenue of the item sale, ...
    mvt: compute the mean value theorem values ie t such that AUC = P(R(t))
    :return: area under PR curve, precision, recall, t (if mvt is True)
    Example:
    vpos = np.array([2] * 100 + [3] * 50 + [5] * 10 + [8] * 30)  (100 sales in position 2, 50 in position 3, 10 in position 5, ...)
    No values (weights) set up.
    np_auc(vpos) = 0.387

    Add weights
    values = np.array([1.0] * 150 + [2.0] * 40)
    arr = np.array([vpos, values]).transpose()
    np_auc(arr) = 0.404
    """
    s = np.shape(arr)
    if len(s) == 1:
        vpos = arr.astype(np.intp)
        values = np.ones(len(vpos), dtype=np.float)
    else:
        vpos = arr[:, 0].astype(np.intp)
        values = arr[:, 1].astype(np.float)
    values /= np.sum(values, dtype=np.float)

    val_pos = bincount(vpos, weights=values)[1:]  # drop position 0
    tp = val_pos.cumsum().astype(np.float)
    rec = tp / tp[-1]

    # compute fp
    n_t = np.cumsum(val_pos > 0, dtype=np.float)  # cumulative count of positions with sales
    thres = np.arange(1, len(tp) + 1)             # threshold array
    fp = (thres / n_t - 1.0) * tp                 # n_t can be 0 is there are no sales in early positions => nan from division by 0
    pre = np.nan_to_num(tp / (tp + fp))           # set nan to 0 in pre. Not to 1!

    # prepend values at threshold 0
    rec = np.insert(rec, 0, 0.0)
    pre = np.insert(pre, 0, 1.0)

    d_rec = np.diff(rec, n=1)                 # R(t) - R(t-1)
    avg_pre = moving_average(pre, 2)          # (P(t) + P(t-1)) / 2
    auc = np.dot(avg_pre, d_rec)

    thres = None
    if mvt is True:
        v = np.power(pre - auc, 2.0)  # error
        thres = np.argmin(v)
    return auc, pre, rec, thres


@numba.jit(nopython=True)
def bincount(values, weights):
    max_val = 1 + np.max(values)
    bins = np.zeros(max_val)
    for idx in range(len(values)):
        bins[values[idx]] += weights[idx]
    return bins


def moving_average(a, w=3):
    ret = np.cumsum(a, dtype=float)
    ret[w:] = ret[w:] - ret[:-w]
    return ret[w - 1:] / w



def np_mrr(arr):
    """
    compute MRR
    uses numpy and no pandas.
    :param
    pos_arr: an nX1 or nX2 numpy array
            typically the result of pos_arr = pos_df.values.astype(np.intp) where pos_df is a df with cols position, value (optional) and size (optional).
            position = arr[:, 0] contains the positions of the products sold. The position array can have repeated values. Positions are are counted from 1, not from 0
            value = arr[:, 1] contains the weights for each position. If not there, its is assume to be all 1's. Otherwise it could be the revenue of the item sale, ...
    :return: MRR = sum_position value_position / position
    Example:
    vpos = np.array([2] * 100 + [3] * 50 + [5] * 10 + [8] * 30)  (100 sales in position 2, 50 in position 3, 10 in position 5, ...)
    No values (weights) set up.
    np_mrr(vpos) = 0.381

    Add weights
    values = np.array([1.0] * 150 + [2.0] * 40)
    arr = np.array([vpos, values]).transpose()
    np_mrr(arr) = 0.340
    """
    s = np.shape(arr)
    if len(s) == 1:
        vpos = arr.astype(np.intp)
        values = np.ones(len(vpos), dtype=np.float)
    else:
        vpos = arr[:, 0].astype(np.intp)
        values = arr[:, 1].astype(np.float)
    values /= np.sum(values)

    val_pos = bincount(vpos, weights=values)[1:]  # drop position 0
    inv_pos = np.array(1.0 / np.arange(1, len(val_pos) + 1, dtype=np.float))

    return np.dot(val_pos, inv_pos), None, None, None  # to maintain consistency with auc

