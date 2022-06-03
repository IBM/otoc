#
# Copyright 2022- IBM Inc. All rights reserved
# SPDX-License-Identifier: MIT
#

import numpy as np

# TODO: remove
# order.py should combine with pava.py
from otoc.iterative.projection.order import euclidean_projection as order_ep
from otoc.iterative.projection.order import euclidean_projection_matrices as order_ep_matrices

from typing import List, Union, Tuple

# type definition
Index = Union[Tuple[int,int], int]

# ------------------------------------------------------
# Helper
# ------------------------------------------------------
"""
convert List[Index] to List[int]
"""
def _convert_indices(
    indices: List[Index],
    shape: Tuple[int,int],
) -> List[int]:

    _, n = shape
    result = []
    for x in indices:
        if isinstance(x, tuple):
            result.append(x[0] * n + x[1])
        else:
            result.append(x)
    return result

# TODO: combine order with pava
from otoc.iterative.projection.order import build_threshold_compute_fn2
from otoc.iterative.util import bisect

# function to solve for tau and lambda0
def _solve_tau_lambda0(
    f, pts, val,  # f - lambda, inflection pts and their values
    v, slope, 
    **kwargs,
):

    # print ("mean", v, "slope", slope)

    # this function solves a monotonic decreasing
    # f, and a monotonic increasing v + lam * slope
    # for positive lam
    # in particular, this f is the special tau function

    if np.abs(v-val[0]) < 1e-4:
        # if within some tolerance
        return 0.

    # leftmost value
    if v > val[0]:
        raise ValueError(f"value \'{v:1.2f}\' cannot be larger than leftmost value \'{val[0]:1.2f}\'")

    # find the right limit
    idx = np.max(np.where(val > v)[0]) 

    if idx < len(val) - 1:
        # internal
        _b = pts[idx+1]
    else:
        # this only works for the tau function
        _, s = f(0) # initial s
        # extrapolate the endpoint from the last
        # inflection point. The slope is the property
        # of the tau function
        _b = val[-1] + (val[-1] - v) * (s + idx + 1)
        _b += pts[-1]

    return bisect(
        lambda x: (v + x * slope) - f(x)[0],
        0., _b,
        **kwargs,
    )

# FIXME: we do the vector only version first
def _euclidean_projection_single_shot_vector(
    x: np.ndarray, 
    order: List[int],
    inplace=False,
) -> np.ndarray:

    if len(order) == 0:
        raise ValueError("order must have at least one element")

    if inplace:
        result = x
    else:
        result = np.copy(x)

    # other indices
    _S = [i for i in range(len(x)) if i not in order]
    _S = [order[0]] + _S
    _n = len(order)

    # build the threshold function
    fn, sigma, pts, vals = build_threshold_compute_fn2(
        x[_S],  # slice will be a copy
        0,      # because that is where we pasted 
    )
    # v = x[order[0]] # initial value
    lam0: float = 0.

    # get the initial tau and s value
    tau, s = fn(lam0)
    # print ("x", x[_S], _S)
    # print ("s",s, len(_S))

    if len(order) == 1:
        # if single order, can stop here
        result[order[0]] = tau
        for i in sigma[:s]:
            result[_S[i]] = tau
        return result

    # otherwise we need to proceed with PAVA
    # initialize the whole list
    # we do not use the first position in this
    # array to unify the indexing with the
    # number of blocks
    _left = [-1] * (_n + 1)
    _right = [-1] * (_n + 1)
    _val = [np.nan] * (_n + 1)

    # initialize block number 1
    _left[1] = 0   # initialize
    _right[1] = 1  # initialize
    _val[1] = tau  # initialize

    # print ("tau", tau)

    # initialize counters
    _bl = 1 # number of blocks
    for i in range(1,_n):

        # increment
        _bl += 1
        _left[_bl], _right[_bl] = i, i + 1
        _val[_bl] = x[order[i]]

        # violation check
        while _bl >= 2 and _val[_bl] <= _val[_bl-1]:

            if _bl == 2: 
                # special block, handled differently
                _lam = _solve_tau_lambda0(
                    fn, pts, vals,
                    # np.mean(x[order[1:]]), 
                    np.mean(x[[
                        order[i] for i in
                        range(_left[_bl-1]+1,_right[_bl])
                    ]]), 
                    1.0 / (_right[_bl] - _left[_bl-1] - 1),
                )
                # m = np.mean(x[[
                #     order[i] for i in
                #     range(_left[_bl-1]+1,_right[_bl])
                # ]]), 
                _val[_bl-1], s = fn(_lam)
                # print ('double block', i, fn(_lam), _lam, m, _left[_bl-1]+1, _right[_bl], "s", s)
            else:
                _n0 = _right[_bl-1] - _left[_bl-1] 
                _n1 = _right[_bl] - _left[_bl]
                _val[_bl-1] = (
                    _n0 * _val[_bl-1] + 
                    _n1 * _val[_bl]
                ) / (_n0 + _n1)

            _right[_bl-1] = _right[_bl]
            _bl -= 1

    # print (_bl, _val)
    # print (_left)
    # print (_right)

    # assign the solution
    # add 1 because first position is dummy
    for i in range(1,_bl+1):
        for j in range(_left[i], _right[i]):
            # print ("pos", j, order[j])
            result[order[j]] = _val[i]

    # print ("s",s)

    # assign the other positions (negative)
    for i in sigma[:s]:
        # print ("sigma", i, _S[i])
        result[_S[i]] = _val[1] # from block 1

    return result


# this returns the projection
def _euclidean_projection_using_single_shot(
    x: np.ndarray, 
    order: List[Index],
    inplace=False,
) -> np.ndarray:

    _order: List[int]
    if len(x.shape) == 1:

        # TODO: refactor
        # should not have a seperate mode for len(x.shape) == 1
        _order = _convert_indices(order, (len(x),1))
        return _euclidean_projection_single_shot_vector(
            x,
            _order,
            inplace=inplace,
        )
    elif len(x.shape) == 2:
        m, n = x.shape
        _order = _convert_indices(order, (m,n))
        y = _euclidean_projection_single_shot_vector(
            x.ravel(), # ravel returns copy
            _order,
            inplace=True,
        )
        return np.reshape(y, (m,n))
    else:
        raise NotImplementedError

# ---------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------
from typing import Union, Tuple

def build_pava_error_computation(
    order: List[Index],
    N: Index,  # limit of the space
):

    # this will work for both matrices and vectors
    _S: List[int]
    _order: List[int]
    if isinstance(N, int):
        _order = _convert_indices(order, (N,1))
        _S = [i for i in range(N) if i not in _order]
    elif isinstance(N, tuple):
        _order = _convert_indices(order, N)
        _n = N[0] * N[1]
        _S = [i for i in range(_n) if i not in _order]

    def compute(
        x: np.ndarray,
    ) -> Tuple[List[float],float]:

        # NOTE: need to do an ravel here because
        # advanced indexing is hard in the matrix space
        # because [(x,y), (i,j), ...] does not mean
        # to return X[x,y], X[i,j], ...
        y = np.ravel(x)

        # compute the error of the remaining guys in S wrt x[order[0]]
        if len(_S) == 0:
            re = 0.
        else:
            re = max(np.max(y[_S]) - y[_order[0]], 0)

        # compute the errors of
        # x[order[0]] <= x[order[1]] <= ... <= x[order[k-1]]
        pe = [max(y[i] - y[j],0) for i,j in zip(_order, _order[1:])]

        return pe, re

    return compute

# NOTE: slightly different from the other projectors, here
# we decided not to have a distinguished matrix version
# so this will work for both vectors and matrices
def euclidean_projection(
    x: np.ndarray, 
    order: List[Index],
    mode='single-shot', # TODO: refactor
    **kwargs,
) -> np.ndarray:

    if isinstance(order, list) == False:
        raise ValueError(f"order must be list, got \'{type(order)}\'")

    if len(order) == 0:
        raise ValueError("order must be at least 1")
    elif len(order) == 1:
        # if this is only a single order projection
        # we can directly use this function. 
        # no need for dykstra
        if x.ndim == 1 and isinstance(order[0], int):
            return order_ep(x, order[0])
        elif x.ndim == 2 and isinstance(order[0], tuple):
            return order_ep_matrices(x, order[0])
        else:
            raise NotImplementedError

    # TODO: 
    # should have only one mode
    if mode == 'single-shot':
        return _euclidean_projection_using_single_shot(
            x, order, **kwargs,
        )
    else:
        raise NotImplementedError(f"unknown mode \'{mode}\'")
