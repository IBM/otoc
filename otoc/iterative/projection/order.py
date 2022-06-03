#
# Copyright 2022- IBM Inc. All rights reserved
# SPDX-License-Identifier: MIT
#

import numpy as np
import logging
from typing import Union, Tuple, cast

logger = logging.getLogger(__name__)

# projection for matrices
def euclidean_projection_matrices(
    X: np.ndarray, 
    index: Union[Tuple[int,int], int],
    **kwargs,
) -> np.ndarray:
    m: int
    n: int
    m, n = X.shape

    if isinstance(index, tuple):
        index = n * index[0] + index[1]

    x = np.ravel(X)
    y = euclidean_projection(x, cast(int, index), **kwargs)
    return np.reshape(y, (m,n))

# projection for vector
def euclidean_projection(
    x: np.ndarray, 
    index: int,
    inplace=False,
) -> np.ndarray:

    # sort in dec order
    sigma = np.argsort(-x)

    # rank of index
    r: int = int(np.where(sigma == index)[0])

    # logger.debug("index {} is ranked {}".format(index, r))
    candidates = list(x[sigma[:r]]) # all those that are ranked less than r
    total: float = x[sigma[r]] # store the sum

    s: int = 0 # size of S 
    while len(candidates) > 0:
        x2: float = candidates.pop(0) # pop the head
        
        threshold: float = (x2 + total) / (s + 2) # try to include head
        # print("|S|: {}, x2: {}, threshold: {}".format(
        #     s, x2, threshold,
        # ))
        if threshold > x2:
            # if the threshold exceeds, do not include anymore
            # break from loop
            break
              
        # can be included into S
        total += x2
        s += 1

    # print("|S| final: {}".format(s))
    # print("total", total)
    if inplace:
        result = x
    else:
        result = np.copy(x)
    result[sigma[:s]] = result[index] = total / (s + 1)
    return result

from typing import List, Callable

# this is a slightly different implementation of euclidean_projection
# above where we have the index value changing but we keep the rest
# of the values the same
def build_threshold_compute_fn(
    x: np.ndarray, 
    index: int,
) -> Tuple[
    Callable[[float], Tuple[float,int,int]],
    List[int],
]: 

    if len(x) == 0:
        raise ValueError("x cannot be zero length")

    # sort in dec order
    sigma = np.argsort(-x)

    # initial rank of index
    r: int = int(np.where(sigma == index)[0])
    sigma = [x for x in sigma if x != index] # remove index
    n: int = len(x) - 1

    v: float = x[index] # this is the starting value of x at index
    s: int = 0
    total: float = 0.

    def helper(x2: float) -> Tuple[float,int,int]:
        nonlocal total, v
        nonlocal s, r

        if x2 > v:
            raise ValueError (f"x \'{x2:1.2f}\' cannot be larger than \'{v:1.2f}\'")

        v = x2 # update 
        # update r because now x2 is a reduction of v
        while r < n and v < x[sigma[r]]:
            # increment the rank to indicate its new position
            r += 1

        # increment s
        while s < r:
            # this should really be the next tau
            tau: float = (x[sigma[s]] + total + v) / (s + 2)

            if tau > x[sigma[s]]:
                break

            total += x[sigma[s]]
            s += 1

        return (total+v) / (s+1), s, r
            

    return helper, sigma 

# this is a different implementation of build_threshold_compute_fn
# in the sense that it builds the entire tau function
# as a function of lambda >= 0
def build_threshold_compute_fn2(
    x: np.ndarray, 
    index: int,
) -> Tuple[
    Callable[[float], Tuple[float, int]],
    List[int],
    np.ndarray,
    np.ndarray,
]: 

    if len(x) == 0:
        raise ValueError("x cannot be zero length")

    # sort in dec order
    sigma = np.argsort(-x)

    # initial rank of index
    r: int = int(np.where(sigma == index)[0])

    sigma = [x for x in sigma if x != index]
    n: int = len(x) - 1

    # increment s
    s: int = 0
    total: float = x[index]
    while s < r:
        # this should really be the next tau
        _t: float = (x[sigma[s]] + total) / (s + 2)

        if _t > x[sigma[s]]:
            break

        total += x[sigma[s]]
        s += 1
    # print("s - ", s)

    # it is gauratnteed that s >= 1
    # if is gauranteed that x[sigma[s]] will always
    # be smaller than tau

    # compute the inflextion points
    _pts = [0.]
    _vals = [ total / (s+1)]
    for s2 in range(s, n):

        _pts.append(
            (_vals[-1] - x[sigma[s2]]) * (s2 + 1)
            + _pts[-1]
        )
        _vals.append(
            x[sigma[s2]]
        )

    inflex = np.array(_pts) # inflextion 
    inflex_vals = np.array(_vals) # values

    def helper(lam: float) -> Tuple[float, int]:

        if lam < 0:
            raise ValueError (f"lam \'{lam:1.2f}\'must be positive")

        _j, = np.where(lam > inflex) # type: ignore
        if len(_j) == 0:
            # assume its zero
            idx = 0
        else:
            idx = max(_j)

        _s: int = s + idx

        return inflex_vals[idx] - (lam - inflex[idx]) / (_s + 1), _s
    
    return helper, sigma, inflex, inflex_vals
