#
# Copyright 2022- IBM Inc. All rights reserved
# SPDX-License-Identifier: MIT
#

import numpy as np

from typing import Callable

# TODO: should we move this function somewhere?
# helper function
def bisect(
    f: Callable, 
    a: float, 
    b: float, 
    stopThreshold: float =1e-4, 
    numItermax: int =100,
) -> float:
    m = (a + b) / 2.
    g = f(m)
    i = 0
    while (i < numItermax) and (np.abs(g) > stopThreshold):
        if g < 0:
            # search right
            a = m
        else:
            # search left
            b = m
            
        # iterate
        m = (a + b) / 2.
        g = f(m)
        i += 1
    return m

# this finds the minimum point of a convex function
def convex_function_minimizer(
    gradient: Callable[[float],float], 
    a: float, b: float,
    **kwargs,
) -> float:

    ga, gb = gradient(a), gradient(b)
    if (ga <= 0) and (gb <= 0):
        # since this is a convex function, it means 
        # that (one of the) optimum is at b
        return b
    elif (ga >= 0) and (gb > 0):
        # since this is a convex function, it means 
        # that (one of the) optimum is at a
        return a
    elif (ga < 0) and (gb > 0):
        # then this must be the case that gradient(a) < 0 and
        # gradient(b) > 0
        return bisect(gradient, a, b, **kwargs)
    else:
        raise RuntimeError("gradient does not belong to convex function")

# for each position, compute the row max excluding the position
def compute_row_max_excluding_current(
    X: np.ndarray,
):
    # if row length is 1, then current should be highly saturated
    # and there is no information from others
    # if row length is 0, then nothing to do
    if X.shape[1] <= 1:
        return np.zeros(X.shape)

    # get the max and 2nd max positions
    I = np.argsort(-X, axis=-1)[:,:2]

    # construct the solution
    rm = np.zeros(X.shape)

    # for each row
    for i in range(X.shape[0]):

        # for positions that are not max, their value 
        # will be the max
        rm[i,:] = X[i,I[i,0]]

        # for the max position, the value will be the 2nd-max
        rm[i, I[i,0]] = X[i,I[i,1]]
    return rm
