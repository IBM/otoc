#
# Copyright 2022- IBM Inc. All rights reserved
# SPDX-License-Identifier: MIT
#

from typing import Tuple, List, Union
import numpy as np

from itertools import product

# type alias
Index = Union[Tuple[int,int], int]
Index2 = Tuple[int,int]

# check feasiblity
# this is the most basic check
# where in general is feasible if
# max a_p b_q / (1 - alp) <= c_k <= .. <= c_1
# holds for c_i <= min(a_ell,b_ell)

# but the simplified version is to consider
# max a_p b_q / (1 - k c) <= c <= min min (a_ell,b_ell)
# this is equiv to solving the quadratic
# c - M - k c^2 = 0 and checking the smaller intercept
# if it is less than min min (a_ell, b_ell) or not
# NOTE: with the simplification the order of index does not matter

# NOTE, this holds only if index does not repeat row or column
# but we dont check for that

# TODO: move this to other file
def check_feasibilty(
    index: List[Index2],
    a: np.ndarray,
    b: np.ndarray,
):

    m, n = len(a), len(b)
    k = len(index)
    _index = set(index)
    M = np.max([
        a[p] * b[q]
        for p,q in
        product(range(m), range(n))
        if (p,q) not in _index
    ])
    R = np.min([
        min(a[p], b[q])
        for p,q in index
    ])

    # intercept
    x = 1. - np.sqrt(1. - 4*k*M) # type: ignore
    x /= (2*k)

    return x <= R
