#
# Copyright 2022- IBM Inc. All rights reserved
# SPDX-License-Identifier: MIT
#

# pyright: reportInvalidStringEscapeSequence=false

from typing import Callable, List, Tuple
import numpy as np

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

"""
This builds a function over (u, \alpha) that solves the following

    min_x \sum_{i=1}^n \phi x_i
    s.t. \sum_{i=1}^n x_i = \alpha
        0 \leq x_i \leq u

we require limits 

    0 \leq \alpha/n \leq u \leq \alpha \leq 1

coefs: coefficients (the \phi)
u: upper_limit (u)
a: capacity (\alpha)
"""
def build_allocation_fn_mu(coefs: np.ndarray, a: float) -> Callable[[float], float]:
    phi = np.sort(coefs)

    # the maximal possible value for ell is n 
    # the smallest possible value for ell is 1
    # we do not check, but in general we do not expect u = 0
    def f(u: float):
        ell = int(np.floor(a/u)) # type: ignore
        return np.sum(phi[:ell]) * u + (a - ell * u) * phi[ell]
    
    return f

def build_allocation_fn_nu(coefs: np.ndarray, a: float) -> Callable[[float], float]:
    phi = np.sort(coefs)

    # the maximal possible value for ell is n 
    # the smallest possible value for ell is 1
    # we do not check, but in general we do not expect u = 0
    def f(u: float):
        ell = int(np.floor(a/u)) # type: ignore
        return np.sum(phi[:ell-1]) * u + (a - (ell) * u) * phi[ell-1]
    
    return f

"""
These are the gradients of the allocation fn's mu and nu
"""
def build_allocation_grads_mu(coefs: np.ndarray, a: float) -> Callable[[float], float]:
    phi = np.sort(coefs)

    # the maximal possible value for ell is n 
    # the smallest possible value for ell is 1
    # we do not check, but in general we do not expect u = 0
    def g(u: float):
        ell = int(np.floor(a/u)) # type: ignore
        return np.sum(phi[:ell])  - ell * phi[ell]
    
    return g

def build_allocation_grads_nu(coefs: np.ndarray, a: float) -> Callable[[float], float]:
    phi = np.sort(coefs)

    # the maximal possible value for ell is n 
    # the smallest possible value for ell is 1
    # we do not check, but in general we do not expect u = 0
    def g(u: float):
        ell = int(np.floor(a/u)) # type: ignore
        return np.sum(phi[:ell-1]) - ell * phi[ell-1]
    
    return g

# ------------------------------------------------------------
# bounds
# ------------------------------------------------------------

Index2 = Tuple[int,int] # also defined in the other file

class TailFunctionSubProblemBound:

    def __init__(
        self, 
        X: np.ndarray,
        index: List[Index2],
        a: np.ndarray,
    ):

        # functions
        self._fns: List[Callable[[float], float]] = []

        # gradients
        self._gs: List[Callable[[float], float]] = []

        # FIXME: not 
        _index = {}
        for i,j in index:
            if i in _index:
                raise ValueError(f"row \'{i}\' repeated multiple times")
            _index[i] = j

        m, n = X.shape
        for i in range(m):
            f: Callable[[float], float]

            if i in _index:
                f = build_allocation_fn_nu(
                        np.delete(X[i,:], _index[i]), 
                        a[i],
                    )
                g = build_allocation_grads_nu(
                        np.delete(X[i,:], _index[i]), 
                        a[i],
                    )
            else:
                f = build_allocation_fn_mu(
                        X[i,:], a[i])
                g = build_allocation_grads_mu(
                        X[i,:], a[i])

            self._fns.append(f)
            self._gs.append(g)

        # basic range
        self._range = (
            np.max(a) / n,
            np.max(a),
        )

    def evaluate(self, x: float) -> float:

        # basic value checks
        if (x <= self._range[0]) or (x > self._range[1]):
            raise ValueError("x={} not in domain [{},{}]".format(x, *self._range))

        result = 0.
        for f in self._fns:
            result += f(x)
        return result

    def gradient(self, x: float) -> float:

        # basic value checks
        if (x <= self._range[0]) or (x > self._range[1]):
            raise ValueError("x={} not in domain [{},{}]".format(x, *self._range))

        result = 0.
        for g in self._gs:
            result += g(x)
        return result

    @property
    def domain(self) -> Tuple[float,float]:
        return self._range

# this returns a pair, one for row and one for column
# this version takes in an array of coefficients X
def tail_function_bound_subproblem(
    X: np.ndarray, 
    index: List[Index2],
    a: np.ndarray,
    b: np.ndarray,
    ) -> Tuple[
        TailFunctionSubProblemBound,
        TailFunctionSubProblemBound,
    ] :

    bound1 = TailFunctionSubProblemBound(
        X,
        index,
        a,
    )

    bound2 = TailFunctionSubProblemBound(
        X.transpose(),
        [(j,i) for i,j in index],
        b
    )

    return bound1, bound2

from otoc.iterative.util import convex_function_minimizer

"""
Returns the lower bound on the optimal value 
"""
def optimal_value_bound_subproblem(
    C: np.ndarray,
    index: List[Index2],
    a: np.ndarray,
    b: np.ndarray,
) -> float:

    if len(index) == 0:
        raise ValueError(f"index cannot be empty")
    
    # otherwise
    # the coefficient in front of index
    leading_coef: float = sum([
        C[i,j] for i, j in index
    ])

    # build the bounds
    bound_row, bound_col = tail_function_bound_subproblem(
        C,
        index,
        a, b
    )

    # helper function to compute the lower bound
    def compute_lower_bound(bound, tol=1e-7):
        if bound.domain[1] - bound.domain[0] < tol:
            # will run into numerical problems
            return bound.domain[0], 0.

        x = convex_function_minimizer(
            lambda x: leading_coef + bound.gradient(x),
            bound.domain[0] + 1e-8, # so we do not evaluate exactly at the left end-point
            bound.domain[1],
        )
        return x, bound.evaluate(x) + leading_coef * x 

    # compute the bounds with the respective x values in which they are obtained
    _, B1 = compute_lower_bound(bound_row)
    _, B2 = compute_lower_bound(bound_col)

    return max(B1, B2)
