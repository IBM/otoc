#
# Copyright 2022- IBM Inc. All rights reserved
# SPDX-License-Identifier: MIT
#

# implement OC with OT as an ADMM

import numpy as np
from typing import Union, List, Tuple

# types for projection 
Index = Union[Tuple[int,int], int]

from otoc.iterative.projection.pava import euclidean_projection as proj1
from otoc.iterative.projection.transport import euclidean_projection_positive_matrices as proj2
from otoc.iterative.projection.transport import euclidean_projection_row_col_sums_matrices as proj3

# main entry point
def projection_subproblem(
    index: Union[Index, List[Index]],
    a: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
    numItermax: int = 10000,
    stopThreshold: float = 1e-4,
    penalty: float = 1.0,
    implementation = 'one-shot', 
):
    """First-order method for solving Optimal Transport with Order Constraints

    Parameters
    ----------
    index: list[tuple(int,int)]
        list of indices over m x n space to indicate where OCs are placed. E.g. [(1,0), (2,2)] means X[2,2] >= X[1,0] >= others
    a : np.ndarray
        souce distribution of size m
    b : np.ndarray
        target distribution of size n
    C : np.ndarray
        cost matrix of size m x n
    numItermax : int
        number of max iterations
    stopThreshold : float
        stop when iteration error reaches this
    penalty : float
        penalty for ADMM
    implementation 
        choose between 'one-shot' or 'iterator'

    """

    # if its not a list, make it into one
    if isinstance(index, list) == False:
        index = [index] # type: ignore
    elif len(index) == 0: # type: ignore
        raise ValueError(f"there must be at least one order constraint")

    # create the iterative object
    algo = _dual_method(
        index, # type: ignore
        a, b, C, # type: ignore
        numItermax=numItermax,
        stopThreshold=stopThreshold,
        penalty=penalty,
    )

    if implementation == 'one-shot':
        for _ in algo:
            pass

        return algo.solution[0], {
            'cost': algo.solution[1],
            'num_iter': algo.iteration_cnt,
            'err_primal': algo.projection_statistics.res_primal, # type: ignore
            'err_dual': algo.projection_statistics.res_dual,  # type: ignore
        }

    elif implementation == 'iterator':
        return algo
    else:
        raise NotADirectoryError(f"unknown implementation \'{implementation}\'")
    

# -----------------------------------------------------
# WITH COST_DESCENT FRAMEWORK
# -----------------------------------------------------

# Its a bit of a misnomer since its more of a dual method
# but will treat f(x) as a cost descent

from otoc.iterative.dual_method import DualMethod, Statistics
from typing import Tuple

def _dual_method(
    index: List[Index],
    a: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
    numItermax: int = 10000,
    stopThreshold: float = 1e-4, 
    penalty: float = 1.0,
) -> DualMethod:

    # we treat the current solution as Z
    # so X and U are bounded by closure
    X = np.zeros(C.shape)
    U = np.zeros(C.shape)
    res_primal, res_dual = 1., 1.

    def termination() -> bool:
        return max(res_primal, res_dual) <= stopThreshold

    # generate X_t+1 from Z_t and U_t
    def pointProjectionX(Z: np.ndarray) -> np.ndarray:
        # this returns X
        return proj3(
            Z - U - C / penalty, # type: ignore
            a, b
        )

    # generate Z_t+1 from X_t+1 and U_t
    # compute res_dual
    def pointProjectionZ(
        _X: np.ndarray,  # this is X_t+1
        Z: np.ndarray, # this is Z_t
    ) -> np.ndarray:
        nonlocal res_dual
        nonlocal X

        X = _X # update
        _T = proj1(
            X + U,
            index
        )
        _T = proj2(_T) # this will be new Z

        # this will be s_{k} = - rho * (z_k - z_{k-1})
        res_dual = penalty * np.linalg.norm(Z - _T) # type: ignore

        # return Z_t+1
        return _T

    # update U_t+1 from X_t+1 and Z_t+1
    # return the stats
    def updateDual(Z: np.ndarray) -> Statistics:
        nonlocal res_primal
        nonlocal U

        # X would already been updated to X_t+1
        # Z will be Z_t+1
        _T = X - Z # type: ignore
        res_primal = np.linalg.norm(_T) # record
        U += _T
        return Statistics(
            res_primal=res_primal, # this is norm(r_k)
            res_dual=res_dual, # this is norm(s_k)
            norm_dual=penalty * np.linalg.norm(U), # this is norm(y_k)
        )

    def cost(Z: np.ndarray) -> float:
        return np.multiply(Z,C).sum() # type: ignore

    return DualMethod(
        cost,
        pointProjectionX,
        pointProjectionZ,
        updateDual,
        np.zeros(C.shape), # initial point
        termination=termination,
        numItermax=numItermax,
    )

