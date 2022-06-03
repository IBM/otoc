#
# Copyright 2022- IBM Inc. All rights reserved
# SPDX-License-Identifier: MIT
#

# pyright: reportInvalidStringEscapeSequence=false

# Euclidean projection for Transport Polytope
# - projection for the positive values
# - projection for the row sums
import numpy as np
from typing import Optional, cast

# projection for positive values
"""
this projection just takes the positive part of the matrix
"""
def euclidean_projection_positive_matrices(X: np.ndarray) -> np.ndarray:
    neg_pos = np.where(X < 0)
    Y = np.copy(X)
    Y[neg_pos] = 0.
    return Y

# projection for row-col sums for matrices
"""
the projection of a matrix X onto the set of matrices 
    \{ Y1 = a, Y^T 1 = b \}
is given as

1. unravel X into x (row-by-row)
2. compute

    y = A^\dagger [a; b] + (I_mn - A^T A^\dagger ) x

    where

    M1 = I_m - J_m / (m+n)
    M2 = I_n - J_n / (m+n)

    A^\dagger = [ 
        1/n ( I_m \kron 1_n - \frac{1}{m + n} ; 1/m ( 1_m \kron I_n - \frac{1}{m + n}
    ] = [
        M1 \kron 1/n 1_n ; 1/m 1_m \kron M2
    ]

    A^T (A^T)^\dagger = (A^\dagger A)^T = A^\dagger A
    = M1 \kron P_n + P_m \kron M2

3. put y back into Y (row-by-row)
"""
def euclidean_projection_row_col_sums_matrices(
    X: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    mode: Optional[str] = 'version2', # TODO: remove
    ) -> np.ndarray:


    if mode == 'version2':
        return version2(X, a, b)
    else:
        raise ValueError(f"unknown mode {mode}")


"""
This one computes using matrices

    M1 = I_m - J_m / (m+n)
    M2 = I_n - J_n / (m+n)
    
    Y1 = (M1 * a) \kron (1/n 1_n) + (1/m 1_m) \kron (M2 * b)
    Y2 = [ M1 * X * P_n ] + [ P_m * X * M2 ]
"""
def version2(
    X: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    ) -> np.ndarray:

    m, n = len(a), len(b)

    # see above for definition of these matrices
    M1: np.ndarray = np.eye(m) - np.ones((m,m)) / (m + n)
    M2: np.ndarray = np.eye(n) - np.ones((n,n)) / (m + n)

    Y1 = compute_y1_version2(a, b)
    Y2 = compute_y2_version2(X, a, b, M1, M2)

    return Y1 + (X - Y2)

# computes in matrix form
"""
This computes y1 in the matrix form
"""
def compute_y1_version2(
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:

    m, n = len(a), len(b)

    a2 = a - np.sum(a) / (m + n)
    b2 = b - np.sum(b) / (m + n)

    return np.tile(a2[:, np.newaxis],(1, n)) / n + np.tile(b2, (m, 1)) / m

# computes in matrix form
"""
This computes y2 in matrix form
"""
def compute_y2_version2(
    X: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    M1: np.ndarray,
    M2: np.ndarray,
) -> np.ndarray:

    m, n = len(a), len(b)
    rs = cast(np.ndarray, np.mean(X, axis=1))
    cs = cast(np.ndarray, np.mean(X, axis=0))

    Y = np.dot(
        M1,
        np.tile(rs[:, np.newaxis],(1, n)),
    )

    Y += np.dot(
        np.tile(cs,(m, 1)),
        M2,
    )

    return cast(np.ndarray, Y)

