#
# Copyright 2022- IBM Inc. All rights reserved
# SPDX-License-Identifier: MIT
#
import unittest

from otoc.tree_search.search import search_optimal_transport_candidates2

import os
import numpy as np

def _test(
    tester,
    case: int,
):

    filename = f'test_case_{case}.json'
    import json

    with open(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            filename,
        ),
        'r',
        encoding='utf-8',
    ) as f:
        data = json.load(f)

    C = np.array(data['costs'], dtype=float)
    X = np.array(data['base']['solution'], dtype=float)
    m, n = C.shape
    a = np.ones((m,)) / m
    b = np.ones((n,)) / n

    active_a, active_b = data["active"]

    # build the filter
    from itertools import product
    _f = [
        (i,j)
        for i, j in product(
            range(len(a)),
            range(len(b)),
        ) 
        if active_a[i] and active_b[j]
    ]

    algo = search_optimal_transport_candidates2(
        a, b, C,
        strategy=('least-saturated-coef', {
            'index_filter': _f,
            'base_solution': X,
            'saturationThreshold': tuple(data["tau"]),
        }),
        numCandidates=data["k2"]-1,
        projection=data["subproblem"],
    )

    for i,(subproblem, hd) in enumerate(
        zip(algo, data['search'])
    ):

        with tester.subTest(instance=i):

            # this runs the subproblem
            for _ in subproblem:
                pass

            h = algo._history[i]

            tester.assertEqual(
                h.index, [tuple(x) for x in hd["index"]], 
                msg="wrong index",
            )

            tester.assertEqual(
                h.skip, hd["skipped"], 
                msg="was not skipped properly",
            )

            tester.assertAlmostEqual(
                h.bound_value, hd['bound'], 
                places=3, msg="bound not equal",
            )

            if h.skip is False:
                tester.assertAlmostEqual(
                    h.algo.solution[1], data['cost'], 
                    places=3, msg="cost not equal",
                )
                tester.assertAlmostEqual(
                    h.algo.projection_statistics.custom.res_primal,  # type: ignore
                    data['primal'], 
                    places=3, msg="primal not equal",
                )
                tester.assertAlmostEqual(
                    h.algo.projection_statistics.custom.res_dual,  # type: ignore
                    data['dual'], 
                    places=3, msg="dual not equal",
                )
                tester.assertEqual(
                    h.algo.projection_statistics.custom.iterCnt,  # type: ignore
                    data['iter'], 
                    msg="iter not equal",
                )

class TestTreeSearchNoCandidateAndDepthLimit(unittest.TestCase):

    def test_five_candidates_case_1(self):
        _test(self, case=1)

if __name__ == '__main__':
    unittest.main()
