#
# Copyright 2022- IBM Inc. All rights reserved
# SPDX-License-Identifier: MIT
#
import unittest

# --------------------------------------------------
# helpers
# --------------------------------------------------

import numpy as np
def _generate_problem(m: int, n: int, rng):
    a = np.ones((m,)) / m
    b = np.ones((n,)) / n
    
    C = rng.random((m,n))
    return C, a, b

def _generate_sizes(rng, max_size: int=100):
    # TODO: sometimes I get zero here
    return int(rng.integers(max_size)), int(rng.integers(max_size))

def generate_test_instances(
    seed: int = 0,
    num_instances: int = 100,
    num_oc: int = 1,
):
    rng = np.random.default_rng(seed) # type: ignore

    for i in range(num_instances):
        m, n = _generate_sizes(rng)
        C, a, b = _generate_problem(m, n, rng)
        if num_oc <= min(m,n):
            yield i, C, a, b

import os
import json
def read_fixture(
    seed: int = 0,
    num_instances: int = 10,
    num_oc: int = 1,
):

    # TODO: this should depend on seed or num,_oc
    filename = f'test_admm_seed_{seed}.jsonl'

    with open(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            filename,
        ),
        'r',
        encoding='utf-8',
    ) as f:
        i = 0
        while i < num_instances:
            try:
                d = f.readline()
                d = json.loads(d)

                # if d['test'] != i:
                #     raise ValueError(f"problem with fixture test \'{i}\'")

                if d['num_oc'] == num_oc:
                    yield d
                    i += 1
            except EOFError:
                raise StopIteration

# --------------------------------------------------
# helpers (END)
# --------------------------------------------------

from otoc.iterative.admm import projection_subproblem

SP_INDEX = [(x,x) for x in range(10)]

PARAMS1 = {
    'numItermax': 10000,
    'stopThreshold': 1e-4,
    'penalty': 1,
}

from typing import cast, Dict

def _test(
    tester,
    **kwargs,
):
    _g1 = generate_test_instances(**kwargs)
    _g2 = read_fixture(**kwargs)
    for (i, C, a, b), data in zip(_g1, _g2):
        with tester.subTest(instance=i):

            _, meta = projection_subproblem(
                SP_INDEX[:kwargs['num_oc']], # type: ignore
                a, b, C,
                numItermax=PARAMS1['numItermax'],
                stopThreshold=PARAMS1['stopThreshold'],
                penalty=PARAMS1['penalty'],
            )

            meta = cast(Dict, meta)
            tester.assertEqual(
                i, data['test'], 
                msg="wrong instance",
            )
            tester.assertAlmostEqual(
                meta['cost'], data['cost'], 
                places=3, msg="cost not equal",
            )
            tester.assertAlmostEqual(
                meta['err_primal'], data['primal'], 
                places=3, msg="primal not equal",
            )
            tester.assertAlmostEqual(
                meta['err_dual'], data['dual'], 
                places=3, msg="dual not equal",
            )
            tester.assertEqual(
                meta['num_iter'], data['iter'], 
                msg="iter not equal",
            )

class TestProjectionSubproblem(unittest.TestCase):

    def test_no_order_constraints(self):
        self.assertRaises(
            ValueError,
            projection_subproblem,
            [], # if no order constraints are provided
            np.array([]), np.array([]), np.array([[]]),
        )

    def test_solve_oc_1_params_1(self):
        _test(self, seed=0, num_instances=5, num_oc=1)

    def test_solve_oc_2_params_1(self):
        _test(self, seed=0, num_instances=5, num_oc=2)

if __name__ == '__main__':
    unittest.main()

