# Optimal Transport With Order Constraints

The `otoc` directory contains the following:
1. `otoc`: optimal transport with order constraints (first-order method).
2. `search_otoc_candidates`: obtain set of alternative transport plan candidates given a base transport plan.
    * The base transport plan should be a regular OT transport plan (without optimal constriants).
    * Can use the [POT package](https://pythonot.github.io/) to compute one, see the [code examples](#code-examples) below.

The above two techniques are described in the below publication; please consider citing if you find this repository useful.

## Citation

Fabian Lim, Laura Wynter, Shiau Hong Lim, "Order Constraints in Optimal Transport", 2022. https://arxiv.org/abs/2110.07275.

## Usage

To install the `python` wheel:

```shell
pip install git+https://github.com/IBM/otoc@main#egg=otoc
```

### Code Examples

`otoc` solves the OT with OC formulation with a first-order method.

```python
from otoc import otoc
import numpy as np

C = np.array([
    [1, 5, 3],
    [0, 1, 4],
])
a = np.array([.5, .5])
b = np.array([1./3, 1./3, 1./3])

# solve the optimal transport with order constraints
# meta give some statistics
X, meta = otoc(
    [(0,2)], # X[0,2] >= all others
    a, b, C, 
)
# X
# array([[0.16658407, 0.        , 0.3333329 ],
#        [0.16674927, 0.3333329 , 0.        ]])

X, meta = otoc(
    [(0,2), (1,0)], # X[1,0] >= X[0,2] >= others
    a, b, C, 
)
# X
# array([[0.        , 0.16669247, 0.333301  ],
#        [0.333301  , 0.16664086, 0.        ]])
```

Alternatively, calling with `implementation='iterator'` will return an `Iterator` object, that iterates once per first-order iteration.

```python
# ... intialize a, b, C as above

# returns an iterator object
algo = otoc(
    [(0,2)], 
    a, b, C, 
    implementation='iterator'
)

# this iterate through the first-order method until stopping
for _ in algo:
    pass

# prints solution and cost
print(algo.solution)
# (array([[0.16658407, 0.        , 0.3333329 ],
#         [0.16674927, 0.3333329 , 0.        ]]),
#  1.4999156499311608)

# prints statistics
print(algo.projection_statistics)
# Statistics(res_primal=8.260397888704896e-05, res_dual=2.4625149137936733e-05, norm_dual=3.570708156906683)

# prints iteration count
print(algo.iteration_cnt)
# 41
```

`otoc_candidates` learns multiple candidates given a base transport plan by searching OC's in a tree-search manner. This is an `Iterator` class that iterates as the tree nodes are searched.

```python
# install the python optimal transport package
# https://pythonot.github.io/
# pip install pot==0.7.0

# --- generate a, b, C as above

# compute the OT base plan using the python optimal
# transport package
import ot # imports POT
base_cost, base_plan = ot.lp.emd2(
    a, b, C,
    return_matrix=True,
)

# search candidates on top of base plan
from otoc import search_otoc_candidates
algo_search = search_otoc_candidates(
    a, b, C, 
    base=base_plan['G'],
    thresholds=(1.,1.), # small example, so just relax thresholds
    k1=10,
    k2=5,
    k3=1,
)

# iterate each node (which is an Iterator object of otoc)
# each node corresponds to an otoc with some learnt order constraints
for node in algo_search:

    # run the first order iterations of otoc
    # until stopping
    for _ in node:
        pass

# third best candidate
algo_search.best_solution(n=2)
# array([[0.16676627, 0.        , 0.33333243],
#        [0.16656706, 0.33333243, 0.        ]])

# history of nodes explored
algo_search._history
# each gives the index, bound_value, and the otoc iterator (algo)
# [History(index=[(0, 2)], bound_value=1.3333333333333333, algo=<otoc.iterative.dual_method.DualMethod object at 0x7fb6242dc9d0>, skip=False),
#  History(index=[(1, 1)], bound_value=1.3333333333333333, algo=<otoc.iterative.dual_method.DualMethod object at 0x7fb6242dc1d0>, skip=False),
#  History(index=[(1, 0)], bound_value=1.3333333333333333, algo=<otoc.iterative.dual_method.DualMethod object at 0x7fb6242dc590>, skip=False),
#  History(index=[(0, 0)], bound_value=1.6666666666666665, algo=<otoc.iterative.dual_method.DualMethod object at 0x7fb6242dce10>, skip=False),
#  History(index=[(0, 1)], bound_value=2.3333333533333334, algo=<otoc.iterative.dual_method.DualMethod object at 0x7fb6242dd210>, skip=True),
#  History(index=[(1, 2)], bound_value=2.0, algo=<otoc.iterative.dual_method.DualMethod object at 0x7fb6242dd490>, skip=True)]

# check which explored node was the second best candidate
algo_search.best_history_index(n=2)
# 1

# confirm
algo_search._history[1].algo.solution
# (array([[0.16676627, 0.        , 0.33333243],
#         [0.16656706, 0.33333243, 0.        ]]),
#  1.5000959909228264)
```

## License

The code in the [otoc](./otoc) and [tests](./tests) directories are released under the [MIT License](./LICENSE). The [otoc](./otoc) has about 1000 lines of code (ignoring space and comments) and only one package dependency (`numpy`).

