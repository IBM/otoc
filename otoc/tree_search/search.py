#
# Copyright 2022- IBM Inc. All rights reserved
# SPDX-License-Identifier: MIT
#

import numpy as np

from typing import NamedTuple, Tuple, Optional, List, cast
from typing import Union, Dict

import bisect

from otoc.iterative.dual_method import DualMethod

from otoc.tree_search.util import Index, Index2

INFTY = np.finfo(np.float64).max # type: ignore

# Modes
MODE_NODES_EXPLORED = 'nodes-explored'
MODE_CANDIDATES_OBATINED = 'candidates-obtained'

# ------------------------------------------------------------
# Classes
# ------------------------------------------------------------

class History(NamedTuple):
    index: List[Index2]
    bound_value: float
    algo: DualMethod
    skip: bool = False

# a bare history without the DualMethod
# for convinience
class HistoryBare(NamedTuple):
    index: List[Index2]
    bound_value: float
    solution: Optional[np.ndarray]
    cost: Optional[float]
    skip: bool = False

# abstract class 
from typing import Iterator

# returns true if index is a subseq of index2
def _is_direct_descendent(
    index: List[Index2],
    index2: List[Index2],
):
    for x, y in zip(index, index2):
        if y != x:
            # then index1 cannot be a descendent
            return False
    return True

class Candidate(NamedTuple):
    cost: float
    index: List[Index2]
    position: int # indexes the history

    def __lt__(self, other):
        # put precendence on the index
        # strictly speaking, children should not have
        # larger cost, but it could be possible due to numeric errors

        if _is_direct_descendent(self.index, other.index):
            # if index is direct descendent
            return False
        elif _is_direct_descendent(other.index, self.index):
            # if self is a direct descendent
            return True
        # otherwise go by costs
        return self.cost < other.cost

class Variate(NamedTuple):
    index: List[Index2]
    metric: float

    def __lt__(self, other):
        return self.metric < other.metric


# ------------------------------------------------------------
# problems
# ------------------------------------------------------------

# TODO: move the parameters somewhere?
DEFAULT_PARAMETERS = {
    'numItermax': 1e4,
    'stopThreshold': 1e-4,
    'penalty': 1.0,
}

"""This is the sub-problem.

Note: History.index is stored with the higher-level node to the left.
Thus it has to be reversed when passing into projection_subproblem 

i.e.: History.index = [(1,0), (2,1)] means X[1,0] >= X[2,1] >= others
"""
def _subproblem(
    index: Union[Index, List[Index]],
    a: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
    projection=DEFAULT_PARAMETERS,
    **kwargs,
) -> DualMethod:

    from otoc.iterative.admm import projection_subproblem

    return projection_subproblem( # type: ignore
        [x for x in reversed(index)],  # type: ignore
        a, b, C,
        **projection,
        implementation='iterator',
    )

from otoc.tree_search.bound import optimal_value_bound_subproblem
from otoc.tree_search.util import check_feasibilty

# ------------------------------------------------------------
# problems: Classes
# ------------------------------------------------------------

# base class
class SubProblemStrategy:

    # initialize
    def initialize(self) -> List[Variate]:
        raise NotImplementedError

    # generate
    def gen_candidates(self, index: List[Index2], X: np.ndarray) -> List[Variate]:
        raise NotImplementedError

    # pop from stack
    def pop(self, stack: List[Variate]) -> Variate:
        raise NotImplementedError

class OptimalConstraintsTreeSearch:

    """Class object that implements the OTOC tree search.

    Iterator that generates a DualMethod for each node searched in the tree.
    """

    def __init__(self,
        a: np.ndarray,
        b: np.ndarray,
        C: np.ndarray,
        subproblems: SubProblemStrategy,
        **kwargs,
    ) :

        """Parameters
        ----------
        a : np.ndarray
            souce distribution of size m
        b : np.ndarray
            target distribution of size n
        C : np.ndarray
            cost matrix of size m x n
        subproblems: SubProblemStrategy
            the tree search strategy 
        **kwargs: 
            numCandidates: int
                the number of additional candidates (on top of base)
            limitCandidates: int
                stop tree search when this number of nodes encountered
            limitCandidatesMode
                determines the nodes are counted when checking limitCandidates
            limitDepth : (int, None)
                the depth of the search tree
            deactivate_bounds : bool
                do not use bounds to skip nodes
            acceptableError: float
                the error bound for DualMethod
        """

        self._a : np.ndarray = a
        self._b : np.ndarray = b
        self._C : np.ndarray = C

        # this is to iterate the subproblems
        self._strategy: SubProblemStrategy = subproblems

        # store the compute history
        self._history: List[History] = []
        self._num_non_skip = 0 # counter

        # stack
        self._stack: List[Variate] = []

        # check feasiblity
        # FIXME: at some point change it to True
        if kwargs.get('checkFeasibility', False) == False:
            # this is the dummy one, just return true
            self._feasible = lambda v: True
        else:
            self._feasible = lambda v: check_feasibilty(v.index, self._a, self._b)

        # initialize the stack
        for variate in self._strategy.initialize():
            if self._feasible(variate):
                bisect.insort(self._stack, variate)

        # store the top candidates
        self._candidates: List[Candidate] = []
        self._best_of: int = kwargs.get('numCandidates', 1)
        self._candidate_limit: Optional[int] = kwargs.get('limitCandidates')
        self._candidate_limit_mode: str = kwargs.get('limitCandidatesMode', MODE_NODES_EXPLORED)
        # different modes of counting when the candidate
        # limit is satisfied
        # - by number of nodes explored
        # - by number of candidates obtained
        self._limit_depth: int = kwargs.get('limitDepth', None)

        if (
            self._candidate_limit_mode not in [
                MODE_NODES_EXPLORED,
                MODE_CANDIDATES_OBATINED,
            ] 
        ):
            raise NotImplementedError(f"invalid mode \'{self._candidate_limit_mode}\'")
           
        # store the parameters 
        self._parameters = kwargs

        if kwargs.get('deactivate_bounds', False):
            self._skip = lambda _, __: False
            # logging.warn('bounds not used to speed up search')
        else:
            self._skip = lambda best_cost, bound_cost: best_cost < bound_cost

        # min uerror
        self._uerror = kwargs.get('acceptableError', 1.)

    def __iter__(self) -> Iterator[DualMethod]:
        return cast(Iterator[DualMethod], self)

    def __next__(self) -> DualMethod:

        while (
            self._best_of > 0 and  # FIXME: lazy
            (   # if not limit is specified, we keep going
                # until stack is exhausted
                (self._candidate_limit is None) and 
                (len(self._stack) > 0) 
            ) or 
            ( 
                (self._candidate_limit is not None) and # candidate check
                (self._candidate_limit_mode == MODE_NODES_EXPLORED) and
                (len(self._history) < self._candidate_limit) and
                (len(self._stack) > 0) 
            ) or 
            ( 
                (self._candidate_limit is not None) and # candidate check
                (self._candidate_limit_mode == MODE_CANDIDATES_OBATINED) and
                (self._num_non_skip < self._candidate_limit) and
                (len(self._stack) > 0) 
            )
        ):
            # NOTE: for simplicity, we assume that all the CostDescents in history has been completed. 
            # should be a fair assumption 

            # this is the previous state of the history
            # we run this here because algo is returned without runing
            # so we need to update from the previous run of algo
            if len(self._history) > 0:
                self.update_candidate(
                    len(self._history)-1, # points to this current entry
                    self._history[-1],
                )

            # use history to generate candidates
            if len(self._history) > 0:
                # add more candidates
                _h = self._history[-1]
                if (
                    (_h.skip == False) and
                    (
                        _h.algo.projection_statistics.U_error # type: ignore
                        < 
                        self._uerror
                    ) and
                    (_h.algo.solution[1] <  # type: ignore
                        self.best_cost(self._best_of)) and
                    (
                        self._limit_depth is None or 
                        len(_h.index) < self._limit_depth
                    )
                ):
                    # only need to add if there is no skip
                    # and if the subtree root can find a better 
                    # candidate
                    # NOTE: no point falling back on the _h.bound_value anymore because
                    # we have already computed the solution 
                    # at this point, and this is gauranteed
                    # to be tighter

                    for variate in self._strategy.gen_candidates(
                        _h.index,
                        _h.algo.solution[0], # type: ignore
                    ):
                        if self._feasible(variate):
                            bisect.insort(self._stack, variate)
                else:
                    # NOTE: we could have done the same by generating history objects and marked skipped. but this would burden the search
                    # so we just choose to not generate the objects
                    pass

            # because it should be sorted in increasing order
            # pop a variate off
            variate = self._strategy.pop(self._stack)

            bound = optimal_value_bound_subproblem(
                self._C, 
                variate.index, 
                self._a, 
                self._b,
            )

            self._history.append(
                History(
                    variate.index,
                    bound, #
                    _subproblem(
                        variate.index,  # type: ignore
                        self._a, self._b, 
                        self._C, **self._parameters,
                    ),
                    # self.best_history_index(self._best_of), 
                    self._skip(
                        self.best_cost(self._best_of),
                        bound,
                    ) # best_cost always updates from best_history_index
                )
            )

            if self._history[-1].skip == False:
                self._num_non_skip += 1
                return self._history[-1].algo

        raise StopIteration

    # update the candidate
    def update_candidate(self, h_index: int, h: History) -> None:
        if h.skip == True:
            # nothing to do
            return 

        if h.algo.projection_statistics.U_error > self._uerror: # type: ignore
            return

        cost = h.algo.solution[1]
        index = h.index
        if cost is not None:
            bisect.insort(self._candidates, Candidate(cost, index, h_index))
            self._candidates = self._candidates[:self._best_of] # truncate

    # return the index to the best history estimate
    def best_history_index(self, n=1) -> Optional[int]:
        if len(self._candidates) >= n:
            return self._candidates[n-1].position
        else:
            return None
            
    # get the current best cost
    def best_cost(self, n=1) -> float:
        if len(self._candidates) >= n:
            return self._candidates[n-1].cost
        else:
            return INFTY

    # get the current best solution
    def best_solution(self, n=1) -> Optional[np.ndarray]:
        idx = self.best_history_index(n)
        if idx is None:
            return None
        h = self._history[idx]
        return h.algo.solution[0]

  
from otoc.iterative.util import compute_row_max_excluding_current
from itertools import product
import numpy.matlib

# this strategy finds the least saturated coefficients, using a 
# pre-computed assignment to determine the level of saturation
class LeastSaturatedCoefficientFirstStrategy(SubProblemStrategy):

    def __init__(
        self, 
        X: np.ndarray, 
        saturationThreshold: Tuple[float,float] = (1.,1.),
        a: Optional[np.ndarray] = None,
        b: Optional[np.ndarray] = None,
        index_filter: Optional[List[Index2]]=None,
        aggregate: str ='min', 
        topn: Optional[int] = None,
    ):

        # this is for aggregating between rows and cols
        if aggregate == 'min':
            self._aggregate = np.minimum
        elif aggregate == 'max':
            self._aggregate = np.maximum
        else: 
            raise NotImplementedError
        
        self._thres = saturationThreshold
        self._index_filter = index_filter
        self._root_solution = X
        self._topn = topn
        self._a = a
        self._b = b
    
    # compute coefs
    def _compute(self, X: np.ndarray) -> List[Tuple[Index2, float]]:

        # ravel and get the min
        # the smaller this value, the more likely the candidate
        m, n = X.shape

        Y: np.ndarray
        if (
            (self._a is None) or
            (self._b is None)
        ):
            # if a,b not specified, we assume its the uniform case
            Y = X * max(m,n) 
        else:
            Y = np.minimum( # type: ignore
                np.matlib.repmat(self._a[:,np.newaxis], 1, n), # type: ignore
                np.matlib.repmat(self._b[np.newaxis,:], m, 1) # type: ignore
            )
            Y = np.divide(X, Y) # type: ignore
 
        _coeffs: np.ndarray
        _coeffs = self._aggregate(
            compute_row_max_excluding_current(Y),
            compute_row_max_excluding_current(Y.transpose()).transpose(),
        ) # type: ignore

        results = []
        for i,j in product(range(m), range(n)):
            if (
                (Y[i,j] <= self._thres[0]) and
                (_coeffs[i,j] <= self._thres[1]) 
            ):
                results.append((
                    (i,j), 
                    _coeffs[i,j],
                ))

        if self._index_filter is not None:
            results = [(idx,cf) for idx,cf in results if idx in self._index_filter]

        if self._topn is not None:
            # take the topn least candidates
            results = sorted([
                (idx,cf) for idx,cf in results 
            ], key=lambda x: x[1])[:self._topn]

        return results

    # initialize
    def initialize(self) -> List[Variate]:
        _results = self._compute(self._root_solution)

        _c = [
            Variate(
                index=[idx],
                metric=metric,
            )
            for idx, metric in _results
        ]
        if self._topn is not None:
            # take the topn least candidates
            return sorted([
                v for v in _c
                ], 
                key=lambda v: v.metric
            )[:self._topn]
        else:
            return _c

    # get candidate
    def gen_candidates(self, index: List[Index2], X: np.ndarray) -> List[Variate]:
        _results = self._compute(X)
        _a, _b = zip(*index)
        _a, _b = set(_a), set(_b)

        _c = [
            Variate(
                index=index+[idx],
                metric=metric,
            )
            for idx, metric in _results
            if (
                (idx[0] not in _a) and
                (idx[1] not in _b) # we do not consider new candidates
            )
        ]
        if self._topn is not None:
            # take the topn least candidates
            return sorted([
                v for v in _c
                ], 
                key=lambda v: v.metric
            )[:self._topn]
        else:
            return _c

    # pop from stack
    def pop(self, stack: List[Variate]) -> Variate:
        return stack.pop(0) # pick the most unsure guy


def search_optimal_transport_candidates2(
    a: np.ndarray, 
    b: np.ndarray, 
    C: np.ndarray,
    strategy: Union[str, Tuple[str, Dict]] = 'least-saturated-coef',
    **kwargs,
) -> OptimalConstraintsTreeSearch:

    """This searches through various candidate OTOC problems that each be solved to get a candidate plan.

    The order constraints are learnt via a search method to ensure that the resulatant optimal transport
    plans are diverse. This is the version that exposes that strategy object for more configuration, 
    and the kwargs into OptimalConstraintsTreeSearch can be freely specified.

    Parameters
    ----------
    a : np.ndarray
        souce distribution of size m
    b : np.ndarray
        target distribution of size n
    C : np.ndarray
        cost matrix of size m x n
    strategy : (str, Dict)
        search strategy (str) and accompanying params (Dict)
    **kwargs : 
        passed to OptimalConstraintsTreeSearch

    Returns
    -------
    OptimalConstraintsTreeSearch 
        Iterator object that generates otoc algorithms given learnt order constraints
    """

    _strategy: str
    _strategy_options: Dict
    if isinstance(strategy, str):
        _strategy = strategy
        _strategy_options = {}
    else:
        _strategy = strategy[0]
        _strategy_options = strategy[1]

    strat : SubProblemStrategy
    if _strategy == 'least-saturated-coef':
        strat = LeastSaturatedCoefficientFirstStrategy(
            _strategy_options['base_solution'],
            **{
                k:v for k,v in _strategy_options.items() if k in [
                    'saturationThreshold', 
                    'index_filter',
                    'index_limit',
                    'aggregate',
                    'a',
                    'b',
                ]
            })
    else:
        raise RuntimeError('unknown subproblem strategy {}'.format(strategy))
    
    return OptimalConstraintsTreeSearch(a, b, C, strat, **kwargs)

def search_optimal_transport_candidates(
    a: np.ndarray, 
    b: np.ndarray, 
    C: np.ndarray,
    base: np.ndarray,
    thresholds: Tuple[float,float] = (.5,.5),
    k1: int = 20, # number of candidates to be searched 
    k2: int = 5, # number of candidates to return
    k3: int = 1, # depth of search tree
) -> OptimalConstraintsTreeSearch:

    """This searches through various candidate OTOC problems that each be solved to get a candidate plan.

    The order constraints are learnt via a search method to ensure that the resulatant optimal transport
    plans are diverse. This is the version that exposes that strategy object for more configuration, 
    and the kwargs into OptimalConstraintsTreeSearch can be freely specified.

    Parameters
    ----------
    a : np.ndarray
        souce distribution of size m
    b : np.ndarray
        target distribution of size n
    C : np.ndarray
        cost matrix of size m x n
    base : np.ndarray
        base solution matrix size m x n
    thresholds : (float, float)
        tau1, tau2 thresholds for controlling diversity of search
    k1: int
        number of candidates to be searched
    k2: int
        number of opt candidates to be returned
    k3: int
        depth of the search tree

    Returns
    -------
    OptimalConstraintsTreeSearch 
        Iterator object that generates otoc algorithms given learnt order constraints
    """

    return search_optimal_transport_candidates2(
        a, b, C,
        strategy=(
            'least-saturated-coef',
            {
                'base_solution': base, # base plan
                'saturationThreshold': thresholds,
                'a': a,
                'b': b,
            }
        ),
        numCandidates=k2-1,
        limitCandidates=k1,
        limitCandidatesMode='candidates-obtained',
        limitDepth=k3,
    )
