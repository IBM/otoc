#
# Copyright 2022- IBM Inc. All rights reserved
# SPDX-License-Identifier: MIT
#

from typing import Callable, Tuple, NamedTuple
from typing import Optional, Iterator, cast
import numpy as np

class Statistics(NamedTuple):
    res_primal: float
    res_dual: float
    norm_dual: float

    # this is the error used to decide termination
    @property
    def U_error(self):
        return max(self.res_primal, self.res_dual)

CostFunction = Callable[[np.ndarray], float]
PointProjectorX = Callable[[np.ndarray], np.ndarray]
PointProjectorZ = Callable[[np.ndarray, np.ndarray], np.ndarray]
DualUpdater = Callable[[np.ndarray], Statistics]
Termination = Callable[[],bool]

# this is a dual method with two projections
class DualMethod:

    def __init__(self, 
        costFunction: CostFunction,
        pointProjectionX: PointProjectorX, 
        pointProjectionZ: PointProjectorZ, 
        dualUpdater: DualUpdater,
        initialPoint: np.ndarray,
        termination: Optional[Termination] = None,
        numItermax: int = int(1e3),
    ) -> None:

        # dependency injection of various strategies
        self._costFunction = costFunction
        self._pointProjectionX = pointProjectionX
        self._pointProjectionZ = pointProjectionZ
        self._dualUpdater = dualUpdater

        if termination is None:
            self._termination = lambda : False
        else:
            self._termination = termination

        # points
        # - current point is the Z
        self._currentPoint : np.ndarray = initialPoint
        self._bestPoint : Optional[np.ndarray] = None
        self._bestCost : Optional[float] = None
        # print ("initial", initialPoint)

        # projection
        self._projectionStats: Optional[Statistics] = None

        # interation counters
        self._numItermax = numItermax
        self._iterCnt : int = 0

    def __iter__(self) -> Iterator[float]:
        return cast(Iterator[float], self)

    def __next__(self) -> float :

        if self._bestPoint is None:

            # if bestPoint is un-initialized
            self._bestPoint = self._currentPoint
            self._bestCost = self._costFunction(self._currentPoint)

            return self._bestCost

        elif self._iterCnt < self._numItermax:

            # if have iterations to complete
            # check the termination
            if self._termination():
                raise StopIteration

            # increment the cnt
            self._iterCnt += 1
        
            # projection
            _X = self._pointProjectionX(
                self._currentPoint,
            )

            # projection
            self._currentPoint = self._pointProjectionZ(
                _X, self._currentPoint
            )

            # update dual
            self._projectionStats = self._dualUpdater(
                self._currentPoint
            )

            # compare the points
            costOfPoint = self._costFunction(
                self._currentPoint,
            )

            self._bestCost = costOfPoint
            self._bestPoint = self._currentPoint

            # return the best point and cost
            return cast(float, self._bestCost)

        raise StopIteration

    @property
    def solution(self) -> Tuple[Optional[np.ndarray], Optional[float]]:
        return (self._bestPoint, self._bestCost)

    @property
    def iteration_cnt(self) -> int:
        return self._iterCnt

    @property
    def projection_statistics(self) -> Optional[Statistics]:
        return self._projectionStats
