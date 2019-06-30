from typing import Callable
import numpy as np
from platypus import NSGAII, Problem, Real
from abc import ABC, abstractmethod


class OptimizerCore(ABC):

    def __init__(self,
                 function: Callable = None,
                 x_range: list = [0, 1],
                 n_variables: int = 1,
                 n_objectives: int = 2) -> None:
        self.n_variables = n_variables
        self.n_objectives = n_objectives
        self._function = function
        self.optimizer_result: list = None
        self.problem = Problem(nvars=n_variables, nobjs=n_objectives)

        self._x_range = np.array(x_range)
        if self._x_range.ndim == 1:
            x_min, x_max = self._x_range
            self.problem.types[:] = Real(x_min, x_max)
        else:
            variables_range_list = []
            for _, (x_min, x_max) in enumerate(self._x_range):
                variables_range_list.append(Real(x_min, x_max))
            self.problem.types[:] = variables_range_list

        if self.function is not None:
            self.problem.function = self.function

    @abstractmethod
    def run(self):
        return

    @property
    def function(self):
        return self._function

    @function.setter
    def function(self, objective_function):
        self._function = objective_function
        self.problem.function = self.function
        return

    @property
    def x_range(self):
        return self._x_range

    @x_range.setter
    def x_range(self, x_range):
        self._x_range = np.array(x_range)
        if self._x_range.ndim == 1:
            self.n_variables = 1
            x_min, x_max = self._x_range
            self.problem.types[:] = Real(x_min, x_max)

        elif self._x_range.ndim == 2:
            self.n_variables = self._x_range.shape[0]
            variables_range_list = []
            for _, (x_min, x_max) in enumerate(self._x_range):
                variables_range_list.append(Real(x_min, x_max))
            self.problem.types[:] = variables_range_list

        elif self._x_range.ndim > 2:
            raise ValueError('x_range has too many dementions')

        return
