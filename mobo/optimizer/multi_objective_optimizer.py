from typing import Callable
import numpy as np
# from platypus import NSGAII, NSGAIII, SMPSO
from platypus import NSGAII
from platypus import NSGAIII
from platypus import SMPSO as SMPSO_platypus

from mobo.optimizer import OptimizerCore


class NSGA2(OptimizerCore):
    def __init__(self,
                 function: Callable = None,
                 x_range: list = [0, 1],
                 n_variables: int = 1,
                 n_objectives: int = 2) -> None:

        super().__init__(function=function,
                         x_range=x_range,
                         n_variables=n_variables,
                         n_objectives=n_objectives)
        return

    def run(self, n_generations=100) -> list:
        try:
            algorithm = NSGAII(self.problem)
            algorithm.run(n_generations)
            self.optimizer_result = algorithm.result
            return self.optimizer_result
        except RuntimeError as err:
            raise err


class NSGA3(OptimizerCore):
    def __init__(self,
                 function: Callable = None,
                 x_range: list = [0, 1],
                 n_variables: int = 1,
                 n_objectives: int = 2) -> None:

        super().__init__(function=function,
                         x_range=x_range,
                         n_variables=n_variables,
                         n_objectives=n_objectives)
        return

    def run(self, n_generations=100, divisions_outer=1) -> list:
        try:
            algorithm = NSGAIII(self.problem,
                                divisions_outer=divisions_outer)
            algorithm.run(n_generations)
            self.optimizer_result = algorithm.result
            return self.optimizer_result
        except RuntimeError as err:
            raise err


class SMPSO(OptimizerCore):
    def __init__(self,
                 function: Callable = None,
                 x_range: list = [0, 1],
                 n_variables: int = 1,
                 n_objectives: int = 2) -> None:

        super().__init__(function=function,
                         x_range=x_range,
                         n_variables=n_variables,
                         n_objectives=n_objectives)
        return

    def run(self,
            max_iterations=100,
            swarm_size=100,
            leader_size=100,
            mutation_probability=0.1,
            mutation_perturbation=0.5,
            mutate=None,) -> list:
        try:
            algorithm = SMPSO_platypus(
                self.problem,
                max_iterations=max_iterations,
                swarm_size=swarm_size,
                leader_size=leader_size,
                mutation_probability=mutation_probability,
                mutation_perturbation=mutation_perturbation)
            algorithm.run(max_iterations)
            self.optimizer_result = algorithm.result._contents
            return self.optimizer_result
        except RuntimeError as err:
            raise err
