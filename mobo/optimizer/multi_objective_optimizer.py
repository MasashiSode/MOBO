import array
import random

import numpy as np

from deap import base
from deap.benchmarks.tools import hypervolume
from deap import creator
from deap import tools


class NSGA2():
    def __init__(self,
                 evaluation_function=None,
                 bound_low=0.0,
                 bound_up=1.0,
                 n_design_variables_dimension=30,
                 n_population=100,
                 n_generation=250,
                 crossover_probability=0.9,
                 random_seed=9):
        self.random_seed = random_seed
        random.seed(self.random_seed)

        self.toolbox = base.Toolbox()
        self.evaluation_function = evaluation_function
        self.bound_low = bound_low
        self.bound_up = bound_up
        self.n_design_variables_dimension =\
            n_design_variables_dimension
        self.n_population = n_population
        self.n_generation = n_generation
        self.crossover_probability = crossover_probability

    def setup(self):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
        creator.create("Individual", array.array, typecode='d',
                       fitness=creator.FitnessMin)

        self.toolbox.register("attr_float", self.uniform,
                              self.bound_low, self.bound_up,
                              self.n_design_variables_dimension)
        self.toolbox.register("individual", tools.initIterate,
                              creator.Individual, self.toolbox.attr_float)
        self.toolbox.register("population", tools.initRepeat,
                              list, self.toolbox.individual)

        if self.evaluation_function:
            self.toolbox.register("evaluate", self.evaluation_function)

        self.toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                              low=self.bound_low, up=self.bound_up, eta=20.0)
        self.toolbox.register("mutate", tools.mutPolynomialBounded,
                              low=self.bound_low, up=self.bound_up, eta=20.0,
                              indpb=1.0 / self.n_design_variables_dimension)
        self.toolbox.register("select", tools.selNSGA2)

        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean, axis=0)
        self.stats.register("std", np.std, axis=0)
        self.stats.register("min", np.min, axis=0)
        self.stats.register("max", np.max, axis=0)
        self.logbook = tools.Logbook()
        self.logbook.header = "gen", "evals", "std", "min", "avg", "max"
        self.pop = self.toolbox.population(n=self.n_population)

    def run(self):
        self.setup()
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in self.pop if not ind.fitness.valid]
        fitnesses = list(
            (self.toolbox.map(self.toolbox.evaluate, invalid_ind)))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # This is just to assign the crowding distance to the individuals
        # no actual selection is done
        self.pop = self.toolbox.select(self.pop, len(self.pop))

        record = self.stats.compile(self.pop)
        self.logbook.record(gen=0, evals=len(invalid_ind), **record)
        print(self.logbook.stream)

        # Begin the generational process
        for i_generation in range(1, self.n_generation):
            # Vary the population
            offspring = tools.selTournamentDCD(self.pop, len(self.pop))
            offspring = [self.toolbox.clone(ind) for ind in offspring]

            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if random.random() <= self.crossover_probability:
                    self.toolbox.mate(ind1, ind2)

                self.toolbox.mutate(ind1)
                self.toolbox.mutate(ind2)
                del ind1.fitness.values, ind2.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Select the next generation population
            self.pop = self.toolbox.select(
                self.pop + offspring, self.n_population)
            record = self.stats.compile(self.pop)
            self.logbook.record(
                gen=i_generation, evals=len(invalid_ind), **record)
            print(self.logbook.stream)

        print("Final population hypervolume is %f" %
              hypervolume(self.pop, [11.0, 11.0]))
        return self.pop, self.logbook

    def uniform(self, low, up, size=None):
        try:
            return [random.uniform(a, b) for a, b in zip(low, up)]
        except TypeError:
            return [random.uniform(a, b) for a, b in zip([low] * size,
                                                         [up] * size)]


if __name__ == "__main__":
    from mobo.test_functions import zdt1

    nsga2 = NSGA2(evaluation_function=zdt1,
                  n_design_variables_dimension=30,
                  n_population=24,
                  n_generation=50)
    pop, stats = nsga2.run()
    pop.sort(key=lambda x: x.fitness.values)

    print(stats)

    import matplotlib.pyplot as plt

    front = np.array([ind.fitness.values for ind in pop])

    plt.scatter(front[:, 0], front[:, 1], c="b")
    plt.axis("tight")
    plt.show()
