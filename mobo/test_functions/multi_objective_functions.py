import array
import random
import json

import numpy

from math import sqrt

from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator
from deap import tools


def uniform(low, up, size=None):
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]


class NSGA2():
    def __init__(self, evaluation_function=None,
                 bound_low=0.0, bound_up=1.0,
                 n_dimension=30, mu=24,
                 gen=100, cxpb=0.9,
                 seed=9):
        random.seed(seed)
        self.toolbox = base.Toolbox()
        self.bound_low = bound_low
        self.bound_up = bound_up
        self.n_dimension = n_dimension
        self.mu = mu
        self.gen = gen
        self.cxpb = cxpb
        self.evaluation_function = evaluation_function

        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
        creator.create("Individual", array.array, typecode='d',
                       fitness=creator.FitnessMin)

        self.toolbox.register("attr_float", uniform,
                              self.bound_low, self.bound_up,
                              self.n_dimension)
        self.toolbox.register("individual", tools.initIterate,
                              creator.Individual, self.toolbox.attr_float)
        self.toolbox.register("population", tools.initRepeat,
                              list, self.toolbox.individual)

        self.toolbox.register("evaluate", self.evaluation_function)
        self.toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                              low=self.bound_low, up=self.bound_up, eta=20.0)
        self.toolbox.register("mutate", tools.mutPolynomialBounded,
                              low=self.bound_low, up=self.bound_up, eta=20.0, indpb=1.0 / self.n_dimension)
        self.toolbox.register("select", tools.selNSGA2)

        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", numpy.mean, axis=0)
        self.stats.register("std", numpy.std, axis=0)
        self.stats.register("min", numpy.min, axis=0)
        self.stats.register("max", numpy.max, axis=0)
        self.logbook = tools.Logbook()
        self.logbook.header = "gen", "evals", "std", "min", "avg", "max"
        self.pop = self.toolbox.population(n=self.mu)

    def run(self):
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in self.pop if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # This is just to assign the crowding distance to the individuals
        # no actual selection is done
        self.pop = self.toolbox.select(self.pop, len(self.pop))

        record = self.stats.compile(self.pop)
        self.logbook.record(gen=0, evals=len(invalid_ind), **record)
        print(self.logbook.stream)

        # Begin the generational process
        for gen in range(1, self.gen):
            # Vary the population
            offspring = tools.selTournamentDCD(self.pop, len(self.pop))
            offspring = [self.toolbox.clone(ind) for ind in offspring]

            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if random.random() <= self.cxpb:
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
            self.pop = self.toolbox.select(self.pop + offspring, self.mu)
            record = self.stats.compile(self.pop)
            self.logbook.record(gen=gen, evals=len(invalid_ind), **record)
            print(self.logbook.stream)

        print("Final population hypervolume is %f" %
              hypervolume(self.pop, [11.0, 11.0]))
        return self.pop, self.logbook


if __name__ == "__main__":
    # with open("pareto_front/zdt1_front.json") as optimal_front_data:
    #     optimal_front = json.load(optimal_front_data)
    # # Use 500 of the 1000 points in the json file
    # optimal_front = sorted(optimal_front[i]
    #                        for i in range(0, len(optimal_front), 2))

    # pop, stats = main()
    nsga2 = NSGA2(evaluation_function=benchmarks.zdt1)
    pop, stats = nsga2.run()
    pop.sort(key=lambda x: x.fitness.values)

    print(stats)
    # print("Convergence: ", convergence(pop, optimal_front))
    # print("Diversity: ", diversity(pop, optimal_front[0], optimal_front[-1]))

    import matplotlib.pyplot as plt
    # import numpy

    front = numpy.array([ind.fitness.values for ind in pop])
    # optimal_front = numpy.array(optimal_front)
    # plt.scatter(optimal_front[:, 0], optimal_front[:, 1], c="r")
    plt.scatter(front[:, 0], front[:, 1], c="b")
    plt.axis("tight")
    plt.show()
