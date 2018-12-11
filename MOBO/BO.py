import MOBO
import numpy as np
# import multiprocessing as mp
import pygmo as pg
# from pygmo.problem import base
import sklearn.gaussian_process as gp
import multiprocessing as mp
from matplotlib import pyplot as plt


class MultiObjectiveBayesianOptimization(object):
    def __init__(self):
        self.mogp = None
        self.n_multiprocessing = mp.cpu_count()

        pass

    def set_train_data(self, x_observed, y_observed):
        '''
        Args:
            x_observed: np.array (n_samples, n_params)
            y_observed: np.array (n_samples, n_obj)

        Example::

            mogp = MOGPOpt.MOGP()
            mogp.set_train_data(x_observed, y_observed)

        '''
        if not isinstance(x_observed, np.ndarray):
            raise ValueError
        if not isinstance(y_observed, np.ndarray):
            raise ValueError

        self.y_observed = y_observed
        self.x_observed = x_observed
        self.n_features = x_observed.shape[0]
        self.n_params = x_observed.shape[1]
        self.n_obj = y_observed.shape[1]
        self.bounds = [[min(x_observed[0]), min(x_observed[1])],
                       [max(x_observed[0]), max(x_observed[1])]]
        self.optimum_direction = -1 * np.ones(self.n_obj)
        return

    def set_optimum_direction(self, direction_list):
        '''
        Args:
            direction_list (list): list of 1 and -1 which expresses the direction of optimum

        Examles::

            direction_list = [-1, 1, -1]
            mogp.set_optimum_direction(direction_list)
        '''
        if type(direction_list) is not list:
            raise ValueError
        if len(direction_list) != self.n_obj:
            print('len(direction_list != n_obj')
            raise ValueError

        self.optimum_direction = direction_list
        return

    def set_number_of_cpu_core(self, n_multiprocessing=mp.cpu_count()):
        '''
        sets number of cpu cores you use in
        multi-objective EI calculation (default all cpu)

        Examples::

            cpu = 4
            mobo.set_number_of_cpu_core(cpu)

        '''

        if type(n_multiprocessing) is not int:
            raise ValueError
        self.n_multiprocessing = n_multiprocessing
        return

    def train_GPModel(self, kernel=gp.kernels.Matern()):
        '''
        Examples::

            mobo = MOBO.MultiObjectiveBayesianOptimization()
            mobo.set_train_data(x_observed, y_observed)
            mobo.train_GPModel()
        '''
        self.mogp = MOBO.MOGP()
        self.mogp.set_train_data(self.x_observed, self.y_observed)
        self.mogp.set_optimum_direction(self.optimum_direction)
        self.mogp.set_number_of_cpu_core(self.n_multiprocessing)
        self.mogp.train()
        return

    def run_moga(self, size=48, gen=100):
        '''
        runs multi-objective genetic algorithm using gaussian process regression.
        objective function is Expected Improvement.

        Args:
            size (int): population size (default=48)
            gen (int): generation size (default=100)

        Examples::

            mobo = MOBO.MultiObjectiveBayesianOptimization()
            mobo.set_train_data(x_observed, y_observed)
            mobo.train_GPModel()
        '''

        self.prob = pg.problem(BayesianOptimizationProblem(self.mogp))
        self.pop = pg.population(self.prob, size=size)
        self.algo = pg.algorithm(pg.nsga2(gen=gen))
        self.pop = self.algo.evolve(self.pop)
        return


class BayesianOptimizationProblem():
    '''
    pyGMO wrapper for gaussian process regression
    '''

    def __init__(self, mogp):
        self.mogp = mogp
        self.bounds = mogp.bounds

    def fitness(self, x):
        ei = self.mogp.expected_improvement(x)
        return - ei

    def get_bounds(self):
        return self.bounds

    def get_nobj(self):
        return self.mogp.n_obj

    def set_bounds(self, bounds):
        self.bounds = bounds
        return

    # def get_nic(self):
    #     return 2

    # def get_nec(self):
    #     return 4


def ReadInput(InputFile):
    data = np.loadtxt(InputFile, delimiter=",")
    return data


if __name__ == "__main__":
    # mp.freeze_support()
    y_observed = ReadInput('InputObj.csv')
    x_observed = ReadInput('InputVar.csv')

    mogp = MOBO.MOGP()
    mogp.set_train_data(x_observed, y_observed)
    # Mogp.set_number_of_cpu_core(1)
    mogp.train()

    x = np.array([-5, -5])
    mu, sigma = mogp.predict(x)
    print('mu: ', mu)
    print('sigma: ', sigma)

    x = np.array([[-4.9, -4.9]])
    ei = mogp.expected_improvement(x)
    print(ei)

    prob = pg.problem(my_mo_problem(mogp))
    print(prob)
    pop = pg.population(prob, size=40)
    algo = pg.algorithm(pg.nsga2(gen=10))
    pop = algo.evolve(pop)
    # pop.plot_pareto_fronts()
    print(pop)
    ax = pg.plot_non_dominated_fronts(pop.get_f())
    plt.show()
