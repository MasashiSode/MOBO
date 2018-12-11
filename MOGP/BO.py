import MOGP
import numpy as np
# import multiprocessing as mp
import pygmo
# from pygmo.problem import base
import sklearn.gaussian_process as gp
from sklearn.cluster import KMeans as km
import multiprocessing as mp
from matplotlib import pyplot as plt


class MultiObjectiveBayesianOptimization(object):
    def __init__(self):
        self.mogp = None
        self.n_multiprocessing = mp.cpu_count()

        pass

    def set_train_data(self, x_observed, y_observed, n_cons=0):
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
        if n_cons is not 0:
            self.flag_cons = True

        self.y_observed = y_observed
        self.x_observed = x_observed
        self.n_features = x_observed.shape[0]
        self.n_params = x_observed.shape[1]
        self.n_obj = y_observed.shape[1] - n_cons
        self.n_cons = n_cons
        self.bounds = ([min(x_observed[:, i]) for i in range(0, x_observed.shape[1])],
                       [max(x_observed[:, i]) for i in range(0, x_observed.shape[1])])
        self.optimum_direction = -1 * np.ones(self.n_obj)
        return

    def set_optimum_direction(self, direction_list):
        '''
        Args:
            direction_list (list): list of 1 and -1 which expresses the direction of optimum

        Examples::

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
            mogp.set_number_of_cpu_core(cpu)
        '''

        if type(n_multiprocessing) is not int:
            raise ValueError
        self.n_multiprocessing = n_multiprocessing
        return

    def train_GPModel(self, kernel=gp.kernels.Matern()):
        '''
        Examples::

            mobo = MOGP.MultiObjectiveBayesianOptimization()
            mobo.set_train_data(x_observed, y_observed)
            mobo.train_GPModel()
        '''
        self.mogp = MOGP.MOGP()
        self.mogp.set_train_data(
            self.x_observed, self.y_observed, n_cons=self.n_cons)
        self.mogp.set_optimum_direction(self.optimum_direction)
        self.mogp.set_number_of_cpu_core(self.n_multiprocessing)
        self.mogp.train()
        print('training done.')
        return

    def run_moga(self, size=48, gen=100):
        '''
        runs multi-objective genetic algorithm using gaussian process regression.
        objective function is Expected Improvement.

        Args:
            size (int): population size (default=48)
            gen (int): generation size (default=100)

        Examples::

            mobo = MOGP.MultiObjectiveBayesianOptimization()
            mobo.set_train_data(x_observed, y_observed)
            mobo.train_GPModel()
        '''

        self.prob = pygmo.problem(BayesianOptimizationProblem(self.mogp))
        self.pop = pygmo.population(self.prob, size=size)
        self.algo = pygmo.algorithm(pygmo.nsga2(gen=gen))
        self.pop = self.algo.evolve(self.pop)

        self.non_dominated_fronts, self.domination_list, self.domination_count, self.non_domination_ranks = pygmo.fast_non_dominated_sorting(
            self.pop.get_f())
        print('moga done.')
        return

    def run_kmeans(self, n_clusters=8, init='k-means++', n_init=10, max_iter=300,
                   tol=0.0001, precompute_distances='auto', verbose=0,
                   random_state=None, copy_x=True, n_jobs=1):
        '''
        clustering parate front solutions by k-means

        Args:
            n_clusters : int, optional, default: 8
                The number of clusters to form as well as the number of
                centroids to generate.

            init : {'k-means++', 'random' or an ndarray}
                Method for initialization, defaults to 'k-means++':
                'k-means++' : selects initial cluster centers for k-mean
                clustering in a smart way to speed up convergence. See section
                Notes in k_init for more details.
                'random': choose k observations (rows) at random from data for
                the initial centroids.
                If an ndarray is passed, it should be of shape (n_clusters, n_features)
                and gives the initial centers.

            n_init : int, default: 10
                Number of time the k-means algorithm will be run with different
                centroid seeds. The final results will be the best output of
                n_init consecutive runs in terms of inertia.

            max_iter : int, default: 300
                Maximum number of iterations of the k-means algorithm for a
                single run.

            tol : float, default: 1e-4
                Relative tolerance with regards to inertia to declare convergence

            precompute_distances : {'auto', True, False}
                Precompute distances (faster but takes more memory).
                'auto' : do not precompute distances if n_samples * n_clusters > 12
                million. This corresponds to about 100MB overhead per job using
                double precision.
                True : always precompute distances
                False : never precompute distances

            verbose : int, default 0
                Verbosity mode.

            random_state : int, RandomState instance or None (default)
                Determines random number generation for centroid initialization. Use
                an int to make the randomness deterministic.

            copy_x : boolean, optional
                When pre-computing distances it is more numerically accurate to center
                the data first.  If copy_x is True (default), then the original data is
                not modified, ensuring X is C-contiguous.  If False, the original data
                is modified, and put back before the function returns, but small
                numerical differences may be introduced by subtracting and then adding
                the data mean, in this case it will also not ensure that data is
                C-contiguous which may cause a significant slowdown.

            n_jobs : int or None, optional (default=None)
                The number of jobs to use for the computation. This works by computing
                each of the n_init runs in parallel.
                ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
                ``-1`` means using all processors.

            algorithm : "auto", "full" or "elkan", default="auto"
                K-means algorithm to use. The classical EM-style algorithm is "full".
                The "elkan" variation is more efficient by using the triangle
                inequality, but currently doesn't support sparse data. "auto" chooses
                "elkan" for dense data and "full" for sparse data.

        Examples::

            mobo = MOGP.MultiObjectiveBayesianOptimization()
            mobo.set_train_data(x_observed, y_observed)
            mobo.train_GPModel()

        See Also:
            https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
        '''

        self.kmeans_clustering = km(
            n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter,
            tol=tol, precompute_distances=precompute_distances, verbose=verbose,
            random_state=random_state, copy_x=copy_x, n_jobs=n_jobs)

        X = self.pop.get_x()
        Y = self.pop.get_f()
        X = np.array(X[tuple(self.non_dominated_fronts)])
        Y = np.array(Y[tuple(self.non_dominated_fronts)])
        X = np.concatenate((X, Y), axis=1)
        self.kmeans_clustering.fit(X)
        self.kmeans_centroids = self.kmeans_clustering.cluster_centers_
        print('kmeans fitting done.')
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
        return -ei

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

    mogp = MOGP.MOGP()
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

    prob = pygmo.problem(my_mo_problem(mogp))
    print(prob)
    pop = pygmo.population(prob, size=40)
    algo = pygmo.algorithm(pygmo.nsga2(gen=10))
    pop = algo.evolve(pop)
    # pop.plot_pareto_fronts()
    print(pop)
    ax = pygmo.plot_non_dominated_fronts(pop.get_f())
    plt.show()
