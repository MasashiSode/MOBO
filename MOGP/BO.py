import copy
import multiprocessing as mp
import numpy as np
import sklearn.gaussian_process as gp
from sklearn.cluster import KMeans as km
import pygmo
from pyDOE import lhs
from . import GP


class MultiObjectiveBayesianOptimization(object):
    def __init__(self):
        self.mogp = None
        self.n_multiprocessing = mp.cpu_count()

    def set_train_data(self, x_observed, y_observed, n_cons=0):
        """
        Args:
            x_observed: np.array (n_samples, n_params)
            y_observed: np.array (n_samples, n_obj + n_cons)
        Example::
            mobo = MOGP.MultiObjectiveBayesianOptimization()
            mobo.set_train_data(x_observed, y_observed)
        """
        if not isinstance(x_observed, np.ndarray):
            raise ValueError
        if not isinstance(y_observed, np.ndarray):
            raise ValueError
        if n_cons is not 0:
            self.flag_cons = True

        self.x_observed_org_coor = copy.deepcopy(x_observed)
        self.y_observed_org_coor = copy.deepcopy(y_observed)

        self.x_observed = copy.deepcopy(x_observed)
        self.y_observed = copy.deepcopy(y_observed)

        # delete duplicate values
        input_observed = np.concatenate(
            [self.x_observed, self.y_observed], axis=1)
        input_observed, indeices = \
            np.unique(input_observed, axis=0, return_inverse=True)
        input_observed = input_observed[indeices]
        self.x_observed = input_observed[:, 0:self.x_observed.shape[1]]
        self.y_observed = \
            input_observed[:,
                           self.x_observed.shape[1]:
                           self.x_observed.shape[1] +
                           self.y_observed.shape[1] + 1]

        self.n_features = self.x_observed.shape[0]
        self.n_params = self.x_observed.shape[1]
        self.n_obj_cons = self.y_observed.shape[1]
        self.n_obj = self.y_observed.shape[1] - n_cons
        self.n_cons = n_cons

        self.x_observed_max = np.zeros(self.n_params)
        self.x_observed_min = np.zeros(self.n_params)
        self.y_observed_max = np.zeros(self.n_obj_cons)
        self.y_observed_min = np.zeros(self.n_obj_cons)

        # normalization
        for i in range(0, self.n_params):
            self.x_observed_max[i] = max(copy.deepcopy(self.x_observed[:, i]))
            self.x_observed_min[i] = min(copy.deepcopy(self.x_observed[:, i]))
            self.x_observed[:, i] = \
                (self.x_observed[:, i] - self.x_observed_min[i]) / \
                (self.x_observed_max[i] - self.x_observed_min[i])

        for i in range(0, self.n_obj):
            self.y_observed_max[i] = max(copy.deepcopy(self.y_observed[:, i]))
            self.y_observed_min[i] = min(copy.deepcopy(self.y_observed[:, i]))
            self.y_observed[:, i] = \
                (self.y_observed[:, i] - self.y_observed_min[i]) / \
                (self.y_observed_max[i] - self.y_observed_min[i])

        for i in range(self.n_obj, self.n_obj_cons):
            self.y_observed_max[i] = max(copy.deepcopy(self.y_observed[:, i]))
            self.y_observed_min[i] = min(copy.deepcopy(self.y_observed[:, i]))
            # self.y_observed[:, i] = \
            #     self.y_observed[:, i] / \
            #     abs(self.y_observed_max[i] - self.y_observed_min[i])

        self.bounds = \
            ([self.x_observed_min[i] for i in range(0, x_observed.shape[1])],
             [self.x_observed_max[i] for i in range(0, x_observed.shape[1])])
        self.optimum_direction = -1 * np.ones(self.n_obj)
        return

    def set_optimum_direction(self, direction_list):
        """
        Args:
            direction_list (list): list of 1 and -1
                which expresses the direction of optimum
        Examples::
            direction_list = [-1, 1, -1]
            mobo.set_optimum_direction(direction_list)
        """
        if type(direction_list) is not list:
            raise ValueError
        if len(direction_list) != self.n_obj:
            print('len(direction_list != n_obj')
            raise ValueError

        self.optimum_direction = direction_list
        return

    def set_number_of_cpu_core(self, n_multiprocessing=mp.cpu_count()):
        """
        sets number of cpu cores you use in
        multi-objective EI calculation (default all cpu)
        Examples::
            cpu = 4
            mobo.set_number_of_cpu_core(cpu)
        """

        if type(n_multiprocessing) is not int:
            raise ValueError
        self.n_multiprocessing = n_multiprocessing
        return

    def train_GPModel(self, kernel=gp.kernels.Matern()):
        """
        Examples::
            mobo = MOGP.MultiObjectiveBayesianOptimization()
            mobo.set_train_data(x_observed, y_observed)
            mobo.train_GPModel()
        """
        print('training running...')
        self.mogp = GP.GaussianProcess()
        self.mogp.set_train_data(
            self.x_observed, self.y_observed, n_cons=self.n_cons)
        self.mogp.set_optimum_direction(self.optimum_direction)
        self.mogp.set_number_of_cpu_core(self.n_multiprocessing)

        self.mogp.train()
        print('training done.')
        return

    def run_moga(self, size=48, gen=100, m=0.03):
        """
        runs multi-objective genetic algorithm
        using gaussian process regression.
        objective function is Expected Improvement.
        Args:
            size (int): population size (default=48)
            gen (int): generation size (default=100)
        Examples::
            mobo = MOGP.MultiObjectiveBayesianOptimization()
            mobo.set_train_data(x_observed, y_observed)
            mobo.train_GPModel()
        """
        print('moga running...')
        self.prob = pygmo.problem(BayesianOptimizationProblem(self.mogp))
        self.pop = pygmo.population(self.prob, size=size)
        self.algo = pygmo.algorithm(pygmo.nsga2(gen=gen, m=m))
        self.pop = self.algo.evolve(self.pop)

        self.non_dominated_fronts, self.domination_list, \
            self.domination_count, self.non_domination_ranks = \
            pygmo.fast_non_dominated_sorting(self.pop.get_f())
        print('moga done.')
        return

    def run_kmeans(self, n_clusters=8, init='k-means++',
                   n_init=10, max_iter=300,
                   tol=0.0001, precompute_distances='auto', verbose=0,
                   random_state=None, copy_x=True, n_jobs=1):
        """
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
                If an ndarray is passed, it should be of shape
                (n_clusters, n_features)
                and gives the initial centers.
            n_init : int, default: 10
                Number of time the k-means algorithm will be run with different
                centroid seeds. The final results will be the best output of
                n_init consecutive runs in terms of inertia.
            max_iter : int, default: 300
                Maximum number of iterations of the k-means algorithm for a
                single run.
            tol : float, default: 1e-4
                Relative tolerance with regards
                to inertia to declare convergence
            precompute_distances : {'auto', True, False}
                Precompute distances (faster but takes more memory).
                'auto' : do not precompute distances
                if n_samples * n_clusters > 12 million.
                This corresponds to about 100MB overhead per job using
                double precision.
                True : always precompute distances
                False : never precompute distances
            verbose : int, default 0
                Verbosity mode.
            random_state : int, RandomState instance or None (default)
                Determines random number generation
                for centroid initialization. Use
                an int to make the randomness deterministic.
            copy_x : boolean, optional
                When pre-computing distances
                it is more numerically accurate to center
                the data first.  If copy_x is True (default),
                then the original data is
                not modified, ensuring X is C-contiguous.
                If False, the original data
                is modified, and put back before the function returns,
                but small numerical differences may be
                introduced by subtracting and then adding
                the data mean, in this case
                it will also not ensure that data is
                C-contiguous which may cause a significant slowdown.
            n_jobs : int or None, optional (default=None)
                The number of jobs to use for the computation.
                This works by computing each of the n_init runs in parallel.
                ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
                context.
                ``-1`` means using all processors.
            algorithm : "auto", "full" or "elkan", default="auto"
                K-means algorithm to use.
                The classical EM-style algorithm is "full".
                The "elkan" variation is more efficient by using the triangle
                inequality, but currently doesn't support sparse data.
                "auto" chooses
                "elkan" for dense data and "full" for sparse data.
        Examples::
            mobo = MOGP.MultiObjectiveBayesianOptimization()
            mobo.set_train_data(x_observed, y_observed)
            mobo.train_GPModel()
        See Also:
            https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
        """

        print('k-means running...')
        self.kmeans_clustering = km(
            n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter,
            tol=tol, precompute_distances=precompute_distances,
            verbose=verbose,
            random_state=random_state, copy_x=copy_x, n_jobs=n_jobs)

        X = self.pop.get_x()
        Y = self.pop.get_f()
        # print(self.non_dominated_fronts[0])
        # indx = self.non_dominated_fronts[:, 0]
        # X = X[tuple(indx)]
        # Y = Y[tuple(self.non_dominated_fronts)]
        X = np.concatenate((X, Y), axis=1)
        self.kmeans_clustering.fit(X)
        self.kmeans_centroids = self.kmeans_clustering.cluster_centers_
        self.kmeans_centroids_original_coor_x = \
            self.kmeans_centroids[:, 0:self.x_observed.shape[1]] * \
            (self.x_observed_max - self.x_observed_min) + self.x_observed_min
        print('k-means fitting done.')
        return

    def run_mobo(self, func=None, args=[],
                 n_dv=0, n_obj_cons=0,
                 n_init_lhs_samples=24,
                 n_iter=10, n_new_ind=16,
                 ga_pop_size=100, ga_gen=50, n_cons=0, mutation=0.03):
        """
        runs multi-objective bayesian optimization
        Args:
            func:
                multi objective function you want to minimize.
                y = f(x, agrs=[])
                0 <= x[i] <=1
            args (list):
                parameters for the function.
            n_dv (int):
                number of design variabels
            n_obj (int):
                number of objective functions
                n_init_lhs_sampling (int):
                initial population of bayesian optimization
            n_iter (int):
                number of iteration of bayesian optimization
                n_new_ind (int): number of new indivisuals in each iteration.
            ga_pop_size (int):
                population size of multi
                objective genetic algorithm (NSGA2)
                ga_pop_size must be multiply number of four.
            ga_gen (int):
                generation number of
                multi objective genetic algorithm (NSGA2)
            n_cons (int):
                number of constraints functions
        """

        # latin hyper cube sampling
        x_observed = lhs(n_dv, samples=n_init_lhs_samples)
        # y_observed = np.zeros((n_init_lhs_samples, n_obj_cons))
        y_observed = np.array(func(x=x_observed, args=args)).T

        for i in range(0, n_iter):
            print('\n--- iter: ', i, '/', n_iter - 1, '---')

            self.set_train_data(x_observed, y_observed, n_cons=n_cons)

            # training Gaussian Process regression
            self.train_GPModel()

            # multi-objective optimization(nsga2) on surrogate model
            self.run_moga(size=ga_pop_size, gen=ga_gen, m=mutation)

            # clustering solutions
            self.run_kmeans(n_clusters=n_new_ind, n_jobs=-1, n_init=20)

            # evaluate new points
            print('function evaluation')
            new_indv_x = self.kmeans_centroids_original_coor_x
            new_indv_y = np.array(func(x=new_indv_x, args=args)).T
            print('function evaluation done.')

            # update observed values
            x_observed = np.concatenate([x_observed, new_indv_x], axis=0)
            y_observed = np.concatenate([y_observed, new_indv_y], axis=0)
            self.x_observed_org_coor = copy.deepcopy(x_observed)
            self.y_observed_org_coor = copy.deepcopy(y_observed)


class BayesianOptimizationProblem():
    """
    pyGMO wrapper for gaussian process regression
    """

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
    #     return
