import collections
import copy
import multiprocessing as mp
import warnings
import numpy as np
from scipy.stats import norm
import sklearn.gaussian_process as gp


class MpHelper(object):
    """
    helper function for multiprocessing to call class method

    Note:
        japanese site
        http://aflc.hatenablog.com/entry/20110825/1314366131

    """

    def __init__(self, cls, mtd_name):
        self.cls = cls
        self.mtd_name = mtd_name

    def __call__(self, *args, **kwargs):
        return getattr(self.cls, self.mtd_name)(*args, **kwargs)


class GaussianProcess():
    """
    Multi-Objective Gaussian Process core class

    Note:
        Set parameters first before train

        (https://thuijskens.github.io/2016/12/29/bayesian-optimisation/)

    Example::

        mogp = MOGP.MOGP()
        mogp.set_train_data(x_observed, y_observed)
        mogp.train()
        x = np.array([[-5, -5]])
        print(mogp.predict(x))

        x = np.array([[-4.9, -4.9]])
        print(mogp.expected_improvement(x))

    """

    def __init__(self):
        self.y_observed = None
        self.x_observed = None
        self.n_features = 0
        self.n_params = 0
        self.n_obj = 0
        self.n_cons = 0
        self.n_obj_cons = 0
        self.gpr = None
        self.bounds = 0
        self.flag_cons = False
        self.n_multiprocessing = mp.cpu_count()
        self.optimum_direction = -1 * np.ones(self.n_obj)
        return

    def set_train_data(self, x_observed, y_observed, n_cons=0):
        """
        Args:
            x_observed: np.array (n_samples, n_params)
            y_observed: np.array (n_samples, n_obj)

        Example::

            mogp = MOGP.MOGP()
            mogp.set_train_data(x_observed, y_observed)

        """
        if not isinstance(x_observed, np.ndarray):
            raise ValueError
        if not isinstance(y_observed, np.ndarray):
            raise ValueError
        if n_cons is not 0:
            self.flag_cons = True

        self.y_observed = copy.deepcopy(y_observed)
        self.x_observed = copy.deepcopy(x_observed)
        self.n_features = x_observed.shape[0]
        self.n_params = x_observed.shape[1]
        self.n_obj_cons = y_observed.shape[1]
        self.n_obj = y_observed.shape[1] - n_cons
        self.n_cons = n_cons

        self.x_observed_max = np.zeros(self.n_params)
        self.x_observed_min = np.zeros(self.n_params)
        self.y_observed_max = np.zeros(self.n_obj)
        self.y_observed_min = np.zeros(self.n_obj)

        # normalization
        for i in range(0, self.n_params):
            self.x_observed_max[i] = max(self.x_observed[:, i])
            self.x_observed_min[i] = min(self.x_observed[:, i])
            self.x_observed[:, i] = \
                (self.x_observed[:, i] - self.x_observed_min[i]) / \
                (self.x_observed_max[i] - self.x_observed_min[i])

        for i in range(0, self.n_obj):
            self.y_observed_max[i] = max(self.y_observed[:, i])
            self.y_observed_min[i] = min(self.y_observed[:, i])
            self.y_observed[:, i] = \
                (self.y_observed[:, i] - self.y_observed_min[i]) / \
                (self.y_observed_max[i] - self.y_observed_min[i])

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
            mogp.set_optimum_direction(direction_list)
        """

        if isinstance(direction_list, collections.Iterable) is False:
            print(direction_list, 'is not iterable')

        if len(direction_list) != self.n_obj:
            print('len(direction_list != n_obj')
            raise ValueError

        self.optimum_direction = direction_list
        return

    def set_number_of_cpu_core(self, n_multiprocessing):
        if type(n_multiprocessing) is not int:
            raise ValueError
        self.n_multiprocessing = n_multiprocessing
        return

    def train(self, kernel=gp.kernels.Matern()):
        """
        trains Gaussian process for regression

        Note:
            multi-objective optimization (n_obj > 1) is also available.

        Args:
            kernel: kernel implemented in sklearn.gp
                (default: gp.kernels.Matern())

        Example::

            mogp = MOGPOpt.MOGP()
            mogp.set_train_data(x_observed, y_observed)
            mogp.train()
        """
        if self.y_observed is None or \
                self.x_observed is None:
            print('set_train_data first')
            raise ValueError
        else:
            pass

        if self.n_obj == 1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.gpr = \
                    gp.GaussianProcessRegressor(
                        kernel=kernel, random_state=5).fit(
                            self.x_observed,
                            self.y_observed)
        else:
            # manager.list shares list in the multi-processing
            with mp.Manager() as manager:
                self.gpr = manager.list([None] * self.n_obj_cons)
                for i_obj in range(0, self.n_obj_cons):
                    self.gpr[i_obj] = \
                        gp.GaussianProcessRegressor(
                            kernel=kernel, random_state=0,
                            n_restarts_optimizer=0)

                with mp.Pool(self.n_multiprocessing) as p:
                    # p = mp.Pool(self.n_multiprocessing)
                    p.map(MpHelper(self, 'wrapper_mp'),
                          range(0, self.n_obj_cons))
                    p.close()
                    p.join()
                    self.gpr = [x for x in self.gpr]

                manager.shutdown()
        return self.gpr

    def wrapper_mp(self, i_obj):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.gpr[i_obj] = self.gpr[i_obj].fit(
                self.x_observed,
                self.y_observed[:, i_obj])
        return

    def predict_original_coor(self, x):
        """
        Note:
            use it after training

        Args:
            x: np.array, size = [n_input, n_params]

        Returns:
            mu, sigma (float or list): mean and std size = [2, n_obj]

        Example::

            x = np.array([-5, -5])
            mu, sigma = mogp.predict(x)
            print(mu, sigma)

        """
        print('start predict')
        x = x.reshape(-1, self.n_params)
        if self.gpr is None:
            print('train first')
            raise

        if self.n_obj == 1:
            mu, sigma = self.gpr.predict(x, return_std=True)
        else:
            mu = np.zeros(self.n_obj_cons)
            sigma = np.zeros(self.n_obj_cons)
            for i_obj in range(0, self.n_obj_cons):
                temp1, temp2 = \
                    self.gpr[i_obj].predict(x, return_std=True)
                mu[i_obj] = temp1[0]
                sigma[i_obj] = temp2[0]

        mu = mu * (self.x_observed_max - self.x_observed_min) + \
            self.x_observed_min
        return np.array([mu, sigma])

    def predict_standarize_value(self, x):
        """
        Note:
            use it after training

        Args:
            x: np.array, size = [n_input, n_params]

        Returns:
            mu, sigma (float or list): mean and std size = [2, n_obj]

        Example::

            x = np.array([-5, -5])
            mu, sigma = mogp.predict(x)
            print(mu, sigma)

        """
        print('start predict')
        x = x.reshape(-1, self.n_params)
        if self.gpr is None:
            print('train first')
            raise

        if self.n_obj == 1:
            mu, sigma = self.gpr.predict(x, return_std=True)
        else:
            mu = np.zeros(self.n_obj)
            sigma = np.zeros(self.n_obj)
            for i_obj in range(0, self.n_obj):
                temp1, temp2 = \
                    self.gpr[i_obj].predict(x, return_std=True)
                mu[i_obj] = temp1[0]
                sigma[i_obj] = temp2[0]
        return np.array([mu, sigma])

    def expected_improvement(self, x):
        """ expected_improvement
        Expected improvement function.

        Arguments:
            x (array-like): shape = [n_samples, n_params]

        Examples::

            x = np.array([[-4.9, -4.9]])
            ei = mogp.expected_improvement(x)
            print(ei)

        """
        x = x.reshape(-1, self.n_params)

        if self.flag_cons is True:
            constrained_ei_x = self.constrained_EI(x)
            return constrained_ei_x

        mu = np.zeros(self.n_obj)
        sigma = np.zeros(self.n_obj)
        ei_x = np.zeros(self.n_obj)
        self.f_ref = np.zeros(self.n_obj)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i_obj in range(0, self.n_obj):
                temp1, temp2 = \
                    self.gpr[i_obj].predict(x, return_std=True)
                mu[i_obj] = temp1[0]
                sigma[i_obj] = temp2[0]

                if self.optimum_direction[i_obj] == 1:
                    self.f_ref[i_obj] = np.max(
                        self.y_observed[:, i_obj])
                else:
                    self.f_ref[i_obj] = np.min(
                        self.y_observed[:, i_obj])

                # In case sigma equals zero
                with np.errstate(divide='ignore'):
                    Z = (self.f_ref[i_obj] - mu[i_obj]) / sigma[i_obj]
                    ei_x[i_obj] = \
                        (self.f_ref[i_obj] - mu[i_obj]) * \
                        norm.cdf(Z) + sigma[i_obj] * norm.pdf(Z)
                ei_x[sigma[i_obj] == 0.0] = 0.0
        return ei_x

    def expected_hypervolume_improvement(self, x):
        """ expected_hypervolume_improvement
        Expected hypervolume improvement function.

        Note:
            under construction!

        """
        # mu = np.zeros(self.n_obj)
        # sigma = np.zeros(self.n_obj)
        # ei_x = np.zeros(self.n_obj)
        # self.f_ref = np.zeros(self.n_obj)

        # for i_obj in range(0, self.n_obj):
        #     with np.errstate(divide='ignore'):
        #         Z = (self.f_ref[i_obj] - mu[i_obj]) / sigma[i_obj]
        #         ei_x[i_obj] = \
        #             (self.f_ref[i_obj] - mu[i_obj]) * \
        #             norm.cdf(Z) + sigma[i_obj] * norm.pdf(Z)
        #     ei_x[sigma[i_obj] == 0.0] = 0.0

        print('this funcion is under construction, use another fucntion')
        raise
        return

    def probability_of_feasibility(self, x):
        """
        calculates the probability of feasibility.
        uses probability of g(x) <= 0. g > 0 is infeasible.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pof = np.ones(self.n_cons)
            mu = np.zeros(self.n_cons)
            sigma = np.zeros(self.n_cons)
            i = 0
            for i_cons in range(self.n_obj, self.n_obj_cons):
                temp1, temp2 = self.gpr[i_cons].predict(x, return_std=True)
                mu[i] = temp1[0]
                sigma[i] = temp2[0]

                with np.errstate(divide='ignore'):
                    # Z = - mu[i] / sigma[i]
                    # pof[i] = norm.cdf(Z)
                    pof[i] = norm.cdf(0, loc=mu[i], scale=sigma[i])
                pof[sigma[i] == 0] = 0
                # print(mu[i], sigma[i], pof[i])
                i = i + 1
        return pof

    def constrained_EI(self, x):
        """
        uses probability of g(x) <= 0. g > 0 is infeasible.
        """

        mu = np.zeros(self.n_obj)
        sigma = np.zeros(self.n_obj)
        ei = np.zeros(self.n_obj)
        self.f_ref = np.zeros(self.n_obj)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i_obj in range(0, self.n_obj):
                temp1, temp2 = \
                    self.gpr[i_obj].predict(x, return_std=True)
                mu[i_obj] = temp1[0]
                sigma[i_obj] = temp2[0]

                if self.optimum_direction[i_obj] == 1:
                    self.f_ref[i_obj] = np.max(
                        self.y_observed[:, i_obj])
                else:
                    self.f_ref[i_obj] = np.min(
                        self.y_observed[:, i_obj])

                # In case sigma equals zero
                with np.errstate(divide='ignore'):
                    Z = (self.f_ref[i_obj] - mu[i_obj]) / sigma[i_obj]
                    ei[i_obj] = \
                        (self.f_ref[i_obj] - mu[i_obj]) * \
                        norm.cdf(Z) + sigma[i_obj] * norm.pdf(Z)
                ei[sigma[i_obj] == 0.0] = 0.0

        pof = self.probability_of_feasibility(x)
        pof_all = np.prod(pof)
        cei = ei * pof_all
        # print(pof_all, cei)
        return cei
