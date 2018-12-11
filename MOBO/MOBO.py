import numpy as np
import sklearn.gaussian_process as gp
import multiprocessing as mp
from scipy.stats import norm
# from scipy.optimize import minimize
# import copy
# from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
# from scipy import integrate


class MpHelper(object):
    '''
    helper function for multiprocessing to call class method

    Note:
        japaese site
        http://aflc.hatenablog.com/entry/20110825/1314366131

    '''

    def __init__(self, cls, mtd_name):
        self.cls = cls
        self.mtd_name = mtd_name

    def __call__(self, *args, **kwargs):
        return getattr(self.cls, self.mtd_name)(*args, **kwargs)


class MOGP():
    '''
    MOGP (Multi-Objective Gaussian Process) core class

    Note:
        Set parameters first before train

        (https://thuijskens.github.io/2016/12/29/bayesian-optimisation/)

    Example::

        mogp = MOGPOpt.MOGP()
        mogp.set_train_data(x_observed, y_observed)
        mogp.train()
        x = np.array([[-5, -5]])
        print(mogp.predict(x))

        x = np.array([[-4.9, -4.9]])
        print(mogp.expected_improvement(x))

    '''

    def __init__(self):
        self.objective_function_observed = None
        self.design_variables_observed = None
        self.n_features = 0
        self.n_params = 0
        self.n_obj = 0
        self.gpr = None
        self.n_multiprocessing = mp.cpu_count()
        return

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

        self.objective_function_observed = y_observed
        self.design_variables_observed = x_observed
        self.n_features = x_observed.shape[0]
        self.n_params = x_observed.shape[1]
        self.n_obj = y_observed.shape[1]
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

    def set_number_of_cpu_core(self, n_multiprocessing):
        if type(n_multiprocessing) is not int:
            raise ValueError
        self.n_multiprocessing = n_multiprocessing
        return

    def train(self, kernel=gp.kernels.Matern()):
        '''
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
        '''
        if self.objective_function_observed is None or \
                self.design_variables_observed is None:
            print('set_train_data first')
            raise ValueError
        else:
            pass

        if self.n_obj == 1:
            self.gpr = \
                gp.GaussianProcessRegressor(kernel=kernel, random_state=0).fit(
                    self.design_variables_observed,
                    self.objective_function_observed)
        else:
            # manager.list shares list in the multi-processing
            manager = mp.Manager()
            self.gpr = manager.list([None] * self.n_obj)
            for i_obj in range(0, self.n_obj):
                self.gpr[i_obj] = \
                    gp.GaussianProcessRegressor(kernel=kernel, random_state=0)

            p = mp.Pool(self.n_multiprocessing)
            p.map(MpHelper(self, 'wrapper_mp'),
                  range(0, self.n_obj))
            p.close()
            p.join()
        return self.gpr

    def wrapper_mp(self, i_obj):
        self.gpr[i_obj] = self.gpr[i_obj].fit(
            self.design_variables_observed,
            self.objective_function_observed[:, i_obj])
        return

    def predict(self, x):
        '''
        Note:
            use it after training

        Args:
            x: np.array, size = [n_input, n_params]

        Returns:
            mu, sigma (float or list): mean and variance size = [2, n_obj]

        Example::

            x = np.array([-5, -5])
            mu, sigma = mogp.predict(x)
            print(mu, sigma)

        '''
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
            x (array-like): shape = [n_samples, n_hyperparams]

        Examples::

            x = np.array([[-4.9, -4.9]])
            ei = mogp.expected_improvement(x)
            print(ei)

        """

        if self.n_obj == 1:
            mu, sigma = self.gpr.predict(x, return_std=True)

            if self.optimum_direction[0] == 1:
                self.f_ref = np.max(self.objective_function_observed)
                if mu < self.f_ref:
                    return 0
            else:
                self.f_ref = np.min(self.objective_function_observed)
                if mu > self.f_ref:
                    return 0

            # In case sigma equals zero
            with np.errstate(divide='ignore'):
                Z = (mu - self.f_ref) / sigma
                ei_x = \
                    (mu - self.f_ref) * norm.cdf(Z) + sigma * norm.pdf(Z)
                # expected_improvement[sigma == 0.0] == 0.0

            return - 1 * ei_x

        else:
            mu = np.zeros(self.n_obj)
            sigma = np.zeros(self.n_obj)
            ei_x = np.zeros(self.n_obj)
            self.f_ref = np.zeros(self.n_obj)

            for i_obj in range(0, self.n_obj):
                temp1, temp2 = \
                    self.gpr[i_obj].predict(x, return_std=True)
                mu[i_obj] = temp1[0]
                sigma[i_obj] = temp2[0]

                if self.optimum_direction[i_obj] == 1:
                    self.f_ref[i_obj] = np.max(
                        self.objective_function_observed[:, i_obj])
                    # if mu[i_obj] < self.f_ref[i_obj]:
                    #     ei_x[i_obj] = 0
                    #     continue

                else:
                    self.f_ref[i_obj] = np.min(
                        self.objective_function_observed[:, i_obj])
                    # if mu[i_obj] > self.f_ref[i_obj]:
                    #     ei_x[i_obj] = 0
                    #     continue

                # In case sigma equals zero
                with np.errstate(divide='ignore'):
                    Z = (mu[i_obj] - self.f_ref[i_obj]) / sigma[i_obj]
                    ei_x[i_obj] = \
                        (mu[i_obj] - self.f_ref[i_obj]) * \
                        norm.cdf(Z) + sigma[i_obj] * norm.pdf(Z)
                    ei_x[sigma[i_obj] == 0.0] == 0.0

                # temp1 = float(mu[i_obj])
                # temp2 = float(sigma[i_obj])
                # ei_x[i_obj] = integrate.quad(
                #     self.ei_func, -np.inf, self.f_ref[i_obj],
                #     args=(temp1, temp2, i_obj))

            return ei_x
        return

    # def ei_func(self, F, mu, sigma, i_obj):
    #     temp1 = norm.pdf(F, mu, sigma)
    #     temp2 = abs(self.f_ref[i_obj] - F)
    #     out = temp1 * temp2
    #     # out = \
    #     #     (abs(self.f_ref[i_obj] - F)) * \
    #     #     norm.pdf(F, mu, sigma)
    #     # out = float(out[0])
    #     return out

    def expected_hypervolume_improvement(self, x):
        """ expected_hypervolume_improvement
        Expected hypervolume improvement function.

        Note:
            under construction!

        """

        print('this funcion is under construction, use another fucntion')
        return
