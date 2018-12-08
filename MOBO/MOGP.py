import numpy as np
import sklearn.gaussian_process as gp

from scipy.stats import norm
from scipy.optimize import minimize
# from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
# from scipy import integrate


class MOGP():
    '''
    MOGP (Multi-Objective Gaussian Process) core class
    (https://thuijskens.github.io/2016/12/29/bayesian-optimisation/)

    Note:
        Sset parameters first before train

    Args:
        No args

    Return:
        no return

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
        return

    def set_train_data(self, x_observed, y_observed):
        '''
        Args:
            x_observed: np.array (n_samples, n_params)
            y_observed: np.array (n_samples, n_obj)

        Return:
            no return

        Example::

            mogp = MOGPOpt.MOGP()
            mogp.set_train_data(x_observed, y_observed)

        '''
        self.objective_function_observed = y_observed
        self.design_variables_observed = x_observed
        self.n_features = x_observed.shape[0]
        self.n_params = x_observed.shape[1]
        self.n_obj = y_observed.shape[1]
        self.optimum_direction = -1 * np.ones(self.n_obj)
        return

    def set_optimum_direction(self, direction_list):
        self.optimum_direction = direction_list
        return

    def train(self):
        '''
        Note:
            multi-objective optimization (n_obj > 1) is also available.

        Args:
            No args

        Return:
            no return

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

        kernel = gp.kernels.Matern()
        if self.n_obj == 1:
            self.gpr = gp.GaussianProcessRegressor(
                kernel=kernel, random_state=0).fit(
                self.design_variables_observed,
                self.objective_function_observed)
        else:
            self.gpr = [None] * self.n_obj
            for i_obj in range(0, self.n_obj):
                self.gpr[i_obj] = gp.GaussianProcessRegressor(
                    kernel=kernel, random_state=0).fit(
                    self.design_variables_observed,
                    self.objective_function_observed)
        return self.gpr

    def predict(self, x):
        '''
        Note:
            use it after training

        Args:
            x: np.array, size = [n_input, n_params]

        Return:
            mu: float or list, size = [n_obj]
            sigma: float or list, size = [n_obj

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
                mu[i_obj] = temp1[0, i_obj]
                sigma[i_obj] = temp2[0]
        return mu, sigma

    def expected_improvement(self, x):
        """ expected_improvement
        Expected improvement acquisition function.

        Arguments:
            x: array-like, shape = [n_samples, n_hyperparams]
                The point for which the expected improvement
                needs to be computed.

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
                mu[i_obj] = temp1[0, i_obj]
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

    def expected_hypervolume_improvement(self, x, gaussian_process, evaluated_loss, greater_is_better=False, n_params=1):
        """ expected_hypervolume_improvement
        Expected improvement acquisition function.
        Arguments:
        ----------
            x: array-like, shape = [n_samples, n_hyperparams]
                The point for which the expected improvement needs to be computed.
            gaussian_process: GaussianProcessRegressor object.
                Gaussian process trained on previously evaluated hyperparameters.
            evaluated_loss: Numpy array.
                Numpy array that contains the values off the loss function for the previously
                evaluated hyperparameters.
            greater_is_better: Boolean.
                Boolean flag that indicates whether the loss function is to be maximised or minimised.
            n_params: int.
                Dimension of the hyperparameter space.
        """

        return
