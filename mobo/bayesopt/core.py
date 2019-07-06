import copy
import numpy as np
import torch
import gpytorch
from pyDOE2 import lhs
from sklearn.cluster import KMeans
from mobo.optimizer import NSGA2
from mobo.model import ExactGPModel
from mobo.acquisition import ei
from mobo.test_functions import zdt1


class MultiObjectiveBayesianOpt():
    def __init__(self,
                 evaluation_function=None,
                 Initializer=lhs,
                 surrogate_model=ExactGPModel,
                 optimizer=NSGA2,
                 acquisition=ei,
                 n_objective_dimension=2,
                 n_design_variables_dimension=30,
                 n_initial_sample=16,
                 bayesian_optimization_iter_max=10,
                 likelihood_optimization_iter_max=5000,
                 likelihood_optimization_criteria=1e-8,
                 n_new_samples=8
                 ):
        self.Initializer = Initializer
        self.surrogate_model = surrogate_model
        self.model = [None] * n_objective_dimension
        self.likelihood = [None] * n_objective_dimension
        self.optimizer = optimizer
        self.evaluation_function = evaluation_function
        self.acquisition = acquisition
        self.n_objective_dimension = n_objective_dimension
        self.n_design_variables_dimension = n_design_variables_dimension
        self.n_initial_sample = n_initial_sample
        self.train_x = None
        self.train_y = [None] * n_objective_dimension
        self.new_x = None
        self.bayesian_optimization_iter_max = \
            bayesian_optimization_iter_max
        self.likelihood_optimization_iter_max = \
            likelihood_optimization_iter_max
        self.likelihood_optimization_criteria = \
            likelihood_optimization_criteria
        self.n_new_samples = n_new_samples

    def _initialize(self):
        self.train_x = self.Initializer(
            self.n_design_variables_dimension,
            self.n_initial_sample).astype(np.float32)
        self.train_x = torch.from_numpy(self.train_x)
        self.train_y = torch.from_numpy(
            self.evaluation_function(self.train_x).T)
        return

    def _train_likelihood(self):

        for i_obj in range(self.n_objective_dimension):
            self.likelihood[i_obj] = \
                gpytorch.likelihoods.GaussianLikelihood()
            self.model[i_obj] = self.surrogate_model(
                self.train_x, self.train_y[:, i_obj], self.likelihood[i_obj])
            self.model[i_obj].train()
            self.likelihood[i_obj].train()

            # Use the adam optimizer for likelihood optimization
            optimizer_likelihood = torch.optim.Adam([
                # Includes GaussianLikelihood parameters
                {'params': self.model[i_obj].parameters()},
            ], lr=0.1)

            # "Loss" for GPs - the marginal log likelihood
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                self.likelihood[i_obj], self.model[i_obj])

            loss_prev = 0.1
            for i in range(self.likelihood_optimization_iter_max):
                # Zero gradients from previous iteration
                optimizer_likelihood.zero_grad()
                # Output from model
                output = self.model[i_obj](self.train_x)
                # Calc loss and backprop gradients
                loss = - \
                    mll(output, self.train_y[:, i_obj])
                loss.backward()
                loss_residual = abs(loss.item() - loss_prev) / abs(loss_prev)
                loss_prev = loss.item()
                print('Iter %d/%d - Loss: %.3f  res: %.8f' % (
                    i + 1, self.likelihood_optimization_iter_max,
                    loss.item(),
                    loss_residual
                ))
                if loss_residual < self.likelihood_optimization_criteria:
                    break
                optimizer_likelihood.step()

        return self.model

    def _wrap_model_and_acquisition(self):

        def ei_with_surrogate_model(x):
            for i_obj in range(self.n_objective_dimension):
                self.model[i_obj].eval()
                self.likelihood[i_obj].eval()
            y_pred = [None] * self.n_objective_dimension
            res = [None] * self.n_objective_dimension
            for i_obj in range(self.n_objective_dimension):
                y_pred[i_obj] = \
                    self.likelihood[i_obj](
                        self.model[i_obj](torch.tensor([x])))
                res[i_obj] = self.acquisition(y_pred[i_obj],
                                              self.train_y[:, i_obj])
            return res
        return ei_with_surrogate_model

    def _find_new_sample(self):
        for i_obj in range(self.n_objective_dimension):
            self.model[i_obj].eval()
            self.likelihood[i_obj].eval()
        with torch.no_grad():
            ei_with_surrogate_model = self._wrap_model_and_acquisition()
            opt = copy.deepcopy(self.optimizer(
                evaluation_function=ei_with_surrogate_model,
                n_design_variables_dimension=self.n_design_variables_dimension))
            pop, _ = opt.run()
            x = np.array([list(ind) for ind in pop])
            y = np.array([ind.fitness.values for ind in pop])
            kmeans = KMeans(n_clusters=self.n_new_samples)
            kmeans.fit(x, y)
            new_samples = kmeans.cluster_centers_
        return torch.from_numpy(new_samples.astype(np.float32))

    # def _judge_termination(self):
    #     if residual_bayesian_opt < criteria:
    #         termination = True
    #     else:
    #         termination = False
    #     return termination

    def optimize(self):
        self._initialize()
        for bayesian_optimization_iter in range(
                self.bayesian_optimization_iter_max):

            self._train_likelihood()

            print('bayesian opt Iter %d/%d' % (
                bayesian_optimization_iter + 1,
                self.bayesian_optimization_iter_max))

            # if self._judge_termination():
            #     break

            self.new_x = self._find_new_sample()
            self.train_x = torch.cat((self.train_x, self.new_x), dim=0)
            self.train_y = \
                torch.cat((self.train_y,
                           torch.from_numpy(self.evaluation_function(self.new_x).T)), dim=0)
        return self.train_x, self.train_y


if __name__ == "__main__":

    opt = MultiObjectiveBayesianOpt(evaluation_function=zdt1)
    res = opt.optimize()

    import matplotlib.pyplot as plt

    front = np.array(res[1])

    plt.scatter(front[:, 0], front[:, 1], c="b")
    plt.axis("tight")
    plt.show()

    print(res)
