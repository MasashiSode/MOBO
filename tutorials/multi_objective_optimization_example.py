import numpy as np
import matplotlib.pyplot as plt

from mobo.optimizer import NSGA2
from mobo.model import ExactGPModel
from mobo.acquisition import ei
from mobo.bayesopt import MultiObjectiveBayesianOpt
from mobo.test_functions import zdt1

if __name__ == "__main__":

    # multi objective genetic algorithm (NSGA2) is implemented with 'DEAP'
    # Gaussian Process model is implemented with 'gpytorch'
    opt = MultiObjectiveBayesianOpt(evaluation_function=zdt1,
                                    surrogate_model=ExactGPModel,
                                    optimizer=NSGA2,
                                    acquisition=ei,
                                    n_objective_dimension=2,
                                    n_design_variables_dimension=30,
                                    n_initial_sample=16,
                                    bayesian_optimization_iter_max=2,
                                    likelihood_optimization_iter_max=1000,
                                    likelihood_optimization_criteria=1e-3,
                                    n_new_samples=16)
    result = opt.optimize()

    front = np.array(result[1])

    plt.scatter(front[:, 0], front[:, 1], c="b")
    plt.axis("tight")
    print(result)
    plt.show()
