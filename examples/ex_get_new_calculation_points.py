import numpy as np
import MOGP


def ReadInput(InputFile):
    data = np.loadtxt(InputFile, delimiter=",")
    return data


if __name__ == "__main__":
    # x_observed: np.array (n_samples, n_params)
    x_observed = ReadInput('InputVar.csv')
    # y_observed: np.array (n_samples, n_obj + n_cons)
    y_observed = ReadInput('InputObj.csv')

    mobo = MOGP.MultiObjectiveBayesianOptimization()
    mobo.set_train_data(x_observed, y_observed, n_cons=0)

    # training Gaussian Process regression
    mobo.train_GPModel()

    # multi-objective optimization(nsga2) on surrogate model
    mobo.run_moga()

    # clustering pareto front solutions
    mobo.run_kmeans()
    print(mobo.kmeans_centroids_original_coor_x)
