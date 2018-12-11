import numpy as np
import MOGP
# import multiprocessing as mp
# import pygmo as pg
# from pygmo.problem import base
# from matplotlib import pyplot as plt


def ReadInput(InputFile):
    data = np.loadtxt(InputFile, delimiter=",")
    return data


if __name__ == "__main__":
    y_observed = ReadInput('InputObj.csv')
    x_observed = ReadInput('InputVar.csv')

    mobo = MOGP.MultiObjectiveBayesianOptimization()
    mobo.set_train_data(x_observed, y_observed, n_cons=0)

    # training Gaussian Process regression
    mobo.train_GPModel()

    # multi-objective optimization(nsga2) on surrogate model
    mobo.run_moga()

    # clustering pareto front solutions
    mobo.run_kmeans()
    print(mobo.kmeans_centroids)

    # ax = pg.plot_non_dominated_fronts(mobo.pop.get_f())
    # plt.show()
