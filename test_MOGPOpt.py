import numpy as np
import MOBO
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

    mobo = MOBO.MultiObjectiveBayesianOptimization()
    mobo.set_train_data(x_observed, y_observed, n_cons=0)
    mobo.train_GPModel()
    mobo.run_moga()
    mobo.run_kmeans()
    print(mobo.kmeans_centroids)

    # ax = pg.plot_non_dominated_fronts(mobo.pop.get_f())
    # plt.show()
