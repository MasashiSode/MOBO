import numpy as np
import MOGP as MOGP
import matplotlib.pyplot as plt
from pyDOE import lhs


def ZDT1(x):
    # m = 30 , x_i in [0,1]
    n_samples = x.shape[0]
    n_dv = x.shape[1]

    g = np.zeros(n_samples)
    h = np.zeros(n_samples)
    for i in range(0, n_samples):
        g[i] = 1 + 9 * sum(x[i, 1:]) / (n_dv - 1)
        h[i] = 1 - np.sqrt(x[i, 0] / g[i])
    return [x[:, 0], g * h]


def ZDT2(x):
    # m = 30 , x_i in [0,1]
    n_samples = x.shape[0]
    n_dv = x.shape[1]

    g = np.zeros(n_samples)
    h = np.zeros(n_samples)
    for i in range(0, n_samples):
        g[i] = 1 + 9 * sum(x[i, 1:]) / (n_dv - 1)
        h[i] = 1 - (x[i, 0] / g[i]) ** 2
    return [x[0], g * h]


def ZDT3(x):
    # m = 30 , x_i in [0,1]
    n_samples = x.shape[0]
    n_dv = x.shape[1]

    g = np.zeros(n_samples)
    h = np.zeros(n_samples)
    for i in range(0, n_samples):
        g[i] = 1 + 9 * sum(x[i, 1:]) / (n_dv - 1)
        h[i] = 1 - np.sqrt(x[i, 0] / g[i]) - (x[i, 0] /
                                              g[i]) * np.sin(10 * np.pi * x[0])
    return [x[0], g * h]


def ZDT4(x):
    # m = 10 , x_1 in [0,1] , x_2,...,x_m in [-5,5]
    n_samples = x.shape[0]
    n_dv = x.shape[1]

    g = np.zeros(n_samples)
    h = np.zeros(n_samples)
    for i in range(0, n_samples):
        g[i] = 1 + 10 * (n_dv - 1) + \
            sum([(t ** 2 - 10 * np.cos(4 * np.pi * t)) for t in x[i, 1:]])
        h[i] = 1 - np.sqrt(x[i, 0] / g[i])
    return [x[0], g * h]


def ZDT5(x):
    # m = 11 , x_1={0,1}^30 , x_2,...,x_m={0,1}^5
    n_samples = x.shape[0]
    n_dv = x.shape[1]

    f1 = np.zeros(n_samples)
    g = np.zeros(n_samples)
    h = np.zeros(n_samples)

    for i in range(0, n_samples):
        f1[i] = 1 + x[i, 0].count('1')
        for i in x[i, 1:]:
            if i.count('1') < 5:
                g[i] += 2 + i.count('1')
            elif i.count('1') == 5:
                g[i] += 1
        h[i] = 1 / f1[i]
    return [f1, g * h]


def ZDT6(x):
    # m = 10 , x_i in [0,1]
    n_samples = x.shape[0]
    n_dv = x.shape[1]

    f1 = np.zeros(n_samples)
    g = np.zeros(n_samples)
    h = np.zeros(n_samples)

    for i in range(0, n_samples):
        f1[i] = 1 - np.exp(-4 * x[i, 0]) * np.sin(6 * np.pi * x[i, 0]) ** 6
        g[i] = 1 + 9 * (sum(x[i, 1:]) / (n_dv - 1)) ** 0.25
        h[i] = 1 - (f1[i] / g[i]) ** 2
    return [f1, g * h]


def ReadInput(InputFile):
    data = np.loadtxt(InputFile, delimiter=",")
    return data


if __name__ == "__main__":
    # x_observed: np.array (n_samples, n_params)
    # x_observed = ReadInput('ZDT1_var.csv')
    # y_observed: np.array (n_samples, n_obj + n_cons)
    # y_observed = ReadInput('ZDT1_obj.csv')
    n_iter = 10
    n_new_ind = 16
    ga_pop_size = 100
    ga_gen = 50

    n_init_samples = 50
    n_dv = 2
    n_obj = 2

    # latin hyper cube sampling
    x_observed = lhs(n_dv, samples=n_init_samples)
    y_observed = np.zeros((n_init_samples, n_obj))
    y_observed[:, 0], y_observed[:, 1] = ZDT1(x_observed)

    np.savetxt('ZDT1_var_init.csv', x_observed, delimiter=',')
    np.savetxt('ZDT1_obj_init.csv', y_observed, delimiter=',')

    for i in range(0, n_iter):
        print('\n--- iter: ', i, '/', n_iter - 1, '---')
        mobo = MOGP.MultiObjectiveBayesianOptimization()
        mobo.set_train_data(x_observed, y_observed, n_cons=0)

        # training Gaussian Process regression
        mobo.train_GPModel()

        # multi-objective optimization(nsga2) on surrogate model
        mobo.run_moga(size=ga_pop_size, gen=ga_gen)

        # clustering solutions
        mobo.run_kmeans(n_clusters=n_new_ind, n_jobs=-1, n_init=20)

        new_indv_x = mobo.kmeans_centroids_original_coor_x
        new_indv_y = np.zeros((new_indv_x.shape[0], 2))
        new_indv_y[:, 0], new_indv_y[:, 1] = ZDT1(new_indv_x)

        # delete duplicate values
        x_observed = np.concatenate([x_observed, new_indv_x], axis=0)
        y_observed = np.concatenate([y_observed, new_indv_y], axis=0)
        input_observed = np.concatenate([x_observed, y_observed], axis=1)
        input_observed, indeices = \
            np.unique(input_observed, axis=0, return_inverse=True)
        input_observed = input_observed[indeices]
        x_observed = input_observed[:, 0:x_observed.shape[1]]
        y_observed = \
            input_observed[:, x_observed.shape[1]:x_observed.shape[1] +
                           y_observed.shape[1] + 1]

    np.savetxt('ZDT1_obj_res.csv', y_observed, delimiter=',')
    np.savetxt('ZDT1_var_res.csv', x_observed, delimiter=',')

    plt.grid(True)
    plt.scatter(y_observed[:, 0], y_observed[:, 1])
    plt.show()
