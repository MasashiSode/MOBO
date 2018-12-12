import numpy as np
import MOGP as MOGP
import matplotlib.pyplot as plt


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


def ReadInput(InputFile):
    data = np.loadtxt(InputFile, delimiter=",")
    return data


if __name__ == "__main__":
    # x_observed: np.array (n_samples, n_params)
    x_observed = ReadInput('ZDT1_var.csv')
    # y_observed: np.array (n_samples, n_obj + n_cons)
    y_observed = ReadInput('ZDT1_obj.csv')
    n_iter = 10
    n_new_ind = 8
    size = 100
    gen = 50

    for i in range(0, n_iter):
        print('\n--- iter: ', i, '/', n_iter - 1, '---')
        mobo = MOGP.MultiObjectiveBayesianOptimization()
        mobo.set_train_data(x_observed, y_observed, n_cons=0)

        # training Gaussian Process regression
        mobo.train_GPModel()

        # multi-objective optimization(nsga2) on surrogate model
        mobo.run_moga(size=size, gen=gen)

        # clustering pareto front solutions
        mobo.run_kmeans(n_clusters=n_new_ind)
        new_indv_x = mobo.kmeans_centroids_original_coor_x
        new_indv_y = np.zeros((n_new_ind, 2))
        new_indv_y[:, 0], new_indv_y[:, 1] = ZDT1(new_indv_x)

        x_observed = np.concatenate([x_observed, new_indv_x], axis=0)
        y_observed = np.concatenate([y_observed, new_indv_y], axis=0)

    np.savetxt('ZDT1_obj_opt.csv', y_observed, delimiter=',')
    np.savetxt('ZDT1_var_opt.csv', x_observed, delimiter=',')

    plt.grid(True)
    plt.scatter(y_observed[:, 0], y_observed[:, 1])
    plt.show()
