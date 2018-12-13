import numpy as np
import MOGP as MOGP
import matplotlib.pyplot as plt
from pyDOE import lhs


if __name__ == "__main__":
    n_zdt = 2

    n_init_lhs_samples = 24

    n_dv = 2
    n_obj = 2

    n_iter = 10
    n_new_ind = 16

    ga_pop_size = 100
    ga_gen = 50

    zdt = MOGP.TestFunctions.ZDT()
    func = zdt.get_func(n_zdt)

    # latin hyper cube sampling
    x_observed = lhs(n_dv, samples=n_init_lhs_samples)
    y_observed = np.zeros((n_init_lhs_samples, n_obj))
    y_observed[:, 0], y_observed[:, 1] = func(x_observed)

    np.savetxt('ZDT' + str(n_zdt) + '_var_init.csv', x_observed, delimiter=',')
    np.savetxt('ZDT' + str(n_zdt) + '_obj_init.csv', y_observed, delimiter=',')

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

        # evaluate new points
        print('function evaluation')
        new_indv_x = mobo.kmeans_centroids_original_coor_x
        new_indv_y = np.zeros((new_indv_x.shape[0], 2))
        new_indv_y[:, 0], new_indv_y[:, 1] = \
            func(new_indv_x)
        print('function evaluation done.')

        # update observed values
        x_observed = np.concatenate([x_observed, new_indv_x], axis=0)
        y_observed = np.concatenate([y_observed, new_indv_y], axis=0)

    np.savetxt('ZDT' + str(n_zdt) + '_obj_res.csv', y_observed, delimiter=',')
    np.savetxt('ZDT' + str(n_zdt) + '_var_res.csv', x_observed, delimiter=',')

    plt.grid(True)
    plt.scatter(y_observed[:, 0], y_observed[:, 1])
    plt.show()
