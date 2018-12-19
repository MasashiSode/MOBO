import numpy as np
import matplotlib.pyplot as plt
import MOBO


if __name__ == "__main__":
    n_zdt = 2

    n_init_lhs_samples = 48

    n_dv = 2
    n_obj_cons = 2

    n_iter = 10
    n_new_ind = 16

    ga_pop_size = 100
    ga_gen = 50

    # user defined function y = f(x, args=[])
    zdt = MOBO.TestFunctions.ZDT()
    func = zdt.get_func(n_zdt)

    mobo = MOBO.MultiObjectiveBayesianOptimization()
    mobo.run_mobo(func=func, args=[],
                  n_dv=n_dv, n_obj_cons=n_obj_cons,
                  n_init_lhs_samples=n_init_lhs_samples,
                  n_iter=n_iter, n_new_ind=n_new_ind,
                  ga_pop_size=ga_pop_size, ga_gen=ga_gen, n_cons=0)

    np.savetxt('ZDT' + str(n_zdt) + '_obj_res.csv',
               mobo.y_observed_org_coor, delimiter=',')
    np.savetxt('ZDT' + str(n_zdt) + '_var_res.csv',
               mobo.x_observed_org_coor, delimiter=',')

    plt.grid(True)
    plt.scatter(mobo.y_observed_org_coor[:, 0], mobo.y_observed_org_coor[:, 1])
    plt.show()
