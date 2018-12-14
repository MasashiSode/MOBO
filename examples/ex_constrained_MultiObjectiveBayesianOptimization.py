import numpy as np
import MOGP
import matplotlib.pyplot as plt
from pyDOE import lhs


if __name__ == "__main__":

    n_init_lhs_samples = 24

    n_dv = 2
    n_obj_cons = 4
    n_cons = 2

    n_iter = 16
    n_new_ind = 4

    ga_pop_size = 100
    ga_gen = 50

    mutation = 0.04

    # user defined function y = f(x, args=[])
    func = MOGP.TestFunctions.ChakongHaimesFunction
    # func = MOGP.TestFunctions.OsyczkaKunduFunction

    mobo = MOGP.MultiObjectiveBayesianOptimization()
    mobo.run_mobo(func=func, args=[],
                  n_dv=n_dv, n_obj_cons=n_obj_cons,
                  n_init_lhs_samples=n_init_lhs_samples,
                  n_iter=n_iter, n_new_ind=n_new_ind,
                  ga_pop_size=ga_pop_size, ga_gen=ga_gen, n_cons=n_cons,
                  mutation=mutation)

    np.savetxt('func_obj_res.csv',
               mobo.y_observed_org_coor, delimiter=',')
    np.savetxt('func_var_res.csv',
               mobo.x_observed_org_coor, delimiter=',')

    index = np.all(
        mobo.y_observed_org_coor[:, n_obj_cons - n_cons + 1:n_obj_cons] < 0, axis=1)
    # index_new_ind = np.all(mobo.y_observed_org_coor[n_init_lhs_samples:-1, n_obj_cons - n_cons + 1:n_obj_cons] < 0, axis=1)

    out_feasible = mobo.y_observed_org_coor[index]
    # out_feasible_new_ind = mobo.y_observed_org_coor[index_new_ind]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.grid(True)
    plt.scatter(
        mobo.y_observed_org_coor[:, 0], mobo.y_observed_org_coor[:, 1], label='infeasible')
    plt.scatter(out_feasible[:, 0], out_feasible[:, 1], label='feasible')
    # plt.scatter(mobo.y_observed_org_coor[n_init_lhs_samples:-1, 0],
    #             mobo.y_observed_org_coor[n_init_lhs_samples:-1, 1],
    #             label='new individuals')
    # plt.scatter(out_feasible_new_ind[:, 0], out_feasible_new_ind[:, 1], label='feasible new individuals')

    ax.legend()
    plt.show()
