import numpy as np
import MOGP as MOGP
import matplotlib.pyplot as plt
from pyDOE import lhs


if __name__ == "__main__":

    n_init_lhs_samples = 128

    n_dv = 2
    n_obj_cons = 4
    n_cons = 2

    n_iter = 3
    n_new_ind = 16

    ga_pop_size = 100
    ga_gen = 50

    # user defined function y = f(x, args=[])
    func = MOGP.TestFunctions.BinhKornFunction

    mobo = MOGP.MultiObjectiveBayesianOptimization()
    mobo.run_mobo(func=func, args=[],
                  n_dv=n_dv, n_obj_cons=n_obj_cons,
                  n_init_lhs_samples=n_init_lhs_samples,
                  n_iter=n_iter, n_new_ind=n_new_ind,
                  ga_pop_size=ga_pop_size, ga_gen=ga_gen, n_cons=n_cons)

    np.savetxt('func_obj_res.csv',
               mobo.y_observed, delimiter=',')
    np.savetxt('func_var_res.csv',
               mobo.x_observed, delimiter=',')

    plt.grid(True)
    plt.scatter(mobo.y_observed[:, 0], mobo.y_observed[:, 1])
    plt.show()
