# MOBO (Multi-Objective Bayesian Optimization)

constrained/unconstrained multi-objective bayesian optimization package

## Install

`python setup.py install`

## Usage

see also [examples](https://github.com/MasashiSode/MOGP/tree/master/examples)

```python
import numpy as np
import matplotlib.pyplot as plt
import MOBO


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
    func = MOBO.TestFunctions.ChakongHaimesFunction
    # func = MOBO.TestFunctions.OsyczkaKunduFunction

    mobo = MOBO.MultiObjectiveBayesianOptimization()
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
        mobo.y_observed_org_coor[:, n_obj_cons - n_cons + 1:n_obj_cons] < 0,
        axis=1)

    out_feasible = mobo.y_observed_org_coor[index]
    # out_feasible_new_ind = mobo.y_observed_org_coor[index_new_ind]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.grid(True)
    plt.scatter(
        mobo.y_observed_org_coor[:, 0], mobo.y_observed_org_coor[:, 1],
        label='infeasible')
    plt.scatter(out_feasible[:, 0], out_feasible[:, 1], label='feasible')

    ax.legend()
    plt.show()

```

## ToDo

- validation
    - zdt1: done
    - zdt2: done
    - zdt3: done
    - zdt6: done
    - Binh and Korn function: done
- EHVI handling
- test code
