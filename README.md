# MOGP (Multi-Objective Gaussian Process)
this package is under construction  
validation not done  

## Install

`python setup.py install`

## Usage

see also https://github.com/MasashiSode/MOGP/tree/master/examples

```python
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

    # user defined function y = f(x, args=[])
    zdt = MOGP.TestFunctions.ZDT()
    func = zdt.get_func(n_zdt)

    mobo = MOGP.MultiObjectiveBayesianOptimization()
    mobo.run_mobo(func=func, args=[],
                  n_dv=n_dv, n_obj=n_obj,
                  n_init_lhs_samples=n_init_lhs_samples,
                  n_iter=n_iter, n_new_ind=n_new_ind,
                  ga_pop_size=ga_pop_size, ga_gen=ga_gen, n_cons=0)

    np.savetxt('ZDT' + str(n_zdt) + '_obj_res.csv',
               mobo.y_observed, delimiter=',')
    np.savetxt('ZDT' + str(n_zdt) + '_var_res.csv',
               mobo.x_observed, delimiter=',')

    plt.grid(True)
    plt.scatter(mobo.y_observed[:, 0], mobo.y_observed[:, 1])
    plt.show()

```

## ToDo

- validation
    - zdt1: done
    - zdt2: done
    - zdt3: done
    - zdt6: done
- constraints handling
- EHVI handling
- test code
