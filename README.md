# MOBO (Multi-Objective Bayesian Optimization)

## Install

`python setup.py install`

## Usage

```python
import MOBO
import numpy as np
import pygmo as pg
from matplotlib import pyplot as plt


def ReadInput(InputFile):
    data = np.loadtxt(InputFile, delimiter=",")
    return data


if __name__ == "__main__":
    y_observed = ReadInput('InputObj.csv')
    x_observed = ReadInput('InputVar.csv')

    mobo = MOBO.MultiObjectiveBayesianOptimization()
    mobo.set_train_data(x_observed, y_observed)
    mobo.train_GPModel()
    mobo.run()
    ax = pg.plot_non_dominated_fronts(mobo.pop.get_f())
    plt.show()


```

under construction
