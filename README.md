# MOBO (Multi-Objective Bayesian Optimization)

## Install

`python setup.py install`

## Usage

```python
import MOBO
import numpy as np


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


```

under construction
