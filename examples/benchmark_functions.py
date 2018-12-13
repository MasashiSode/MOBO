
import math
import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
# https://github.com/scitosci/MOCCA-II


def ZDT1(x):
    # m = 30 , x_i in [0,1]
    n_samples = x.shape[0]
    n_dv = x.shape[1]

    g = np.zeros(n_samples)
    h = np.zeros(n_samples)
    for i in range(0, n_samples):
        g[i] = 1 + 9 * sum(x[i, 1:]) / (n_dv - 1)
        h[i] = 1 - np.sqrt(x[i, 0] / g[i])
    return (x[:, 0], g * h)


def ZDT2(x):
    # m = 30 , x_i in [0,1]
    n_samples = x.shape[0]
    n_dv = x.shape[1]

    g = np.zeros(n_samples)
    h = np.zeros(n_samples)
    for i in range(0, n_samples):
        g[i] = 1 + 9 * sum(x[i, 1:]) / (n_dv - 1)
        h[i] = 1 - (x[i, 0] / g) ** 2
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

    f = np.zeros(n_samples)
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


if __name__ == "__main__":
    n_samples = 50
    n_dv = 2
    x = lhs(n_dv, samples=n_samples)
    f1, f2 = ZDT1(x)
    out1 = np.concatenate([[f1], [f2]], axis=0)
    np.savetxt('ZDT1_obj_init.csv', out1.T, delimiter=',')
    np.savetxt('ZDT1_var_init.csv', x, delimiter=',')
    # plt.grid(True)
    # plt.scatter(f1, f2)
    # plt.show()
