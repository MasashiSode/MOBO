
import math
import numpy as np
import matplotlib.pyplot as plt
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


# def ZDT2(x):
#     # m = 30 , x_i in [0,1]
#     n_samples = x.shape[0]
#     n_dv = x.shape[1]
#     m = len(x)
#     g = 1 + 9 * sum(x[1:]) / (m - 1)
#     h = 1 - (x[0] / g) ** 2
#     return (x[0], g * h)


# def ZDT3(x):
#     # m = 30 , x_i in [0,1]
#     n_samples = x.shape[0]
#     n_dv = x.shape[1]
#     m = len(x)
#     g = 1 + 9 * sum(x[1:]) / (m - 1)
#     h = 1 - math.sqrt(x[0] / g) - (x[0] / g) * math.sin(10 * math.pi * x[0])
#     return (x[0], g * h)


# def ZDT4(x):
#     # m = 10 , x_1 in [0,1] , x_2,...,x_m in [-5,5]
#     n_samples = x.shape[0]
#     n_dv = x.shape[1]
#     m = len(x)
#     g = 1 + 10 * (m - 1) + \
#         sum([(t ** 2 - 10 * math.cos(4 * math.pi * t)) for t in x[1:]])
#     h = 1 - math.sqrt(x[0] / g)
#     return (x[0], g * h)


# def ZDT5(x):
#     # m = 11 , x_1={0,1}^30 , x_2,...,x_m={0,1}^5
#     f1 = 1 + x[0].count('1')
#     g = 0
#     for i in x[1:]:
#         if i.count('1') < 5:
#             g += 2 + i.count('1')
#         elif i.count('1') == 5:
#             g += 1
#     h = 1 / f1
#     return (f1, g * h)


# def ZDT6(x):
#     # m = 10 , x_i in [0,1]
#     m = len(x)
#     f1 = 1 - math.exp(-4 * x[0]) * math.sin(6 * math.pi * x[0]) ** 6
#     g = 1 + 9 * (sum(x[1:]) / (m - 1)) ** 0.25
#     h = 1 - (f1 / g) ** 2
#     return (f1, g * h)


if __name__ == "__main__":
    n_samples = 50
    n_dv = 2
    x = np.random.rand(n_samples, n_dv)
    # f1, f2 = sin(x)
    f1, f2 = ZDT1(x)
    out1 = np.concatenate([[f1], [f2]], axis=0)
    np.savetxt('ZDT1_obj.csv', out1.T, delimiter=',')
    np.savetxt('ZDT1_var.csv', x, delimiter=',')
    # plt.grid(True)
    # plt.scatter(f1, f2)
    # plt.show()
