# https://sop.tik.ee.ethz.ch/download/supplementary/testproblems/
import numpy as np
import copy


class ZDT(object):
    def __init__(self):
        self.func = [self.ZDT1,
                     self.ZDT2,
                     self.ZDT3,
                     None,
                     None,
                     self.ZDT6]
        return

    def get_obj_values(self, x, n_zdt=1):
        if n_zdt == 5 or n_zdt == 4:
            raise ValueError
        self.n_zdt = n_zdt - 1
        return self.func[self.n_zdt](x)

    def get_func(self, n_zdt=1):
        if n_zdt == 5 or n_zdt == 4:
            raise ValueError
        self.n_zdt = n_zdt - 1
        return self.func[self.n_zdt]

    def ZDT1(self, x, args=[]):
        # m = 30 , x_i in [0,1]
        n_samples = x.shape[0]
        n_dv = x.shape[1]

        g = np.zeros(n_samples)
        h = np.zeros(n_samples)
        for i in range(0, n_samples):
            g[i] = 1 + 9 * sum(x[i, 1:]) / (n_dv - 1)
            h[i] = 1 - np.sqrt(x[i, 0] / g[i])
        return [x[:, 0], g * h]

    def ZDT2(self, x, args=[]):
        # m = 30 , x_i in [0,1]
        n_samples = x.shape[0]
        n_dv = x.shape[1]

        g = np.zeros(n_samples)
        h = np.zeros(n_samples)
        for i in range(0, n_samples):
            g[i] = 1 + 9 * sum(x[i, 1:]) / (n_dv - 1)
            h[i] = 1 - (x[i, 0] / g[i]) ** 2
        return [x[:, 0], g * h]

    def ZDT3(self, x, args=[]):
        # m = 30 , x_i in [0,1]
        n_samples = x.shape[0]
        n_dv = x.shape[1]

        g = np.zeros(n_samples)
        h = np.zeros(n_samples)
        for i in range(0, n_samples):
            g[i] = 1 + 9 * sum(x[i, 1:]) / (n_dv - 1)
            h[i] = 1 - np.sqrt(x[i, 0] / g[i]) - \
                (x[i, 0] / g[i]) * np.sin(10 * np.pi * x[i, 0])
        return [x[:, 0], g * h]

    # def ZDT4(self, x):
    #     # m = 10 , x_1 in [0,1] , x_2,...,x_m in [-5,5]
    #     n_samples = x.shape[0]
    #     n_dv = x.shape[1]

    #     g = np.zeros(n_samples)
    #     h = np.zeros(n_samples)
    #     for i in range(0, n_samples):
    #         g[i] = 1 + 10 * (n_dv - 1) + \
    #             sum([(t ** 2 - 10 * np.cos(4 * np.pi * t))
    #                  for t in x[i, 1:]])
    #         h[i] = 1 - np.sqrt(x[i, 0] / g[i])
    #     return [x[:, 0], g * h]

    # def ZDT5(self, x):
    #     # m = 11 , x_1={0,1}^30 , x_2,...,x_m={0,1}^5
    #     n_samples = x.shape[0]

    #     f1 = np.zeros(n_samples)
    #     g = np.zeros(n_samples)
    #     h = np.zeros(n_samples)

    #     for i in range(0, n_samples):
    #         f1[i] = 1 + x[i, 0].count('1')
    #         for i in x[i, 1:]:
    #             if i.count('1') < 5:
    #                 g[i] += 2 + i.count('1')
    #             elif i.count('1') == 5:
    #                 g[i] += 1
    #         h[i] = 1 / f1[i]
    #     return [f1, g * h]

    def ZDT6(self, x, args=[]):
        # m = 10 , x_i in [0,1]
        n_samples = x.shape[0]
        n_dv = x.shape[1]

        f1 = np.zeros(n_samples)
        g = np.zeros(n_samples)
        h = np.zeros(n_samples)

        for i in range(0, n_samples):
            f1[i] = 1 - np.exp(-4 * x[i, 0]) * \
                (np.sin(6 * np.pi * x[i, 0]) ** 6)
            g[i] = 1 + 9 * (sum(x[i, 1:]) / (n_dv - 1)) ** 0.25
            h[i] = 1 - (f1[i] / g[i]) ** 2
        return [f1, h]


def BinhKornFunction(x, args=[]):

    x_in = copy.deepcopy(x)
    x_in[:, 0] = x_in[:, 0] * 5
    x_in[:, 1] = x_in[:, 1] * 3

    f1 = 4 * (x_in[:, 0] ** 2) + 4 * (x_in[:, 1] ** 2)
    f2 = (x_in[:, 0] - 5) ** 2 + (x_in[:, 1] - 5) ** 2
    g1 = (x_in[:, 0] - 5) ** 2 + x_in[:, 1] ** 2 - 25
    g2 = 7.7 - (x_in[:, 0] - 8) ** 2 + (x_in[:, 1] + 3) ** 2
    return [f1, f2, g1, g2]


def ChakongHaimesFunction(x, args=[]):

    x_in = copy.deepcopy(x)
    x_in[:, 0] = x_in[:, 0] * 40 - 20
    x_in[:, 1] = x_in[:, 1] * 40 - 20

    f1 = 2 + (x_in[:, 0] - 2) ** 2 + (x_in[:, 1] - 1) ** 2
    f2 = 9 * x_in[:, 0] - (x_in[:, 1] - 1) ** 2
    g1 = x_in[:, 0] ** 2 + x_in[:, 1] ** 2 - 255
    g2 = x_in[:, 0] - 3 * x_in[:, 1] + 10
    return [f1, f2, g1, g2]


def OsyczkaKunduFunction(x, args=[]):

    x_in = copy.deepcopy(x)
    x_in[:, 0] = x_in[:, 0] * 10
    x_in[:, 1] = x_in[:, 1] * 10
    x_in[:, 2] = x_in[:, 2] * 4 + 1
    x_in[:, 3] = x_in[:, 3] * 6
    x_in[:, 4] = x_in[:, 4] * 4 + 1
    x_in[:, 5] = x_in[:, 5] * 10

    f1 = -25 * (x_in[:, 0] - 2) ** 2 - (x_in[:, 1] - 2) ** 2 \
        - (x_in[:, 2] - 1) ** 2 - (x_in[:, 3] - 4) ** 2 \
        - (x_in[:, 4] - 1) ** 2
    f2 = 0
    for i in range(0, 6):
        f2 = f2 + x_in[:, i] ** 2
    g1 = (x_in[:, 0] + x_in[:, 1] - 2)
    g2 = (6 - x_in[:, 0] - x_in[:, 1])
    g3 = (2 + x_in[:, 1] - x_in[:, 0])
    g4 = (2 - x_in[:, 0] + 3 * x_in[:, 1])
    g5 = (4 - (x_in[:, 2] - 3) ** 2 - x_in[:, 3])
    g6 = ((x_in[:, 4] - 3) ** 2 + x_in[:, 5] - 4)

    return [f1, f2, g1, g2, g3, g4, g5, g6]
