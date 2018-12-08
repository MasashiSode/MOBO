import MOBO
import numpy as np


def ReadInput(InputFile):
    data = np.loadtxt(InputFile, delimiter=",")
    return data


y_observed = ReadInput('InputObj.csv')
x_observed = ReadInput('InputVar.csv')
# print(x_observed.T[:, 0])
# print(dir(Kr))

Mogp = MOBO.MOGP()
Mogp.set_train_data(x_observed, y_observed)
# print(K.objective_function_observed)
Mogp.train()
x = np.array([-5, -5])
mu, sigma = Mogp.predict(x)
print('mu: ', mu)
print('sigma: ', sigma)

x = np.array([[-4.9, -4.9]])
ei = Mogp.expected_improvement(x)
print(ei)
# print('predict: ', K.predict([-5, -5]))
