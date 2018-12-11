import MOBO
import numpy as np
# import multiprocessing as mp


def ReadInput(InputFile):
    data = np.loadtxt(InputFile, delimiter=",")
    return data


if __name__ == "__main__":
    # mp.freeze_support()
    y_observed = ReadInput('InputObj.csv')
    x_observed = ReadInput('InputVar.csv')

    Mogp = MOBO.MOGP()
    Mogp.set_train_data(x_observed, y_observed)
    # Mogp.set_number_of_cpu_core(1)
    Mogp.train()

    x = np.array([-5, -5])
    mu, sigma = Mogp.predict(x)
    print('mu: ', mu)
    print('sigma: ', sigma)

    x = np.array([[-4.9, -4.9]])
    ei = Mogp.expected_improvement(x)
    print(ei)
