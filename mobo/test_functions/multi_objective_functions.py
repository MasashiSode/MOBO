import copy
import numpy as np


def zdt1(x: np.ndarray) -> np.ndarray:
    """
    Args:
        x (np.ndarray): x must be in [0, 1]
    Returns:
        multi_objective_value (np.ndarray):
    """
    if type(x) is not np.ndarray:
        x_in = np.array(x)
    else:
        x_in = copy.deepcopy(x)
    if np.any(x_in > 1) or np.any(x_in < 0):
        raise ValueError('0 <= x <= 1 is required')
    x_in = x_in.T

    n_dv = x_in.shape[0]
    g = 1 + 9 * sum(x_in[1:]) / (n_dv - 1)
    h = 1 - np.sqrt(x_in[0] / g)
    f1 = x_in[0]
    f2 = g * h
    return np.array([f1, f2])


def zdt2(x: np.ndarray) -> np.ndarray:
    """
    Args:
        x (np.ndarray): x must be in [0, 1]
    Returns:
        multi_objective_value (np.ndarray):
    """
    if type(x) is not np.ndarray:
        x_in = np.array(x)
    else:
        x_in = copy.deepcopy(x)
    if np.any(x_in > 1) or np.any(x_in < 0):
        raise ValueError('0 <= x <= 1 is required')

    n_dv = x_in.shape[0]
    g = 1 + 9 * sum(x_in[1:]) / (n_dv - 1)
    h = 1 - (x_in[0] / g) ** 2
    f1 = x_in[0]
    f2 = g * h
    return np.array([f1, f2])


def schaffer_n1(x: np.ndarray) -> np.ndarray:
    """
    Args:
        x (np.ndarray): x must be in [0, 1]
    Returns:
        multi_objective_value (np.ndarray):
    """
    if type(x) is not np.ndarray:
        x_in = np.array(x)
    else:
        x_in = copy.deepcopy(x)
    if np.any(x_in > 1) or np.any(x_in < 0):
        raise ValueError('0 <= x <= 1 is required')

    f1 = x[0]**2
    f2 = (x[0]-2)**2
    return np.array([f1, f2])


def binh_korn(x: np.ndarray) -> np.ndarray:
    """
    Args:
        x (np.ndarray): x must be in [0, 1]
    Returns:
        multi_objective_value (np.ndarray):
    """
    if type(x) is not np.ndarray:
        x_in = np.array(x)
    else:
        x_in = copy.deepcopy(x)
    if np.any(x_in > 1) or np.any(x_in < 0):
        raise ValueError('0 <= x <= 1 is required')

    x_in = copy.deepcopy(x)
    x_in[0] = x_in[0] * 5
    x_in[1] = x_in[1] * 3

    f1 = 4 * (x_in[0] ** 2) + 4 * (x_in[1] ** 2)
    f2 = (x_in[0] - 5) ** 2 + (x_in[1] - 5) ** 2
    g1 = (x_in[0] - 5) ** 2 + x_in[1] ** 2 - 25
    g2 = 7.7 - (x_in[0] - 8) ** 2 + (x_in[1] + 3) ** 2
    return np.array([f1, f2, g1, g2])


def chakong_haimes(x: np.ndarray) -> np.ndarray:
    """
    Args:
        x (np.ndarray): x must be in [0, 1]
    Returns:
        multi_objective_value (np.ndarray):
    """
    if type(x) is not np.ndarray:
        x_in = np.array(x)
    else:
        x_in = copy.deepcopy(x)
    if np.any(x_in > 1) or np.any(x_in < 0):
        raise ValueError('0 <= x <= 1 is required')
    x_in = copy.deepcopy(x)
    x_in[0] = x_in[0] * 40 - 20
    x_in[1] = x_in[1] * 40 - 20

    f1 = 2 + (x_in[0] - 2) ** 2 + (x_in[1] - 1) ** 2
    f2 = 9 * x_in[0] - (x_in[1] - 1) ** 2
    g1 = x_in[0] ** 2 + x_in[1] ** 2 - 255
    g2 = x_in[0] - 3 * x_in[1] + 10

    return np.array([f1, f2, g1, g2])


def osyczka_kundu(x: np.ndarray) -> np.ndarray:
    """
    Args:
        x (np.ndarray): x must be in [0, 1]
    Returns:
        multi_objective_value (np.ndarray):
    """
    if type(x) is not np.ndarray:
        x_in = np.array(x)
    else:
        x_in = copy.deepcopy(x)
    if np.any(x_in > 1) or np.any(x_in < 0):
        raise ValueError('0 <= x <= 1 is required')
    x_in = copy.deepcopy(x)
    x_in[0] = x_in[0] * 10
    x_in[1] = x_in[1] * 10
    x_in[2] = x_in[2] * 4 + 1
    x_in[3] = x_in[3] * 6
    x_in[4] = x_in[4] * 4 + 1
    x_in[5] = x_in[5] * 10

    f1 = -25 * (x_in[0] - 2) ** 2 - (x_in[1] - 2) ** 2
    - (x_in[2] - 1) ** 2 - (x_in[3] - 4) ** 2
    - (x_in[4] - 1) ** 2
    f2 = 0
    for i in range(0, 6):
        f2 = f2 + x_in[i] ** 2
    g1 = (x_in[0] + x_in[1] - 2)
    g2 = (6 - x_in[0] - x_in[1])
    g3 = (2 + x_in[1] - x_in[0])
    g4 = (2 - x_in[0] + 3 * x_in[1])
    g5 = (4 - (x_in[2] - 3) ** 2 - x_in[3])
    g6 = ((x_in[4] - 3) ** 2 + x_in[5] - 4)
    return np.array([f1, f2, g1, g2, g3, g4, g5, g6])


if __name__ == "__main__":
    pass
