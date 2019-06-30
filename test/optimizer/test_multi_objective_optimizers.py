from mobo.test_functions import schaffer_n1
from mobo.optimizer import NSGA2, NSGA3, SMPSO


def test_nsga2():
    optimizer = NSGA2()
    x_range = [0, 1]
    optimizer.x_range = x_range
    optimizer.function = schaffer_n1
    res = optimizer.run(n_generations=2)
    assert type(res) == list


def test_nsga3():
    optimizer = NSGA3()
    x_range = [0, 1]
    optimizer.x_range = x_range
    optimizer.function = schaffer_n1
    res = optimizer.run(n_generations=2)
    assert type(res) == list


def test_smpso():
    optimizer = SMPSO()
    x_range = [0, 1]
    optimizer.x_range = x_range
    optimizer.function = schaffer_n1
    res = optimizer.run(max_iterations=2)
    assert type(res) == list
