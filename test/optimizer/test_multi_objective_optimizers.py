import numpy as np
from mobo.test_functions import zdt1
from mobo.optimizer import NSGA2


def test_nsga2():
    n_population = 4
    nsga2 = NSGA2(evaluation_function=zdt1,
                  bound_low=0.0,
                  bound_up=1.0,
                  n_design_variables_dimension=30,
                  n_population=n_population,
                  n_generation=1)
    pop, _ = nsga2.run()
    assert len(pop) == n_population


if __name__ == "__main__":
    test_nsga2()
