# Author: Nathaniel H. Gatesh
# Date:   27 August 2022

import numpy as np
from scipy.optimize import minimize, LinearConstraint


def objective_function(x):
    return (x[1] - x[2]) ** 2 + (x[2] + x[3] - 2) ** 2 + (x[4] - 1) ** 2 + (x[5] - 1) ** 2


A = np.array([[1, 3, 0, 0, 0],
              [0, 0, 1, 1, -2],
              [0, 1, 0, 0, -1]])

B = np.array([0, 0, 0]).T

constraint = LinearConstraint(A, lb=B, ub=B)

bounds = [(-10, 10) for i in range(5)]

