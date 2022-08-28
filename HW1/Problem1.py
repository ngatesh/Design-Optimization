# Author: Nathaniel H. Gatesh
# Date:   27 August 2022

import numpy as np
from scipy.optimize import minimize, LinearConstraint


# Define function to minimize.
def objective_function(x):
    return (x[0] - x[1])**2 + (x[1] + x[4] - 2)**2 + (x[3] - 1)**2 + (x[4] - 1)**2


# Define linear constraint coefficient matrix (AX=b).
A = np.array([[1, 3, 0, 0, 0],
              [0, 0, 1, 1, -2],
              [0, 1, 0, 0, -1]])

# Define linear constraint value matrix (AX=b).
b = np.array([0, 0, 0])

constraint = LinearConstraint(A, lb=b, ub=b)    # Initialize constraint object.
bounds = [(-10, 10) for i in range(5)]          # Define bounds for all x values.
guess1 = np.array([0, 0, 0, 0, 0])              # Initial guess for x values.
guess2 = np.array([-3, 4, 9, -6, -1])
guess3 = np.array([0, 1, 1, 4, -2])

# Get Minimization results.

print(f"\nFor Guess: {guess1}")
res = minimize(objective_function, x0=guess1, constraints=constraint, bounds=bounds)
print(f"f(X): {res.fun}\nX: {res.x}\n")

print(f"For Guess: {guess2}")
res = minimize(objective_function, x0=guess2, constraints=constraint, bounds=bounds)
print(f"f(X): {res.fun}\nX: {res.x}\n")

print(f"For Guess: {guess3}")
res = minimize(objective_function, x0=guess3, constraints=constraint, bounds=bounds)
print(f"f(X): {res.fun}\nX: {res.x}")

"""
Written Answer: Changing the initial guess does not meaningfully affect the final result (see output.png).
"""