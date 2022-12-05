import numpy as np
from SQP import SQP

# Design Optimization HW5 - P1: Sequential Quadratic Programming
# min f = x1^2 + (x2-3)^2
# s.t. g1 = x2^2 - 2x1 <= 0
#      g2 = (x2-1)^2 + 5x1 - 15 <= 0


# Define objective function
def f(X):
    x1 = X[0][0]
    x2 = X[1][0]
    return x1**2 + (x2-3)**2


# Define equality constraints (none)
def h(X):
    return np.array([[]]).reshape(0, 1)


# Define inequality constraints
def g(X):
    x1 = X[0][0]
    x2 = X[1][0]

    g1 = x2**2 - 2*x1
    g2 = (x2-1)**2 + 5*x1 - 15

    return np.array([[g1, g2]]).T


# Define gradient of objective function
def gradF(X):
    x1 = X[0][0]
    x2 = X[1][0]
    return np.array([[2*x1, 2*(x2-3)]]).T


# Define gradient of equality constraints
def gradH(X):
    return np.array([[]]).reshape(0, np.size(X, 0))


# Define gradient of inequality constraints
def gradG(X):
    x2 = X[1][0]
    return np.array([[-2, 2*x2], [5, 2*(x2-1)]])


# Solve problem with SQP algorithm
X0 = np.array([[1, 1]]).T
[X_min, f] = SQP.solve(X0, f, h, g, gradF, gradH, gradG)

# Display results
print("\nAnswer")
print(f'X:\t{X_min[:,0]}')
print(f'f(x):\t{f}')
