# Author: Nathaniel H. Gatesh
# Date:   18 September 2022

import numpy as np


def objective(x):
    r = np.array([[-2+2*x[0][0]+3*x[1][0], -x[0][0], 1-x[1][0]]]).T
    return r.T.dot(r)


def grad(x):
    return np.array([[10*x[0][0]+12*x[1][0]-8, 12*x[0][0]+20*x[1][0]-14]]).T


def hess(x):
    return np.array([[10, 12], [12, 20]])


def alpha(x):
    g = grad(x)
    H = hess(x)
    return (g.T.dot(g)) / (g.T.dot(H).dot(g))


X = np.array([[0, 0]]).T
a = alpha(X)
g = grad(X)

while g.T.dot(g) > 0.00001:
    X = X - a*g
    g = grad(X)

x1 = 1-2*X[0][0]-3*X[1][0]
x2 = X[0][0]
x3 = X[1][0]

print(f"Closest Point:{x1}, {x2}, {x3}\t at distance: {objective(X)}")

