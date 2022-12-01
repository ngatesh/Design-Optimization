import numpy as np
from SQP import SQP


def functionA(x, y):
    def functionB(c):
        return (x+y)**c

    return functionB


customFun = functionA(3, 4)
print(customFun(2))


def f(X):
    x1 = X[0][0]
    x2 = X[1][0]
    return x1**2 + (x2-3)**2


def h(X):
    return np.array([[]]).reshape(0, 1)


def g(X):
    x1 = X[0][0]
    x2 = X[1][0]

    g1 = x2**2 - 2*x1
    g2 = (x2-1)**2 + 5*x1 - 15

    return np.array([[g1, g2]]).T


def gradF(X):
    x1 = X[0][0]
    x2 = X[1][0]
    return np.array([[2*x1, 2*(x2-3)]]).T


def gradH(X):
    return np.array([[]]).reshape(0, np.size(X, 0))


def gradG(X):
    x2 = X[1][0]
    return np.array([[-2, 2*x2], [5, 2*(x2-1)]])


X0 = np.array([[1, 1]]).T
[X_min, f] = SQP.solve(X0, f, h, g, gradF, gradH, gradG)
