import numpy as np


def f(X):
    x1 = X[0][0]
    x2 = X[1][0]
    return x1**2 + (x2-3)**2


def h(X):
    return np.array([[]])


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
    return np.array([[]])


def gradG(X):
    x2 = X[1][0]
    return np.array([[-2, 2*x2], [5, 2*(x2-1)]])


def quasiNewtonW(s, y, H):
    # todo: check PD

    a = np.matmul(y, y.T)
    b = np.matmul(y.T, s)
    c = np.matmul(np.matmul(H, s), np.matmul(s.T, H))
    d = np.matmul(np.matmul(s.T, H), s)

    H1 = H + (a / b) - (c / d)
    return H1











