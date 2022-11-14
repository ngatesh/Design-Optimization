import numpy as np
from numpy.linalg import inv

k = 0
epsilon = 10**-3


def _h(s, d):
    s = s[0]
    d = d[0]
    h1 = 1/4*s[0]**2 + 1/5*s[1]**2 + 1/25*d[0]**2 - 1
    h2 = s[0] + s[1] - d[0]
    return np.array([[h1, h2]]).T


def _f(s, d):
    s = s[0]
    d = d[0]
    return s[0]**2 + s[1]**2 + d[0]**2


def df_dd(s, d):
    s = s[0]
    d = d[0]
    return 2*d[0]


def df_ds(s, d):
    s = s[0]
    d = d[0]
    return np.array([[2*s[0], 2*s[1]]])


def dh_dd(s, d):
    s = s[0]
    d = d[0]
    return np.array([[2/25*d[0], -1]]).T


def dh_ds(s, d):
    s = s[0]
    d = d[0]
    return np.array([[1/2*s[0], 2/5*s[1]],
                     [1,        1]])


def df_Dd(s, d):
    dfdd = df_dd(s, d)
    dfds = df_ds(s, d)
    dhdd = dh_dd(s, d)
    dhds_inv = inv(dh_ds(s, d))

    return dfdd - np.matmul(np.matmul(dfds, dhds_inv), dhdd)


def solve(s, d):
    h = _h(s, d)

    while np.sum(h**2) > epsilon:
        s = s - np.matmul(h.T, inv(dh_ds(s, d)).T)
        h = _h(s, d)

    return s


def lineSearch(dfDd, s, d):
    a = 1
    b = 0.5
    t = 0.3

    ds = 0
    dd = 0

    f = 1
    phi = 0

    while f > phi:

        dhds_inv = inv(dh_ds(s, d))
        dhdd = dh_dd(s, d)

        dd = -a * dfDd
        ds = a * np.matmul(np.matmul(dhds_inv, dhdd), dfDd.T).T

        f = _f(s + ds, d + dd)
        phi = _f(s, d) - a * t * np.matmul(dfDd, dfDd.T)

        a = b * a

    return [s + ds, d + dd]


d = np.array([[1]])
s = np.array([[1, 1]])
s = solve(s, d)

dfDd = df_Dd(s, d)

while sum(dfDd**2) > epsilon:
    [s, d] = lineSearch(dfDd, s, d)
    s = solve(s, d)
    dfDd = df_Dd(s, d)

    x1 = s[0][0]
    x2 = s[0][1]
    x3 = d[0][0]

    print(f'x1: {x1:.2f}\tx2: {x2:.2f}\tx3: {x3:.2f}\tdfDd: {dfDd[0][0]:.2f}\tf: {_f(s, d):.2f}\t')
