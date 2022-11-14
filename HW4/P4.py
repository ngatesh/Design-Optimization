import numpy as np
from numpy.linalg import inv

k = 0
epsilon = 10**-3


def _h(s, d):
    h1 = 1/4*s[0]**2 + 1/5*s[1]**2 + 1/25*d[0]**2 - 1
    h2 = s[0] + s[1] - d[0]
    return np.array([[h1, h2]]).T


def _f(s, d):
    return s[0]**2 + s[1]**2 + d[0]**2


def df_dd(s, d):
    return 2*d[0]


def df_ds(s, d):
    return np.array([[2*s[0], 2*s[1]]])


def dh_dd(s, d):
    return np.array([[2/25*d[0], -1]]).T


def dh_ds(s, d):
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

    while h.T.h > epsilon:
        s = s.T - np.matmul(np.invert(dh_ds(s, d)), h)
        h = _h(s, d)

    return s


def lineSearch(dfDd, s, d):
    a = 1
    b = 0.5
    t = 0.3

    f = 1
    phi = 0

    while f > phi:

        dhds_inv = inv(dh_ds(s, d))
        dhdd = dh_dd(s, d)
        dfdd = df_dd(s, d)

        dd = -a * dfDd
        ds = a * np.matmul(np.matmul(dhds_inv, dhdd), dfdd.T).T

        f = _f(s + ds, d + dd)
        phi = _f(s, d) - a * t * np.matmul(dfDd, dfDd.T)

        a = b * a

    return a


d = np.array([1])
s = np.array([1, 1])
s = solve(s, d)

dfDd = df_Dd(s, d)

while dfDd.T.dfDd > epsilon:
    alpha = lineSearch(dfDd, s, d)
    d = d - alpha * dfDd
    s


print(df_dd(s, d))
print(df_ds(s, d))
print(dh_dd(s, d))
print(dh_ds(s, d))

