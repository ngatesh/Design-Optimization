import numpy as np


def df_dd(s, d):
    return 2*d[0]


def df_ds(s, d):
    return np.array([[2*s[0], 2*s[1]]])


def dh_dd(s, d):
    return np.array([[2/25*d[0], -1]]).T


def dh_ds(s, d):
    return np.array([[1/2*s[0], 2/5*s[1]],
                     [1,        1]])


k = 0
epsilon = 10**-3

s = np.array([1, 2])
d = np.array([3])

print(df_dd(s, d))
print(df_ds(s, d))
print(dh_dd(s, d))
print(dh_ds(s, d))

