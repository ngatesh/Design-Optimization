import numpy as np
from numpy.linalg import inv

# Termination condition
epsilon = 10**-3


# Define equality constraint matrix.
def _h(s, d):
    s = s[0]
    d = d[0]
    h1 = 1/4*s[0]**2 + 1/5*s[1]**2 + 1/25*d[0]**2 - 1
    h2 = s[0] + s[1] - d[0]
    return np.array([[h1, h2]]).T


# Define objective function.
def _f(s, d):
    s = s[0]
    d = d[0]
    return s[0]**2 + s[1]**2 + d[0]**2


# Derivative of objective w.r.t. decision variable.
def df_dd(s, d):
    s = s[0]
    d = d[0]
    return 2*d[0]


# Partial derivative of objective w.r.t state variable.
def df_ds(s, d):
    s = s[0]
    d = d[0]
    return np.array([[2*s[0], 2*s[1]]])


# Partial derivative of equality constraint w.r.t decision variable.
def dh_dd(s, d):
    s = s[0]
    d = d[0]
    return np.array([[2/25*d[0], -1]]).T


# Partial derivative of equality constraint w.r.t state variable.
def dh_ds(s, d):
    s = s[0]
    d = d[0]
    return np.array([[1/2*s[0], 2/5*s[1]],
                     [1,        1]])


# Reduced gradient of objective function.
def df_Dd(s, d):
    dfdd = df_dd(s, d)
    dfds = df_ds(s, d)
    dhdd = dh_dd(s, d)
    dhds_inv = inv(dh_ds(s, d))

    return dfdd - np.matmul(np.matmul(dfds, dhds_inv), dhdd)


# Uses Newton-Ralphson to solve h(x) = 0 by adjusting state variable (s).
def solve(s, d):
    h = _h(s, d)

    while np.sum(h**2) > epsilon:
        s = s - np.matmul(h.T, inv(dh_ds(s, d)).T)
        h = _h(s, d)

    return s


# Perform line search to minimize objective function.
# Returns the new decision variable (d) and approximate state variable (s).
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


d = np.array([[1]])     # Initial decision guess.
s = np.array([[1, 1]])  # Initial state guess.
s = solve(s, d)         # Solve for state to satisfy h(x) = 0.

dfDd = df_Dd(s, d)      # Reduced gradient.

# Minimize
while sum(dfDd**2) > epsilon:
    [s, d] = lineSearch(dfDd, s, d)     # Perform line search to adjust (s) and (d).
    s = solve(s, d)                     # Solve for exact (s) to satisfy constraints.
    dfDd = df_Dd(s, d)                  # Re-evaluate reduced gradient at new state.

    # Print results.
    x1 = s[0][0]
    x2 = s[0][1]
    x3 = d[0][0]
    print(f'x1: {x1:.2f}\tx2: {x2:.2f}\tx3: {x3:.2f}\tdfDd: {dfDd[0][0]:.2f}\tf: {_f(s, d):.2f}\t')
