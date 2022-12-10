import numpy as np


def f(x):
    return x**4 - 2*x**2*np.sin(x) + np.sin(x)**2


def gradX(x):
    return 4*x**3 - 4*x*np.sin(x) - 2*x**2*np.cos(x) + 2*np.sin(x)*np.cos(x)


def grad2X(x):
    return 12*x**2 + (2*x**2-4)*np.sin(x) - 8*x*np.cos(x) + 2*np.cos(x)**2 - 2*np.sin(x)**2

x = 0.8
g = 10

count = 0
while np.abs(g) > 10**-8:
    count = count + 1

    g = gradX(x)
    h = grad2X(x)

    x = x -g/h

    print(f'X: {x}\t count: {count}')
