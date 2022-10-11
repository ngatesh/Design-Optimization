from matplotlib import pyplot as plt
from skopt import gp_minimize
from skopt.plots import plot_convergence

# Author: Nathaniel H. Gatesh
# Date: 11 October 2022

def f(x):
    x1 = x[0]
    x2 = x[1]
    return (4 - 2.1*x1**2 + 1/3*x1**4)*x1**2 + x1*x2 + (-4 + 4*x2**2)*x2**2


res = gp_minimize(f, [(-3.0, 3.0), (-2.0, 2.0)], n_calls=40, n_random_starts=10, random_state=5657)

print(f"X1= {res.x[0]}\nX2= {res.x[1]}\nf= {res.fun}")
plot_convergence(res)
plt.show()
