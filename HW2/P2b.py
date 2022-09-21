# Author: Nathaniel H. Gatesh
# Date:   20 September 2022

import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt

"""
SUMMARY: See Convergence1, Convergence2, Output1, and Output2.png in this folder.
        Impression: Both methods, and both initial guesses, produced the same final answer.
                    However, Newton's method converged faster.
"""


# Define objective function to minimize.
def objective(x):
    r = np.array([[-2+2*x[0][0]+3*x[1][0], -x[0][0], 1-x[1][0]]]).T
    return r.T.dot(r)[0][0]


# Define gradient of objective function.
def grad(x):
    return np.array([[10*x[0][0]+12*x[1][0]-8, 12*x[0][0]+20*x[1][0]-14]]).T


# Define Hessian of objective function.
def hess(x):
    return np.array([[10, 12], [12, 20]])


# Calculates exact line search alpha for Gradient Descent.
def alpha(x):
    g = grad(x)
    H = hess(x)
    return (g.T.dot(g)) / (g.T.dot(H).dot(g))


# Calculates dx for Newton's Method.
def delta(x):
    return -np.linalg.inv(hess(x)).dot(grad(x))


# Method 1: Gradient Descent, Exact Line Search
X = np.array([[5, -8]]).T
print(f"GD Method for starting guess ({X[0]}, {X[1]}):")

f = objective(X)
f_last = f + 1

errorA = [f]

while abs(f-f_last) > 10**-8:
    X = X - alpha(X)*grad(X)
    f_last = f
    f = objective(X)
    errorA.append(abs(f-0))

x1 = 1-2*X[0][0]-3*X[1][0]  # Calculate the dependent point.
x2 = X[0][0]
x3 = X[1][0]

print(f"\tClosest Point:{x1}, {x2}, {x3}\t at Distance: {np.sqrt(objective(X))}")

# Method 2: Newton's Method
X = np.array([[5, -8]]).T
print(f"Newton's Method for starting guess ({X[0]}, {X[1]}):")

f = objective(X)
f_last = f + 1

errorB = [f]

while abs(f-f_last) > 10**-8:
    X = X + delta(X)
    f_last = f
    f = objective(X)
    errorB.append((abs(f-0)))

x1 = 1-2*X[0][0]-3*X[1][0]  # Calculate the dependent point
x2 = X[0][0]
x3 = X[1][0]

print(f"\tClosest Point:{x1}, {x2}, {x3}\t at Distance: {np.sqrt(objective(X))}")

# Create the Convergence Plot
errorA = np.log10(errorA)
plt.plot(range(len(errorA)), errorA)

errorB = np.log10(errorB)
plt.plot(range(len(errorB)), errorB)

plt.title("Log Convergence Chart")
plt.xlabel("Iteration #")
plt.ylabel("log|f - 0|")

plt.show()
