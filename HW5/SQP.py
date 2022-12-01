import numpy as np
from QPSolver import QPSolver
from LineSearch import LineSearch


class SQP:
    @staticmethod
    def solve(X0, f, h, g, fx, hx, gx, epsilon=10**-3):
        X = X0
        W = np.eye(np.size(X0))
        gradL = 10

        while np.linalg.norm(gradL) > epsilon:
            [s, lam, mu] = QPSolver.solve(X, fx, h, hx, g, gx, W)

            alpha = LineSearch(X, s, f, fx, h, hx, g, gx, lam, mu).search()

            W = SQP.quasiNewtonW(X, alpha * s, g, W)

            X = X + alpha * s

            gradL = fx(X) + np.matmul(lam.T, hx(X)) + np.matmul(mu.T, gx(X))

        return [X, f(X)]

    @staticmethod
    def quasiNewtonW(X0, s, g, W0):
        X1 = X0 + s

        g0 = g(X0)
        g1 = g(X1)
        y = g1 - g0

        # todo: check PD

        a = np.matmul(y, y.T)
        b = np.matmul(y.T, s)
        c = np.matmul(np.matmul(W0, s), np.matmul(s.T, W0))
        d = np.matmul(np.matmul(s.T, W0), s)

        return W0 + (a / b) - (c / d)
