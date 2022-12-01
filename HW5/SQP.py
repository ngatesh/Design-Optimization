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

            W = SQP.quasiNewtonW(X, alpha * s, W, fx, hx, gx, lam, mu)

            X = X + alpha * s

            gradL = fx(X) + np.matmul(lam.T, hx(X)) + np.matmul(mu.T, gx(X))

        return [X, f(X)]

    @staticmethod
    def quasiNewtonW(X0, s, W0, fx, hx, gx, lam, mu):
        X1 = X0 + s

        Lx0 = fx(X0) + np.matmul(lam.T, hx(X0)) + np.matmul(mu.T, gx(X0))
        Lx1 = fx(X1) + np.matmul(lam.T, hx(X1)) + np.matmul(mu.T, gx(X1))

        y = Lx1 - Lx0

        sy = np.matmul(s.T, y)
        sws = np.matmul(s.T, np.matmul(W0, s))

        theta = 1 if sy >= 0.2*sws else 0.8 * sws / (sws - sy)

        y = theta * y + (1-theta) * np.matmul(W0, s)

        a = np.matmul(y, y.T)
        b = np.matmul(y.T, s)
        c = np.matmul(np.matmul(W0, s), np.matmul(s.T, W0))
        d = np.matmul(np.matmul(s.T, W0), s)

        return W0 + (a / b) - (c / d)
