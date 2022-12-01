import numpy as np
from QPSolver import QPSolver
from LineSearch import LineSearch


class SQP:
    @staticmethod
    def solve(X0, f, h, g, fx, hx, gx, epsilon=10**-5):
        X = X0
        W = np.eye(np.size(X0))
        gradL = 10

        while np.linalg.norm(gradL) > epsilon:
            print("QPSolver...")
            [s, lam, mu, gActive, gxActive] = QPSolver.solve(X, fx, h, hx, g, gx, W)

            print("LineSearch...")
            alpha = LineSearch(X, s, f, fx, h, hx, gActive, gxActive, lam, mu).search()

            print("quasiNewtonW...")
            W = SQP.quasiNewtonW(X, alpha * s, W, fx, hx, gxActive, lam, mu)

            X = X + alpha * s

            gradL = fx(X) + np.matmul(lam.T, hx(X)).T + np.matmul(mu.T, gxActive(X)).T

        return [X, f(X)]

    @staticmethod
    def quasiNewtonW(X0, s, W0, fx, hx, gx, lam, mu):
        X1 = X0 + s

        Lx0 = fx(X0) + np.matmul(lam.T, hx(X0)).T + np.matmul(mu.T, gx(X0)).T
        Lx1 = fx(X1) + np.matmul(lam.T, hx(X1)).T + np.matmul(mu.T, gx(X1)).T

        y = Lx1 - Lx0

        sy = np.matmul(s.T, y)[0, 0]
        sws = np.matmul(s.T, np.matmul(W0, s))[0, 0]

        theta = 1 if sy >= 0.2*sws else 0.8 * sws / (sws - sy)

        y = theta * y + (1-theta) * np.matmul(W0, s)

        a = np.matmul(y, y.T)
        b = np.matmul(y.T, s)
        c = np.matmul(np.matmul(W0, s), np.matmul(s.T, W0))
        d = np.matmul(np.matmul(s.T, W0), s)

        return W0 + (a / b) - (c / d)
