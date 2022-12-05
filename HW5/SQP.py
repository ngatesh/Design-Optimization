import numpy as np
from QPSolver import QPSolver
from LineSearch import LineSearch


# Sequential Quadratic Programming Algorithm
class SQP:
    @staticmethod
    def solve(X0, f, h, g, fx, hx, gx, epsilon=10**-5):
        """
        Solves an SQP problem with the given objective and constraints.

        :param X0: initial guess (column vector)
        :param f: objective (function)
        :param h: equality constraint (function)
        :param g: inequality constraint (function)
        :param fx: gradient of objective (function)
        :param hx: gradient of equality constraint (function)
        :param gx: gradient of inequality constraint (function)
        :param epsilon: stop condition triggers when the norm of the gradient of the Lagrangian falls below this value
        :return: [X, f(X)]
        """

        print("SQP Solver:")

        X = X0                      # initial guess
        W = np.eye(np.size(X0))     # initial hessian defined as identity matrix
        gradL = 10                  # gradient tracker

        # Main program loop. Run until the gradient of the lagrangian is small enough.
        while np.linalg.norm(gradL) > epsilon:
            # Solve the QP sub-problem and get the step direction, lambda's, mu's,
            # active inequality constraints, and active inequality gradients.
            [s, lam, mu, gActive, gxActive] = QPSolver.solve(X, fx, h, hx, g, gx, W)

            # Perform line search to get the best step size.
            alpha = LineSearch(X, s, f, fx, h, hx, gActive, gxActive, lam, mu).search()

            # Update the hessian matrix using a quasi-newton method.
            W = SQP.quasiNewtonW(X, alpha * s, W, fx, hx, gxActive, lam, mu)

            # Perform the step.
            X = X + alpha * s

            # Update gradient of Lagrangian.
            gradL = fx(X) + np.matmul(lam.T, hx(X)).T + np.matmul(mu.T, gxActive(X)).T

            print(f'\tGradient Norm: {np.linalg.norm(gradL):0.6f}')

        return [X, f(X)]

    @staticmethod
    def quasiNewtonW(X0, s, W0, fx, hx, gx, lam, mu):
        """
        Estimates the hessian of the Lagrangian between two points: X0 and X0 + s

        :param X0: initial X (column vector)
        :param s: step (column vector)
        :param W0: previous hessian estimate
        :param fx: objective gradient (function)
        :param hx: equality constraint gradient (function)
        :param gx: inequality constraint gradient (function)
        :param lam: associated lambda's
        :param mu: associated mu's
        :return: the new hessian estimate (W)
        """

        X1 = X0 + s  # Step X forward

        Lx0 = fx(X0) + np.matmul(lam.T, hx(X0)).T + np.matmul(mu.T, gx(X0)).T   # Gradient at first point (X0)
        Lx1 = fx(X1) + np.matmul(lam.T, hx(X1)).T + np.matmul(mu.T, gx(X1)).T   # Gradient at second point (X1)

        y = Lx1 - Lx0   # Change in gradient

        sy = np.matmul(s.T, y)[0, 0]                    # s.T*y : scalar used in computation
        sws = np.matmul(s.T, np.matmul(W0, s))[0, 0]    # s.T*W*S : scalar used in computation

        # Making sure s.T*y is positive definite
        theta = 1 if sy >= 0.2*sws else 0.8 * sws / (sws - sy)
        y = theta * y + (1-theta) * np.matmul(W0, s)

        # Calculate new hessian estimate
        yy = np.matmul(y, y.T)
        ys = np.matmul(y.T, s)
        wssw = np.matmul(np.matmul(W0, s), np.matmul(s.T, W0))

        return W0 + (yy / ys) - (wssw / sws)
