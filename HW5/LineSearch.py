import numpy as np


class LineSearch:
    def __init__(self, X, s, _f, _fx, _h, _hx, _g, _gx, lam, mu):
        """
        Initializes a Line Search object to search in the given step direction
        :param X: starting X (column vector)
        :param s: step direction (column vector)
        :param _f: objective (function)
        :param _fx: objective gradient (function)
        :param _h: equality constraint (function)
        :param _hx: equality constraint gradient (function)
        :param _g: inequality constraint (function)
        :param _gx: inequality constraint gradient (function)
        :param lam: equality constraint lambda's
        :param mu: inequality constraint mu's
        """

        # Save instance variables
        self.X = X
        self.s = s
        self.f = _f
        self.fx = _fx
        self.h = _h
        self.hx = _hx
        self.g = _g
        self.gx = _gx
        self.lam = lam
        self.mu = mu

        # Calculate alpha-gradients once.
        self.h_alpha = self.h_alpha()
        self.g_alpha = self.g_alpha()
        self.f_alpha = self.f_alpha()

        # Initialize merit function weights.
        self.wh = np.abs(lam)
        self.wg = np.abs(mu)

    def search(self, t=0.7):
        """
        Perform line search

        :param t: Q step scale factor
        :return: alpha
        """

        # Line search. Scale back alpha until Q(alpha) > F(alpha).
        alpha = 1
        F = self.meritF(alpha)
        Q = self.Q(alpha, t)

        while F > Q:
            self.weightUpdate()

            alpha = alpha / 2
            F = self.meritF(alpha)
            Q = self.Q(alpha, t)

        return alpha

    def meritF(self, alpha):
        """
        Merit function, evaluated at X + alpha * s

        :param alpha: step scale factor
        """

        X_step = self.X + alpha * self.s    # Step X forward
        f = self.f(X_step)                  # Objective value
        h = self.h(X_step)                  # Equality constraint value
        g = self.g(X_step)                  # Inequality constraint value

        g_size = np.size(g, 0)              # Number of inequality constraints

        # Apply weights to h and g
        sumH = np.matmul(self.wh.T, np.abs(h))
        sumG = np.matmul(self.wg.T, np.maximum(g, np.zeros((g_size, 1))))

        # Add everything up
        return f + sumH + sumG

    def Q(self, alpha, t):
        """
        Query function: Q = F(0) + t * alpha * dF/d_alpha

        :param alpha: step scale factor
        :param t: query step scale factor
        """
        return self.meritF(0) + t * alpha * self.F_alpha()

    def F_alpha(self):
        """
        Derivative of merit function F with respect to alpha
        """
        return self.f_alpha + np.matmul(self.wh.T, self.h_alpha) + np.matmul(self.wg.T, self.g_alpha)

    def f_alpha(self):
        """
        Derivative of objective function f with respect to alpha
        """
        return np.matmul(self.fx(self.X).T, self.s)

    def h_alpha(self):
        """
        Derivative of equality constraint function h with respect to alpha
        """
        h = self.h(self.X)
        hx = self.hx(self.X)

        return np.multiply(np.matmul(hx, self.s), np.sign(h))

    def g_alpha(self):
        """
        Derivative of inequality constraint function g with respect to alpha
        """
        g = self.g(self.X)
        gx = self.gx(self.X)

        g_alpha = np.matmul(gx, self.s)
        g_alpha[g < 0] = 0  # Positive (non-violated) constraints not counted

        return g_alpha

    def weightUpdate(self):
        """
        Update weights in the merit function
        """
        self.wh = np.maximum(np.abs(self.wh), 0.5 * (self.wh + np.abs(self.lam)))
        self.wg = np.maximum(np.abs(self.wg), 0.5 * (self.wg + np.abs(self.mu)))
