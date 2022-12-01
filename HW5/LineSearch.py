import numpy as np


class LineSearch:
    def __init__(self, X, s, _f, _fx, _h, _hx, _g, _gx, lam, mu):
        self.X = X
        self.s = s
        self.f = _f
        self.fx = _fx
        self.h = _h
        self.hx = _hx
        self.g = _g
        self.gx = _gx

        self.h_alpha = self.h_alpha()
        self.g_alpha = self.g_alpha()
        self.f_alpha = self.f_alpha()

        self.lam = lam
        self.mu = mu
        self.wh = np.abs(lam)
        self.wg = np.abs(mu)

    def search(self, t=0.7):
        alpha = 1
        F = self.meritF(alpha)
        Q = self.Q(alpha, t)

        while F > Q:
            alpha = alpha / 2
            Q = self.Q(alpha, t)

        return alpha

    def meritF(self, alpha):
        X_step = self.X + alpha * self.s
        f = self.f(X_step)
        h = self.h(X_step)
        g = self.g(X_step)

        sumH = np.matmul(self.wh.T, np.abs(h))
        sumG = np.matmul(self.wg.T, np.maximum(g, np.zeros(np.size(g))))

        return f + sumH + sumG

    def Q(self, alpha, t):
        return self.meritF(0) + t * alpha * self.F_alpha()

    def F_alpha(self):
        return self.f_alpha + np.matmul(self.wh, self.h_alpha) + np.matmul(self.wg, self.g_alpha)

    def f_alpha(self):
        return np.matmul(self.fx.T, self.s)

    def h_alpha(self):
        h = self.h(self.X)
        hx = self.hx(self.X)

        return np.multiply(np.matmul(hx, self.s), np.sign(h))

    def g_alpha(self):
        X_step = self.X
        g = self.g(X_step)
        gx = self.gx(X_step)

        g_alpha = np.matmul(gx, self.s)
        g_alpha[g < 0] = 0

        return g_alpha

    def weightUpdate(self):
        self.wh = np.max(np.abs(self.wh), 0.5 * (self.wh + self.lam))
        self.wg = np.max(np.abs(self.wg), 0.5 * (self.wg + self.mu))
