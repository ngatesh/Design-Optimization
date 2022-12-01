import numpy as np


class LineSearch:
    def __init__(self, X, s, _f, _h, _g, lam, mu):
        self.X = X
        self.s = s
        self.f = _f
        self.h = _h
        self.g = _g

        self.lam = lam
        self.mu = mu
        self.wh = np.abs(lam)
        self.wg = np.abs(mu)

    def meritF(self):
        f = self.f(self.X)
        sumH = np.matmul(self.wh.T, np.abs(self.h))
        sumG = np.matmul(self.wg.T, np.maximum(self.g, np.zeros(np.size(self.g))))

        return f + sumH + sumG

    def weightUpdate(self):
        self.wh = np.max(np.abs(self.wh), 0.5 * (self.wh + self.lam))
        self.wg = np.max(np.abs(self.wg), 0.5 * (self.wg + self.mu))
