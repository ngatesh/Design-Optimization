import numpy as np


class QPSolver:
    @staticmethod
    def solve(X, _fx, _h, _hx, _g, _gx, W):
        fx = _fx(X)
        h = _h(X)
        hx = _hx(X)
        g = _g(X)
        gx = _gx(X)
        W = W

        Nx = np.size(X, 0)
        Nh = np.size(h, 0)
        Ng = np.size(g, 0)

        A = hx

        while True:
            h1 = np.concatenate((W, A.T), axis=1)
            h2 = np.concatenate((A, np.zeros(np.size(A, 0))), axis=1)

            Big = np.concatenate((h1, h2), axis=0)
            small = np.concatenate((-fx, -h), axis=0)

            ans = np.matmul(np.invert(Big), small)
            Na = np.size(ans, 0)

            s = ans[0:Nx, 0]
            lam = ans[Nx:Nh, 0]
            mu = ans[Nh:Na, 0]

            mostNegativeIndex = -1

            for i in range(Nh-Na):
                if (mu[i, 0] < 0) and (mostNegativeIndex == -1 or mu[i, 0] < mu[mostNegativeIndex, 0]):
                    mostNegativeIndex = i

            if mostNegativeIndex != -1:
                A = np.delete(A, mostNegativeIndex + Nh)
                continue

            mostPositiveIndex = -1

            g_next = np.matmul(gx, s) + g
            for i in range(Ng):
                if (g_next[i, 0] > 0) and (mostPositiveIndex == -1 or g_next[i, 0] > g_next[mostPositiveIndex, 0]):
                    mostPositiveIndex = i

            if mostPositiveIndex != -1:
                A = np.concatenate((A, gx[mostPositiveIndex, 0]), 0)
                continue

            return [s, lam, mu]
