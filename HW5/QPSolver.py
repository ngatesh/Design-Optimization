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
        h_bar = h
        activeList = np.array([[]], dtype=int).reshape(0, 1)

        while True:
            A_size = np.size(A, 0)
            row1 = np.concatenate((W, A.T), axis=1)
            row2 = np.concatenate((A, np.zeros((A_size, A_size))), axis=1)

            Big = np.concatenate((row1, row2), axis=0)
            small = np.concatenate((-fx, -h_bar), axis=0)

            ans = np.matmul(np.linalg.inv(Big), small)
            Na = np.size(ans, 0)

            s = ans[0:Nx, 0].reshape(Nx, 1)
            lam = ans[Nx:Nh, 0].reshape(Nh-Nx, 1)
            mu = ans[Nh+Nx:Na, 0].reshape(Na-Nh-Nx, 1)

            Nmu = np.size(mu, 0)

            mostNegativeIndex = -1

            for i in range(Nmu):
                if (mu[i, 0] < -10**-8) and (mostNegativeIndex == -1 or mu[i, 0] < mu[mostNegativeIndex, 0]):
                    mostNegativeIndex = i

            if mostNegativeIndex != -1:
                A = np.delete(A, mostNegativeIndex + Nh, 0)
                h_bar = np.delete(h_bar, mostNegativeIndex + Nh, 0)

                activeList = np.delete(activeList, mostNegativeIndex, 0)
                continue

            mostPositiveIndex = -1

            g_next = np.matmul(gx, s) + g
            for i in range(Ng):
                if (g_next[i, 0] > 10**-8) and (mostPositiveIndex == -1 or g_next[i, 0] > g_next[mostPositiveIndex, 0]):
                    mostPositiveIndex = i

            if mostPositiveIndex != -1:
                gxRow = np.array([gx[mostPositiveIndex]])
                gVal = np.array([g[mostPositiveIndex]])

                A = np.concatenate((A, gxRow), 0)
                h_bar = np.concatenate((h_bar, gVal), 0)

                activeList = np.concatenate((activeList, np.array([[mostPositiveIndex]])), 0)
                continue

            numActive = np.size(activeList)

            def gActive(x):
                g_all = _g(x)
                g_active = np.zeros((numActive, 1))

                for j in range(numActive):
                    g_active[j] = g_all[activeList[j, 0]]

                return g_active

            def gxActive(x):
                gx_all = _gx(x)
                gx_active = np.zeros((numActive, Nx))

                for j in range(numActive):
                    gx_active[j] = gx_all[activeList[j]]

                return gx_active

            return [s, lam, mu, gActive, gxActive]
