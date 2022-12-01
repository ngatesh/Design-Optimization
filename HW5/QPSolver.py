import numpy as np


class QPSolver:
    def __init__(self, X, _fx, _h, _hx, _g, _gx, W):
        fx = _fx(X)
        h = _h(X)
        hx = _hx(X)
        g = _g(X)
        gx = _gx(X)

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
