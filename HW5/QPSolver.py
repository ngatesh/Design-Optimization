import numpy as np


class QPSolver:
    @staticmethod
    def solve(X, _fx, _h, _hx, _g, _gx, W):
        """
        Solves a given quadratic programming problem with active set strategy

        :param X: current X (column vector)
        :param _fx: objective gradient (function)
        :param _h: equality constraint (function)
        :param _hx: equality constraint gradient (function)
        :param _g: inequality constraint (function)
        :param _gx: inequality constraint gradient (function)
        :param W: hessian matrix of Lagrangian
        :returns:
        [s, lam, mu, gActive, gxActive]
        s: step size (column vector).
        lam: lambda's for equality constraints (column vector).
        mu: mu's for active inequality constraints (column vector).
        gActive: active inequality constraints (function).
        gxActive: active inequality constraint gradient (function).
        """

        # Get all function values at current X
        fx = _fx(X)
        h = _h(X)
        hx = _hx(X)
        g = _g(X)
        gx = _gx(X)

        Nx = np.size(X, 0)  # Length of X vector
        Nh = np.size(h, 0)  # Number of equality constraints (# rows of matrix)
        Ng = np.size(g, 0)  # Number of inequality constraints (# rows of matrix)

        A = hx      # Active set gradient. Initialized to only equality constraints.
        h_bar = h   # Active set. Initialized to only equality constraints.
        activeList = np.array([[]], dtype=int).reshape(0, 1)    # Tracks order of active inequality constraints.

        # Just keep looping until the function returns.
        while True:
            # Set up matrix equation.
            # [ W A ]       [-fx   ]
            # [ A 0 ] * X = [-h_bar]  ==>   Big * X = small

            A_size = np.size(A, 0)                                          # Number of active constraints
            row1 = np.concatenate((W, A.T), axis=1)                         # First row of Big:  [W A]
            row2 = np.concatenate((A, np.zeros((A_size, A_size))), axis=1)  # Second row of Big: [A 0]

            Big = np.concatenate((row1, row2), axis=0)      # Construct Big matrix
            small = np.concatenate((-fx, -h_bar), axis=0)   # Construct small matrix

            ans = np.matmul(np.linalg.inv(Big), small)      # Solve linear problem
            Na = np.size(ans, 0)                            # Length of answer vector [s, lam's, mu's].T

            s = ans[0:Nx, 0].reshape(Nx, 1)                 # Get step vector
            lam = ans[Nx:Nh, 0].reshape(Nh-Nx, 1)           # Get lambda vector
            mu = ans[Nh+Nx:Na, 0].reshape(Na-Nh-Nx, 1)      # Get mu vector

            Nmu = np.size(mu, 0)    # Number of mu's (active inequality constraints)

            mostNegativeIndex = -1  # Index of most negative value within mu vector

            # Search for most negative mu.
            for i in range(Nmu):
                # -10^-8 is the threshold here because nothing is ever numerically ZERO.
                if (mu[i, 0] < -10**-8) and (mostNegativeIndex == -1 or mu[i, 0] < mu[mostNegativeIndex, 0]):
                    mostNegativeIndex = i

            # Delete most negative mu (if one was found) along with corresponding constraint.
            if mostNegativeIndex != -1:
                A = np.delete(A, mostNegativeIndex + Nh, 0)
                h_bar = np.delete(h_bar, mostNegativeIndex + Nh, 0)
                activeList = np.delete(activeList, mostNegativeIndex, 0)

                # Re-solve and try again.
                continue

            mostPositiveIndex = -1          # Index of most positive inequality constraint after step.
            g_next = np.matmul(gx, s) + g   # Calculate inequality constraint values after step.

            # Search for most positive (violated) inequality constraint after the step is taken.
            for i in range(Ng):
                if (g_next[i, 0] > 10**-8) and (mostPositiveIndex == -1 or g_next[i, 0] > g_next[mostPositiveIndex, 0]):
                    mostPositiveIndex = i

            # Add most positive constraint (if one was found) to active set.
            if mostPositiveIndex != -1:
                gxRow = np.array([gx[mostPositiveIndex]])
                gVal = np.array([g[mostPositiveIndex]])

                A = np.concatenate((A, gxRow), 0)
                h_bar = np.concatenate((h_bar, gVal), 0)
                activeList = np.concatenate((activeList, np.array([[mostPositiveIndex]])), 0)

                # Re-solve and try again.
                continue

            # If the loop gets this far, that means the correct active set has been found and no more re-solves.

            numActive = np.size(activeList)  # Number of active constraints.

            # Define a new inequality constraint function that returns just the active constraints.
            def gActive(x):
                g_all = _g(x)
                g_active = np.zeros((numActive, 1))

                for j in range(numActive):
                    g_active[j] = g_all[activeList[j, 0]]

                return g_active

            # Define a new inequality constraint gradient function that returns just the active constraint gradients.
            def gxActive(x):
                gx_all = _gx(x)
                gx_active = np.zeros((numActive, Nx))

                for j in range(numActive):
                    gx_active[j] = gx_all[activeList[j]]

                return gx_active

            # Return the X-step, lambda's, mu's, and active inequality functions.
            return [s, lam, mu, gActive, gxActive]
