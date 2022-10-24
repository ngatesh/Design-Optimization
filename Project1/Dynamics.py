from torch import nn
import torch as t

FRAME_TIME = 0.1      # time interval
GRAVITY_ACCEL = 9.81  # gravity constant [m/s^2]

MASS = 25_000         # mass of rocket [kg]
MOMENT = 3_300_000    # angular moment of inertia [kgm^2]
ARM = 20              # distance from thruster to center of mass [m]

FMask = t.tensor([1, 0], requires_grad=False).float().T
phiMask = t.tensor([0, 1], requires_grad=False).float().T

xMask = t.tensor([1, 0, 0, 0, 0, 0], requires_grad=False).float().T
yMask = t.tensor([0, 1, 0, 0, 0, 0], requires_grad=False).float().T
thetaMask = t.tensor([0, 0, 1, 0, 0, 0], requires_grad=False).float().T

xDotMask = t.tensor([0, 0, 0, 1, 0, 0], requires_grad=False).float().T
yDotMask = t.tensor([0, 0, 0, 0, 1, 0], requires_grad=False).float().T
thetaDotMask = t.tensor([0, 0, 0, 0, 0, 1], requires_grad=False).float().T


class Dynamics(nn.Module):

    def __init__(self):
        super(Dynamics, self).__init__()

    @staticmethod
    def forward(state, action):
        """
        state: [x, y, theta, x_dot, y_dot, theta_dot]
        action: [thrust, phi]
        """

        F = t.matmul(FMask.T, action)
        phi = t.matmul(phiMask.T, action)
        theta = t.matmul(thetaMask.T, state)

        d_xDot = -F * t.sin(theta - phi) / MASS * FRAME_TIME
        d_yDot = (F * t.cos(theta - phi) / MASS - GRAVITY_ACCEL) * FRAME_TIME
        d_thetaDot = F * t.sin(phi) * ARM / MOMENT * FRAME_TIME

        state = state + (xDotMask * d_xDot) + (yDotMask * d_yDot) + (thetaDotMask * d_thetaDot)

        stepMatrix = t.tensor([[1, 0, 0, FRAME_TIME, 0, 0],
                               [0, 1, 0, 0, FRAME_TIME, 0],
                               [0, 0, 1, 0, 0, FRAME_TIME],
                               [0, 0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 1, 0],
                               [0, 0, 0, 0, 0, 1]], requires_grad=False)

        state = t.matmul(stepMatrix, state)

        return state
