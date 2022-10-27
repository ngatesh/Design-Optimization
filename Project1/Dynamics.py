from torch import nn
import torch as t

FRAME_TIME = 0.2      # time interval [s].
GRAVITY_ACCEL = 9.81  # gravity constant [m/s^2].

# Rocket constants sourced roughly from the Falcon 9.
MASS = 25_000         # mass of rocket [kg].
MOMENT = 3_300_000    # angular moment of inertia [kgm^2].
ARM = 20              # distance from thruster to center of mass [m].

# Masks to represent the locations of states within the state vector.
xDotMask = t.tensor([0, 0, 0, 1, 0, 0], requires_grad=False).float()
yDotMask = t.tensor([0, 0, 0, 0, 1, 0], requires_grad=False).float()
thetaDotMask = t.tensor([0, 0, 0, 0, 0, 1], requires_grad=False).float()


class Dynamics(nn.Module):
    def __init__(self):
        super(Dynamics, self).__init__()

    @staticmethod
    def forward(state, action):
        """
        state: [x, y, theta, x_dot, y_dot, theta_dot]
        action: [thrust, phi]
        """

        F = action[0]       # Engine thrust [N].
        phi = action[1]     # Thrust angle [rad].
        theta = state[2]    # Rocket angle [rad].

        # Calculate accelerations.
        d_xDot = -F * t.sin(theta - phi) / MASS * FRAME_TIME
        d_yDot = (F * t.cos(theta - phi) / MASS - GRAVITY_ACCEL) * FRAME_TIME
        d_thetaDot = F * t.sin(phi) * ARM / MOMENT * FRAME_TIME

        # Apply accelerations to update the speed states.
        state = state + (xDotMask * d_xDot) + (yDotMask * d_yDot) + (thetaDotMask * d_thetaDot)

        # Apply speeds to update the position states.
        stepMatrix = t.tensor([[1, 0, 0, FRAME_TIME, 0, 0],
                               [0, 1, 0, 0, FRAME_TIME, 0],
                               [0, 0, 1, 0, 0, FRAME_TIME],
                               [0, 0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 1, 0],
                               [0, 0, 0, 0, 0, 1]], requires_grad=False)

        state = t.matmul(stepMatrix, state)

        return state
