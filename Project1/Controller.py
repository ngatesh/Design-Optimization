import torch as t
from torch import nn

MAX_THRUST = 500_000.0  # maximum thrust force [N]
MAX_PHI = 0.52        # maximum thrust angle [rad]. Approx 30deg.


class Controller(nn.Module):

    def __init__(self, dim_input, dim_hidden, dim_output):
        """
        dim_input: Number of system states
        dim_output: Number of actions
        dim_hidden: Width of hidden net
        """
        super(Controller, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.Tanh(),
            nn.Linear(dim_hidden, dim_output)
        )

    def forward(self, state):
        action = self.network(state)
        thrust = t.sigmoid(action[0]) * MAX_THRUST
        phi = t.tanh(action[1]) * MAX_PHI

        return t.stack((thrust, phi), dim=0)
