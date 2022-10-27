import torch as t
from torch import nn

# Rocket constants sourced roughly from the Falcon 9.
MAX_THRUST = 500_000.0  # maximum thrust force [N]
MAX_PHI = 0.52          # maximum thrust angle [rad]. Approx 30deg.


class Controller(nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_output):
        """
        :param dim_input: Number of system states.
        :param dim_hidden: Number of action states.
        :param dim_output: Width of hidden net.
        """

        super(Controller, self).__init__()

        # Define neural net with two hidden layers and hyperbolic-tangent activations.
        self.network = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.Tanh(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.Tanh(),
            nn.Linear(dim_hidden, dim_output)
        )

    # Produce the control action given a system state.
    def forward(self, state):
        action = self.network(state)    # Get network result.

        # Activate thrust with sigmoid to constrain from 0 to MAX_THRUST.
        thrust = t.sigmoid(action[0]) * MAX_THRUST
        # Activate thrust angle with tanh to constrain between +/- MAX_PHI.
        phi = t.tanh(action[1]) * MAX_PHI

        return t.stack((thrust, phi), dim=0)
