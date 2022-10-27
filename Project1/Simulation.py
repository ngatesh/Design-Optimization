from torch import nn
import torch as t
from numpy.random import uniform


class Simulation(nn.Module):

    def __init__(self, controller, dynamics, T, batchSize):
        """
        :param controller: Control system object.
        :param dynamics: Dynamics definition object.
        :param T: Number of time steps to simulate.
        :param batchSize: Number of simulations to optimize simultaneously.
        """
        super(Simulation, self).__init__()

        # Instance variables.
        self.controller = controller
        self.dynamics = dynamics
        self.T = T
        self.batchSize = batchSize

        # Generate {batchSize} initial states to simulate.
        self.states = self.initialize_states()

        # Storage for state and action trajectories through all time steps.
        self.action_trajectory = []
        self.state_trajectory = []

    def forward(self, states):
        """
        Run the simulation to completion, given a set of initial states.

        :param states: Initial states.
        :return: Loss value.
        """
        loss = 0    # Loss value placeholder.

        # Run through every initial condition in the batch of initial states.
        for i in range(self.batchSize):
            thisState = states[i]   # Initial state to simulate.

            # Reset trajectories.
            self.action_trajectory = []
            self.state_trajectory = []

            # Loop through all time steps.
            for _ in range(self.T):
                action = self.controller.forward(thisState)             # Determine action from the controller.
                thisState = self.dynamics.forward(thisState, action)    # Update system state through dynamics.

                # Append new action and state to the trajectories.
                self.action_trajectory.append(action)
                self.state_trajectory.append(thisState)

            # Calculate the error in the final state, then add it to the total loss.
            loss = loss + self.error(thisState)

        # Normalize loss to the batch size.
        return loss / self.batchSize

    def initialize_state(self):
        """
        :return: an array of initial state tensors, randomly generated.
        """
        states = []
        for _ in range(self.batchSize):
            # Generate random states values.
            xState = uniform(-50, 50)
            yState = uniform(50, 110)
            thetaState = uniform(-0.75, 0.75)
            xDotState = uniform(-20, 20)
            yDotState = uniform(-30, 0)
            thetaDotState = uniform(-0.02, 0.02)

            # Combine into a tensor object and add to state array.
            state = t.tensor([xState, yState, thetaState, xDotState, yDotState, thetaDotState], requires_grad=False)
            # state = t.tensor([1, 1, 0, 0, 0, 0], requires_grad=False).float()
            states.append(state)

        return states

    @staticmethod
    def error(state):
        """
        :param state: System state
        :return: Weighted error
        """
        return state[0]**2 + state[1]**2 + 100*state[2]**2 + state[3]**2 + state[4]**2 + 100*state[5]**2
