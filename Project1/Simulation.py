from torch import nn
import torch as t


class Simulation(nn.Module):

    def __init__(self, controller, dynamics, T):
        super(Simulation, self).__init__()
        self.state = self.initialize_state()
        self.controller = controller
        self.dynamics = dynamics
        self.T = T
        self.action_trajectory = []
        self.state_trajectory = []

    def forward(self, state):
        self.action_trajectory = []
        self.state_trajectory = []
        for _ in range(self.T):
            action = self.controller.forward(state)
            state = self.dynamics.forward(state, action)
            self.action_trajectory.append(action)
            self.state_trajectory.append(state)
        return self.error()

    @staticmethod
    def initialize_state():
        state = [1.0, 1.0, 0.17, 0.89, 0.4, -0.001]  # TODO: need batch of initial states
        return t.tensor(state, requires_grad=False).float()

    @staticmethod
    def error1(state):
        return state[0]**2 + state[1]**2 + state[2]**2 + state[3]**2 + state[4]**2 + state[5]**2

    def error(self):
        total = 0
        for i in range(self.T):
            total = total + t.sum(self.state_trajectory[i] ** 2)
        return total
