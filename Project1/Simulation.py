from torch import nn
import torch as t
from numpy.random import uniform


class Simulation(nn.Module):

    def __init__(self, controller, dynamics, T, batchSize):
        super(Simulation, self).__init__()
        self.controller = controller
        self.dynamics = dynamics
        self.T = T
        self.batchSize = batchSize
        self.state = self.initialize_state()
        self.action_trajectory = []
        self.state_trajectory = []

    def forward(self, state):
        loss = 0
        for i in range(self.batchSize):
            self.action_trajectory = []
            self.state_trajectory = []

            thisState = state[i]

            for _ in range(self.T):
                action = self.controller.forward(thisState)
                thisState = self.dynamics.forward(thisState, action)
                self.action_trajectory.append(action)
                self.state_trajectory.append(thisState)

            loss = loss + self.error(thisState)

        return loss / self.batchSize

    def initialize_state(self):
        states = []
        for _ in range(self.batchSize):
            xState = uniform(-50, 50)
            yState = uniform(50, 110)
            thetaState = uniform(-0.75, 0.75)
            xDotState = uniform(-20, 20)
            yDotState = uniform(-30, 0)
            thetaDotState = uniform(-0.02, 0.02)

            state = t.tensor([xState, yState, thetaState, xDotState, yDotState, thetaDotState], requires_grad=False)
            #state = t.tensor([1, 1, 0, 0, 0, 0], requires_grad=False).float()
            states.append(state)

        return states

    @staticmethod
    def error(state):
        return state[0]**2 + state[1]**2 + 100*state[2]**2 + state[3]**2 + state[4]**2 + 100*state[5]**2

    def error1(self):
        total = 0
        for i in range(self.T - 1):
            total = total + t.sum(self.state_trajectory[i] ** 2)
        total = total + 1000.0 * t.sum(self.state_trajectory[-1] ** 2)

        return total
