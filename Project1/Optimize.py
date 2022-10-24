from torch import optim
import numpy as np
import matplotlib.pyplot as plt


class Optimize:
    def __init__(self, simulation):
        self.simulation = simulation
        self.parameters = simulation.controller.parameters()
        self.optimizer = optim.LBFGS(self.parameters, lr=0.01)

    def step(self):
        def closure():
            loss = self.simulation(self.simulation.state)
            self.optimizer.zero_grad()
            loss.backward()
            return loss

        self.optimizer.step(closure)
        return closure()

    def train(self, epochs):
        for epoch in range(epochs):
            loss = self.step()
            print('[%d] loss: %.3f' % (epoch + 1, loss))
        self.visualize()

    def visualize(self):
        data = np.array([self.simulation.state_trajectory[i].detach().numpy() for i in range(self.simulation.T)])
        x = data[:, 0]
        y = data[:, 1]
        theta = data[:, 2]
        xDot = data[:, 3]
        yDot = data[:, 4]
        thetaDot = data[:, 5]
        time = [i for i in range(self.simulation.T)]

        plt.plot(time, x)
        plt.plot(time, y)
        plt.plot(time, theta)
        plt.plot(time, xDot)
        plt.plot(time, yDot)
        plt.plot(time, thetaDot)
        plt.legend(['X', 'Y', 'Theta', 'Vx', 'Vy', 'w'])
        plt.show()

