from torch import optim
import numpy as np
import matplotlib.pyplot as plt
import Dynamics


class Optimize:
    def __init__(self, simulation):
        # Set up optimizer.
        self.simulation = simulation
        self.parameters = simulation.controller.parameters()
        self.optimizer = optim.LBFGS(self.parameters, lr=0.01)

    def step(self):
        """
        Calculate loss and update parameters through the optimizer.

        :return: loss
        """
        def closure():
            loss = self.simulation(self.simulation.states)
            self.optimizer.zero_grad()
            loss.backward()
            return loss

        self.optimizer.step(closure)
        return closure()

    def train(self, epochs):
        """
        :param epochs: Number of times to step forward the optimizer
        """

        print(f"Initial State: {self.simulation.states}")

        for epoch in range(epochs):
            loss = self.step()
            print('[%d] loss: %.3f' % (epoch + 1, loss))

        print(f"Final State: {self.simulation.state_trajectory[-1]}")
        self.visualize()

    # Plot out the final state and action trajectories.
    def visualize(self):
        stateData = np.array([self.simulation.state_trajectory[i].detach().numpy() for i in range(self.simulation.T)])
        x = stateData[:, 0]
        y = stateData[:, 1]
        theta = stateData[:, 2] * 180.0 / np.pi
        xDot = stateData[:, 3]
        yDot = stateData[:, 4]
        thetaDot = stateData[:, 5] * 180.0 / np.pi
        time = [i*Dynamics.FRAME_TIME for i in range(self.simulation.T)]

        actionData = np.array([self.simulation.action_trajectory[i].detach().numpy() for i in range(self.simulation.T)])
        F = actionData[:, 0]
        phi = actionData[:, 1] * 180.0 / np.pi

        plt.subplot(2, 2, 1)
        plt.plot(time, x)
        plt.plot(time, y)
        plt.plot(time, xDot)
        plt.plot(time, yDot)
        plt.legend(['X [m]', 'Y [m]', 'Vx [m/s]', 'Vy[m/s]'])
        plt.title("State Space Variables Over Time")

        plt.subplot(2, 2, 2)
        plt.plot(time, F)
        plt.legend(['Thrust [N]'])
        plt.title("Action Space Variables Over Time")

        plt.subplot(2, 2, 3)
        plt.plot(time, theta)
        plt.plot(time, thetaDot)
        plt.legend(['theta [deg]', 'w [deg/s]'])
        plt.xlabel("Time [sec]")

        plt.subplot(2, 2, 4)
        plt.plot(time, phi)
        plt.legend(['Thrust Angle [deg]'])
        plt.xlabel("Time [sec]")

        plt.show()
