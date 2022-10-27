from Dynamics import Dynamics
from Controller import Controller
from Simulation import Simulation
from Optimize import Optimize

T = 50  # number of time steps
dim_input = 6  # state space dimensions
dim_hidden = 12  # latent dimensions
dim_output = 2  # action space dimensions
d = Dynamics()  # define dynamics
c = Controller(dim_input, dim_hidden, dim_output)  # define controller

s = Simulation(c, d, T, 1)  # define simulation
o = Optimize(s)
o.train(70)  # solve the optimization problem