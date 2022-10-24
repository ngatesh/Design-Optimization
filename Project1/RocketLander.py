from Dynamics import Dynamics
from Controller import Controller
from Optimize import Optimize
from Simulation import Simulation

T = 100  # number of time steps
dim_input = 6  # state space dimensions
dim_hidden = 18  # latent dimensions
dim_output = 2  # action space dimensions
d = Dynamics()  # define dynamics
c = Controller(dim_input, dim_hidden, dim_output)  # define controller
s = Simulation(c, d, T)  # define simulation
o = Optimize(s)  # define optimizer
o.train(30)  # solve the optimization problem