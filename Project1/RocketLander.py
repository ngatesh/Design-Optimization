from Dynamics import Dynamics
from Controller import Controller
from Simulation import Simulation
from Optimize import Optimize

T = 50              # Number of time steps
dim_input = 6       # State space dimensions
dim_hidden = 12     # Width of hidden layer
dim_output = 2      # Action space dimensions

d = Dynamics()                                      # Define dynamics
c = Controller(dim_input, dim_hidden, dim_output)   # Define controller
s = Simulation(c, d, T, 2)                          # Define simulation
o = Optimize(s)                                     # Define optimizer

o.train(60)  # Solve the optimization problem
