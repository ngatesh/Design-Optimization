import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

# Define constants
T = 20
a_w = np.array([8.07131, 1730.63, 233.426])
a_d = np.array([7.43155, 1554.679, 240.337])

# Define saturation pressures
p_w = 10 ** (a_w[0] - a_w[1]/(T + a_w[2]))
p_d = 10 ** (a_d[0] - a_d[1]/(T + a_d[2]))

# Define training data
P = np.array([[28.1, 34.4, 36.7, 36.9, 36.8, 36.7, 36.5, 35.4, 32.9, 27.7, 17.5]])
X1 = np.array([[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])
X2 = 1 - X1

# Convert everything into torch tensors for autograd
P = torch.tensor(P, requires_grad=False, dtype=torch.float32)
X1 = torch.tensor(X1, requires_grad=False, dtype=torch.float32)
X2 = torch.tensor(X2, requires_grad=False, dtype=torch.float32)
A = Variable(torch.tensor([1.0, 1.0]), requires_grad=True)

pred = 0    # Predicted curve
loss = 0    # Loss score for predicted curve
a = 0.001   # Learning rate

# Track changes in A to determine stopping point
thisA = np.array([[0, 0]])
lastA = np.array([[1, 1]])
deltaALim = np.ones([1, 2]) * 10**-8

count = 0   # Loop counter

# While change in A is greater than the threshold 'deltaALim', keep training
while np.greater(abs(thisA - lastA), deltaALim).any():
    # Evaluate objective function.
    pred = X1 * torch.exp(A[0] * (A[1]*X2 / (A[0]*X1 + A[1]*X2)) ** 2) * p_w \
         + X2 * torch.exp(A[1] * (A[0]*X1 / (A[0]*X1 + A[1]*X2)) ** 2) * p_d

    # Calculate loss relative to training data
    loss = ((P - pred) ** 2).sum()
    loss.backward()

    # Calculate gradient and update A
    with torch.no_grad():
        A -= a * A.grad
        A.grad.zero_()

        lastA = thisA.copy()
        thisA = A.data.numpy().copy()

    count += 1

# Print results
print(f'A12 : {A[0]}\tA21 : {A[1]}\tLoss : {loss.data.numpy()}\t Iterations: {count}')

# Plot results
pred = pred.detach().numpy()[0]
P = P.detach().numpy()[0]
X1 = X1.detach().numpy()[0]

plt.plot(X1, pred)
plt.plot(X1, P)
plt.legend(['Predicted Pressure', 'Actual Pressure'])
plt.xlabel('X1')
plt.ylabel('Pressure')
plt.title('Predicted Pressure vs. Actual Pressure')
plt.show()
