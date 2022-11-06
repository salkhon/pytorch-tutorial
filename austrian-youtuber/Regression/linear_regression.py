# 1. Design model (input, output size, forward pass)
# 2. Construct loss and optimizer
# 3. Training loop
#   - forward pass: computer prediction and less
#   - backward prop: gradients
#   - update weights
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# 0. Prepare data
X_numpy, Y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)  # type: ignore
X = torch.from_numpy(X_numpy.astype(np.float32))
Y = torch.from_numpy(Y_numpy.astype(np.float32))

y = Y.view(Y.shape[0], 1)

n_samples, n_features = X.shape

# 1. model
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)

# 2. loss and optimizer
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3. Training loop
num_epochs = 100
for epoch in range(num_epochs):
    # forward pass
    y_pred = model(X)
    loss = criterion(y_pred, y)

    # back prop
    loss.backward()

    # update weights
    optimizer.step()

    optimizer.zero_grad()

    if epoch + 1 % 10 == 0:
        print(f"epoch: {epoch+1}, loss = {loss.item():.4f}")

# plot
predicted = model(X).detach().numpy()
plt.plot(X_numpy, Y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()
