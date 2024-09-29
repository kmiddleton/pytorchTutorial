import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# 0) Prepare data
np.random.seed(326636)
X_numpy = np.random.rand(500, 1) * 10
X_numpy = np.sort(X_numpy, axis=0)

# Logistic growth equation parameters
L = 10  # The curve's maximum value
k = 1   # The logistic growth rate or steepness of the curve
x0 = 5  # The x-value of the sigmoid's midpoint

# Generate y values using the logistic growth equation
y_numpy = L / (1 + np.exp(-k * (X_numpy - x0)))
noise = np.random.normal(0, 0.25, y_numpy.shape)
y_numpy += noise
y_numpy = np.abs(y_numpy)

# # Plot y vs X
# plt.scatter(X_numpy, y_numpy, color='red', label='Original data')
# plt.xlabel('X')
# plt.ylabel('y')
# plt.title('Original Data')
# plt.legend()
# plt.show()

# cast to float Tensor
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1)

n_samples, n_features = X.shape

# 1) Model
# Linear model f = wx + b
input_size = n_features
output_size = 1

# Define a simple neural network with one hidden layer
class NonlinearModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NonlinearModel, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x

hidden_size = 100  # You can adjust the hidden layer size as needed
model = NonlinearModel(input_size, hidden_size, output_size)

# 2) Loss and optimizer
learning_rate = 0.01

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

# 3) Training loop
num_epochs = 20000
loss_values = []

for epoch in range(num_epochs):
    # Forward pass and loss
    y_predicted = model(X)
    loss = criterion(y_predicted, y)
    
    # Backward pass and update
    loss.backward()
    optimizer.step()

    # zero grad before new step
    optimizer.zero_grad()

    # Store loss value for plotting
    loss_values.append(loss.item())

    if (epoch+1) % 500 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot loss vs. epoch starting from epoch 500 on the first subplot
ax1.plot(range(500, num_epochs), loss_values[500:], label='Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Loss vs. Epoch (Starting from Epoch 500)')
ax1.legend()

# Plot original data and model predictions on the second subplot
X_test = np.linspace(0.01, 10, 200).reshape(-1, 1).astype(np.float32)
X_test_tensor = torch.from_numpy(X_test)
predicted_test = model(X_test_tensor).detach().numpy()

ax2.plot(X_numpy, y_numpy, 'ro', label='Original data')
ax2.plot(X_test, predicted_test, 'bo', label='Model predictions')
ax2.set_xlabel('X')
ax2.set_ylabel('y')
ax2.set_title('Original Data vs. Model Predictions')
ax2.legend()

# Show the figure
plt.tight_layout()
plt.show()
