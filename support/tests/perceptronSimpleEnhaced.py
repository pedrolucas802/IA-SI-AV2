import numpy as np
import matplotlib.pyplot as plt

def sign(u):
    if u >= 0:
        return 1
    else:
        return -1

# Define the data
X = np.array([
    [1, 1],
    [0, 1],
    [0, 2],
    [1, 0],
    [2, 2],
    [4, 1.5],
    [1.5, 6],
    [3, 5],
    [3, 3],
    [6, 4],
])

y = np.array([
    [1],
    [1],
    [1],
    [1],
    [1],
    [-1],
    [-1],
    [-1],
    [-1],
    [-1],
])

N, p = X.shape

# Normalize the data
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / X_std

# Add a column of ones to the feature matrix
ones_column = np.ones((N, 1))
X = np.concatenate((ones_column, X), axis=1)

# Learning rate and other parameters
initial_LR = 0.1
LR = initial_LR
max_epochs = 100
convergence_threshold = 0.01

# Initialize w with zeros
w = np.zeros((p + 1, 1))

# Initialize variables for tracking convergence
converged = False
convergence_counter = 0

for epoch in range(max_epochs):
    errors = 0
    for t in range(N):
        x_t = X[t, :].reshape((p + 1, 1))
        u_t = (w.T @ x_t)[0, 0]
        y_t = sign(u_t)
        d_t = y[t, 0]
        e_t = d_t - y_t
        w = w + LR * e_t * x_t

        if e_t != 0:
            errors += 1

    if errors == 0:
        converged = True
        break

    if initial_LR > 0.01:
        LR = initial_LR / (1 + epoch)  # Reduce the learning rate over time

    if errors < convergence_threshold * N:
        convergence_counter += 1
    else:
        convergence_counter = 0

    if convergence_counter >= 3:
        break

# Plot the decision boundary line using the final weights
x_boundary = np.array([X[:, 1].min(), X[:, 1].max()])
y_boundary = (-w[0, 0] - w[1, 0] * x_boundary) / w[2, 0]

plt.scatter(X[y[:, 0] == 1, 1], X[y[:, 0] == 1, 2], color='blue', edgecolors='k')
plt.scatter(X[y[:, 0] == -1, 1], X[y[:, 0] == -1, 2], color='red', edgecolors='k')
plt.plot(x_boundary, y_boundary, color='green', label='Decision Boundary')
plt.xlim(X[:, 1].min() - 0.1, X[:, 1].max() + 0.1)
plt.ylim(X[:, 2].min() - 0.1, X[:, 2].max() + 0.1)
plt.show()

print("Final weights (w):")
print(w)