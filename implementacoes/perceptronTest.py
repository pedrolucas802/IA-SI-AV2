import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the sign function
def sign(u):
    if u >= 0:
        return 1
    else:
        return -1

# Read data from CSV file using pandas
data = pd.read_csv('/Users/evanete/Desktop/IASI-AV2/DataAV2.csv')
X = data.iloc[:, :-1].values  # Features
y = data.iloc[:, -1].values  # Labels

N, p = X.shape

# Set up the figure for plotting
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', edgecolors='k', label='Class 1')
plt.scatter(X[y == -1, 0], X[y == -1, 1], color='red', edgecolors='k', label='Class -1')
plt.xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
plt.ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)

X = X.T
X = np.concatenate((-np.ones((1, N)), X))

LR = 0.01  # Increase the learning rate
erro = True
epoch = 0
max_epochs = 1000  # Maximum number of epochs

w = np.zeros((p + 1, 1))  # Fixed missing parenthesis here

while erro and epoch < max_epochs:
    erro = False
    e = 0
    for t in range(N):
        x_t = X[:, t].reshape((p + 1, 1))
        u_t = (w.T @ x_t)[0, 0]
        y_t = sign(u_t)
        d_t = y[t]
        e_t = int(d_t - y_t)
        w = w + (e_t * x_t * LR) / 2
        if y_t != d_t:
            erro = True
            e += 1

    epoch += 1

# Plot the decision boundary
x_boundary = np.linspace(X[1, :].min(), X[1, :].max(), 100)
y_boundary = -(w[0] + w[1] * x_boundary) / w[2]
plt.plot(x_boundary, y_boundary, color='green', linestyle='--', label='Decision Boundary')

plt.legend()
plt.show()

print("Final weights (w):")
print(w)