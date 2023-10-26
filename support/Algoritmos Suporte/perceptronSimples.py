import numpy as np
import matplotlib.pyplot as plt

def sign(u):
    if u >= 0:
        return 1
    else:
        return -1

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

plt.scatter(X[0:5, 0], X[0:5, 1], color='blue', edgecolors='k')
plt.scatter(X[5:, 0], X[5:, 1], color='red', edgecolors='k')
plt.xlim(-0.25, 6.2)
plt.ylim(-0.25, 6.2)

X = X.T

X = np.concatenate((
    -np.ones((1, N)), X
))

LR = 0.001
erro = True
epoch = 0

# Initialize w with zeros
w = np.zeros((p + 1, 1))

while erro:
    erro = False
    w_anterior = w
    e = 0
    for t in range(N):
        x_t = X[:, t].reshape((p + 1, 1))
        u_t = (w.T @ x_t)[0, 0]

        y_t = sign(u_t)
        d_t = y[t, 0]
        e_t = int(d_t - y_t)
        w = w + (e_t * x_t * LR) / 2
        if y_t != d_t:
            erro = True
            e += 1

    epoch += 1

plt.show()

print("Final weights (w):")
print(w)