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
# plt.xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
# plt.ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
plt.xlim(-30.25, 30.2)
plt.ylim(-30.25, 30.2)

X = X.T
X = np.concatenate((
    -np.ones((1, N)), X
))

LR = 0.001
epoch = 99

np.random.seed(0)
w = np.random.rand(p + 1, 1)

erro = True

while erro and epoch > 0:
    erro = False
    e = 0
    for t in range(N):
        x_t = X[:, t].reshape((p + 1, 1))
        u_t = (w.T @ x_t)[0, 0]

        y_t = sign(u_t)
        d_t = y[t, 0]

        # e_t = int(d_t - y_t)
        # w = w + (e_t * x_t * LR) / 2

        if y_t != d_t:
            w = w + (d_t - y_t) * x_t * LR
            erro = True
            e += 1

    epoch -= 1

x_boundary = np.array([X[1, :].min() - 1, X[1, :].max() + 1])
y_boundary = (-w[0, 0] - w[1, 0] * x_boundary) / w[2, 0]

plt.plot(x_boundary, y_boundary, color='green', label='Decision Boundary')

plt.show()

print("Final weights (w):")
print(w)