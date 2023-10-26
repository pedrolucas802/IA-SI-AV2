import numpy as np
import matplotlib.pyplot as plt

def EQM(X, y, w):
    seq = 0
    us = []
    p, N = X.shape
    for t in range(X.shape[1]):
        x_t = X[:, t].reshape(X.shape[0], 1)
        u_t = w.T @ x_t
        us.append(u_t)
        d_t = y[t]  # Remove the unnecessary indexing
        seq += (d_t - u_t) ** 2

    return seq / (2 * X.shape[1])

Data = np.loadtxt('/DataAV2.csv', delimiter=',')
X = Data[:, :-1]
y = Data[:, -1]

N, p = X.shape

X = X.T

X = np.concatenate((
    -np.ones((1, N)), X
))

lr = 1e-2
pr = 1e-7

maxEpoch = 1000

epoch = 0
EQM1 = 1
EQM2 = 0

w = np.zeros((p + 1, 1))

while (epoch < maxEpoch and abs(EQM1 - EQM2) > pr):
    EQM1 = EQM(X, y, w)
    for t in range(N):
        x_t = X[:, t].reshape(p + 1, 1)
        u_t = w.T @ x_t
        d_t = y[t]
        e_t = (d_t - u_t)
        w = w + lr * e_t * x_t

    epoch += 1
    print("epoch: "+str(epoch))
    EQM2 = EQM(X, y, w)


plt.scatter(X[1, y == 1], X[2, y == 1], color='blue', edgecolors='k')
plt.scatter(X[1, y == -1], X[2, y == -1], color='red', edgecolors='k')

x_boundary = np.array([X[1, :].min() - 1, X[1, :].max() + 1])
y_boundary = (-w[0, 0] - w[1, 0] * x_boundary) / w[2, 0]
plt.plot(x_boundary, y_boundary, color='green')

plt.show()