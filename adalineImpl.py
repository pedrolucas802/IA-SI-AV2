import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from util import EQM

# matplotlib.use("TkAgg")

Data = np.loadtxt('DataAV2.csv', delimiter=',')
X = Data[:, :-1]
y = Data[:, -1]

N, p = X.shape

plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', edgecolors='k', label='Class 1')
plt.scatter(X[y == -1, 0], X[y == -1, 1], color='red', edgecolors='k', label='Class -1')

plt.xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
plt.ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)

X = X.T

X = np.concatenate((
    -np.ones((1, N)), X
))

lr = 1e-4
pr = 1e-12

maxEpoch = 1000

epoch = 0
EQM1 = 1
EQM2 = 0

# w = np.zeros((p + 1, 1))
w = np.random.random_sample((p + 1, 1))-.5

while (epoch < maxEpoch and abs(EQM1 - EQM2) > pr):
    EQM1 = EQM(X, y, w)
    for t in range(N):
        x_t = X[:, t].reshape(p + 1, 1)
        u_t = w.T @ x_t
        d_t = y[t]
        e_t = (d_t - u_t)
        w = w + lr * e_t * x_t

    if(epoch == 0):
        x_axis = np.linspace(-15, 8, 100)
        x2 = w[0, 0] / w[2, 0] - x_axis * (w[1, 0] / w[2, 0])
        plt.plot(x_axis, x2, color='yellow')

    epoch += 1
    print("epoch: "+str(epoch))
    EQM2 = EQM(X, y, w)

x_axis = np.linspace(-15,8,100)
x2 = w[0,0]/w[2,0] - x_axis*(w[1,0]/w[2,0])
plt.plot(x_axis, x2, color='green')

plt.show()
# y = w1x1 + w2x2 -w0