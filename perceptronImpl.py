import numpy as np
import matplotlib.pyplot as plt
from util import sign, gerar_dados, perceptron_decision_boundary_3d_plotly

# matplotlib.use("TkAgg")


# Data = np.loadtxt('DataAV2.csv', delimiter=',')
Data = gerar_dados()

# def perceptron(data, LR, )

X = Data[:, :-1]
y = Data[:, -1]

N, p = X.shape

plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', edgecolors='k', label='Class 1')
plt.scatter(X[y == -1, 0], X[y == -1, 1], color='red', edgecolors='k', label='Class -1')

# plt.xlim(-0.25, 6.2)
# plt.ylim(-0.25, 6.2)
plt.xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
plt.ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)


X = X.T

X = np.concatenate((
    -np.ones((1, N)), X
))

LR = 0.001
erro = True
max_epoch = 100
epoch = 0

w = np.zeros((p + 1, 1))

while erro and epoch < max_epoch:
    erro = False
    w_anterior = w
    e = 0
    for t in range(N):
        x_t = X[:, t].reshape((p + 1, 1))
        u_t = (w.T @ x_t)[0, 0]

        y_t = sign(u_t)
        # d_t = y[t, 0]
        d_t = y[t]

        e_t = int(d_t - y_t)
        w = w + (e_t * x_t * LR) / 2
        if y_t != d_t:
            erro = True
            e += 1

    print("epoch: "+str(epoch))
    epoch += 1


x_axis = np.linspace(-15,8,100)
x2 = w[0,0]/w[2,0] - x_axis*(w[1,0]/w[2,0])
plt.plot(x_axis, x2, color='green')

plt.show()
# y = w1x1 + w2x2 -w0

# perceptron_decision_boundary_3d_plotly(X, y, w)

print("Final weights (w):")
print(w)