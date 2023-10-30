import numpy as np
import matplotlib.pyplot as plt
from util import sign, gerar_dados, divide_data, shuffle_data, plot_results, plot_scatter


def perceptron(LR, max_epoch, X_treino, y_treino, N,p, w):
    X_treino = X_treino.T
    X_treino = np.concatenate((-np.ones((1, N)), X_treino))

    epoch = 0
    erro = True

    while erro and epoch < max_epoch:
        erro = False
        w_anterior = w
        e = 0
        for t in range(N):
            x_t = X_treino[:, t].reshape((p + 1, 1))
            u_t = (w.T @ x_t)[0, 0]
            y_t = sign(u_t)
            d_t = y_treino[t]
            e_t = int(d_t - y_t)
            w = w + (e_t * x_t * LR) / 2
            if y_t != d_t:
                erro = True
                e += 1
        print("epoch: " + str(epoch))
        epoch += 1

    return w


# Data = np.loadtxt('DataAV2.csv', delimiter=',')
Data = gerar_dados()

X_treino, y_treino, X_teste, y_teste = divide_data(Data[:, :-1], Data[:, -1])

plot_scatter(X_treino, y_treino)
plot_scatter(X_teste, y_teste)

N_treino, p_treino = X_treino.shape
N_test, p_test = X_teste.shape

LR = 0.001
max_epoch = 50
w = np.zeros((p_treino + 1, 1))

w_training = perceptron(LR, max_epoch, X_treino, y_treino, N_treino,p_treino, w)

w_test = perceptron(LR, max_epoch, X_treino, y_treino, N_test,p_test, w_training)

plot_results(w_test)

print("Final weights (w):")
print(w_training)
