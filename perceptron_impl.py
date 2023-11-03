import numpy as np

from util_stats import calc_confusion_matrix
from util import sign, divide_data, plot_results, calculate_accuracy

# matplotlib.use("TkAgg")
def perceptron(Data):
    X_treino, y_treino, X_teste, y_teste = divide_data(Data[:, :-1], Data[:, -1])

    N, p = X_treino.shape
    X_treino = X_treino.T
    X_treino = np.concatenate((-np.ones((1, N)), X_treino))

    LR = 0.0001
    erro = True
    max_epoch = 400
    epoch = 0

    # w = np.zeros((p + 1, 1))
    w = np.random.rand(p + 1, 1)


    while erro and epoch < max_epoch:
        erro = False
        w_anterior = w
        e = 0
        for t in range(N):
            x_t = X_treino[:, t].reshape((p + 1, 1))
            u_t = (w.T @ x_t)[0, 0]

            y_t = sign(u_t)
            # d_t = y[t, 0]
            d_t = y_treino[t]

            e_t = int(d_t - y_t)
            w = w + (e_t * x_t * LR) / 2
            if y_t != d_t:
                erro = True
                e += 1

        # print("epoch training: "+str(epoch))
        epoch += 1

    # print(epoch)
    # print("Training weights (w):")
    # print(w)

    # plot_results(X_teste,y_teste, w)

    # print("X_teste")
    # print(X_teste)
    # print("y_teste")
    # print(y_teste)

    return calculate_accuracy(X_teste, y_teste, w), w

    # print("Accuracy: {:.2f}%".format(accuracy_percentage))