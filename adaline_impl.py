import numpy as np
from util import EQM, divide_data, plot_results, calculate_accuracy

# matplotlib.use("TkAgg")

def adaline(Data):

    X_treino, y_treino, X_teste, y_teste = divide_data(Data[:, :-1], Data[:, -1])


    N, p = X_treino.shape

    X_treino = X_treino.T

    X_treino = np.concatenate((
        -np.ones((1, N)), X_treino
    ))

    lr = 1e-4
    pr = 1e-15

    maxEpoch = 1000

    epoch = 0
    EQM1 = 1
    EQM2 = 0

    # w = np.zeros((p + 1, 1))
    w = np.random.random_sample((p + 1, 1))-.5

    while (epoch < maxEpoch and abs(EQM1 - EQM2) > pr):
        EQM1 = EQM(X_treino, y_treino, w)
        for t in range(N):
            x_t = X_treino[:, t].reshape(p + 1, 1)
            u_t = w.T @ x_t
            d_t = y_treino[t]
            e_t = (d_t - u_t)
            w = w + lr * e_t * x_t

        # if(epoch == 0):
        #     x_axis = np.linspace(-15, 8, 100)
        #     x2 = w[0, 0] / w[2, 0] - x_axis * (w[1, 0] / w[2, 0])
        #     plt.plot(x_axis, x2, color='yellow')

        epoch += 1
        # print("epoch: "+str(epoch))
        EQM2 = EQM(X_treino, y_treino, w)


    # plot_results(X_teste,y_teste, w)

    return calculate_accuracy(X_teste,y_teste, w), w, epoch
