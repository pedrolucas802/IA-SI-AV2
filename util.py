from datetime import datetime
from typing import Any
from random import random as rd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt


def sign(u):
    if u >= 0:
        return 1
    else:
        return -1


def EQM(X, y, w):
    seq = 0
    us = []
    p, N = X.shape
    for t in range(N):
        x_t = X[:, t].reshape(X.shape[0], 1)
        u_t = w.T @ x_t
        us.append(u_t)
        d_t = y[t]
        seq += (d_t - u_t) ** 2

    return (seq / (2 * N))[0, 0]


def shuffle_data(X, y):
    np.random.seed(0)
    N = X.shape[0]
    seed = np.random.permutation(N)
    X_random = X[seed]
    y_random = y[seed]
    return X_random, y_random

def divide_data(X, y, train_rt):
    X_random, y_random = shuffle_data(X, y)
    N = X_random.shape[0]
    N_train = int(N * train_rt)
    X_treino = X_random[:N_train, :]
    y_treino = y_random[:N_train]
    X_teste = X_random[N_train:, :]
    y_teste = y_random[N_train:]
    return X_treino, y_treino, X_teste, y_teste


def gerar_dados():
    np.random.seed(0)

    n_samples = 1200
    mean1 = [np.random.rand() + 2, np.random.rand() - 2]
    cov1 = [[np.random.rand() + 1, 0], [0, np.random.rand() + 1]]
    d1 = np.random.multivariate_normal(mean1, cov1, n_samples)
    r1 = np.ones((n_samples, 1))

    mean3 = [0, np.random.rand() - 2]
    cov3 = [[np.random.rand() + 0.2, np.random.rand() - 0.2], [np.random.rand() - 0.2, np.random.rand() + 0.2]]
    d2n = np.random.multivariate_normal(mean3, cov3, int(n_samples * 0.25)) - 0.9
    r2n = np.ones((int(n_samples * 0.25), 1))

    mean2 = [np.random.rand() - 4, np.random.rand() - 4]
    cov2 = [[np.random.rand() + 1, np.random.rand() - 0.8], [np.random.rand() - 0.8, np.random.rand() + 1]]
    d2 = np.random.multivariate_normal(mean2, cov2, n_samples) - 2.2
    r2 = -np.ones((n_samples, 1))

    mean3 = [0, 0]
    cov3 = [[0.2, np.random.rand() - 0.2], [np.random.rand() + 0.2, -0.4]]
    d1n = np.random.multivariate_normal(mean3, cov3, int(n_samples * 0.55)) - 2.9
    r1n = -np.ones((int(n_samples * 0.55), 1))

    X = np.concatenate((d1, d2n, d2, d1n))
    Y = np.concatenate((r1, r2n, r2, r1n))
    return np.array(np.concatenate((X, Y), axis=1))


def printar_progresso(valor):
    inicio = datetime.now()
    agora = datetime.now()
    delta = agora - inicio
    print(
        f"\rProgresso de classificação: {valor:.2%}. Tempo decorrido: {int(delta.total_seconds())} segundos.",
        end="",
    )

def plot_results(X, y, w):
    x_axis = np.linspace(-15, 8, 100)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', edgecolors='k', label='Class 1')
    plt.scatter(X[y == -1, 0], X[y == -1, 1], color='red', edgecolors='k', label='Class -1')

    plt.xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
    plt.ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
    x2 = w[0, 0] / w[2, 0] - x_axis * (w[1, 0] / w[2, 0])
    plt.plot(x_axis, x2, color='green')

    plt.show()


def plot_scatter(X, y):
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', edgecolors='k', label='Class 1')
    plt.scatter(X[y == -1, 0], X[y == -1, 1], color='red', edgecolors='k', label='Class -1')

    plt.xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
    plt.ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)


def calculate_accuracy(X_test, y_test, w):
    correct = 0
    total = len(y_test)

    for i in range(len(X_test)):
        x_t = np.concatenate((np.array([-1]), X_test[i]))
        u_t = np.dot(w.T, x_t)[0]
        y_t = np.sign(u_t)

        if y_t > 0 and y_test[i] == 1:
            correct += 1
        elif y_t <= 0 and y_test[i] == -1:
            correct += 1

    accuracy = (correct / total) * 100
    # print(f"Accuracy: {accuracy:.2f}%")

    return accuracy



def print_stats(data):

    mean_accuracy = np.mean(data)
    std_deviation = np.std(data)
    max_accuracy = np.max(data)
    min_accuracy = np.min(data)

    print("Mean Accuracy:", mean_accuracy)
    print("Standard Deviation:", std_deviation)
    print("Maximum Accuracy:", max_accuracy)
    print("Minimum Accuracy:", min_accuracy)








