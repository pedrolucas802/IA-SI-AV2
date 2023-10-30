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

    return (seq / (2 * N))[0,0]


def embaralhar_dados(
    X: np.ndarray[Any, np.dtype[Any]], y: np.ndarray[Any, np.dtype[Any]]
):
    seed = np.random.permutation(X.shape[0])
    X_random = X[seed, :]
    y_random = y[seed, :]
    return (X_random, y_random)

def processar_dados(
    X: np.ndarray[Any, np.dtype[Any]], y: np.ndarray[Any, np.dtype[Any]]
):
    N, _ = X.shape

    (X_random, y_random) = embaralhar_dados(X, y)
    X_treino = X_random[0 : int(N * 0.8), :]
    y_treino = y_random[0 : int(N * 0.8), :]
    X_teste = X_random[int(N * 0.8) :, :]
    y_teste = y_random[int(N * 0.8) :, :]
    return (X_treino, y_treino, X_teste, y_teste, X_random, y_random)


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


def perceptron_decision_boundary_3d_plotly(X, y, w):
    N, p = X.shape

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X[y == 1, 0], X[y == 1, 1], X[y == 1, 2], color='blue', label='Class 1')
    ax.scatter(X[y == -1, 0], X[y == -1, 1], X[y == -1, 2], color='red', label='Class -1')

    # Create a grid for the decision boundary
    x_grid, y_grid = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100),
                                 np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100))
    z_grid = (-w[0, 0] - w[1, 0] * x_grid - w[2, 0] * y_grid) / w[3, 0]

    # Plot the decision boundary surface
    ax.plot_surface(x_grid, y_grid, z_grid, color='green', alpha=0.3)

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('Perceptron Decision Boundary (3D)')

    plt.legend(loc='best')
    plt.show()