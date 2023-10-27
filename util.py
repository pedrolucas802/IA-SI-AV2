from typing import Any

import numpy as np


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

