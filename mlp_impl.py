import numpy as np
from organiza_imagens import organize_images

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def tanH(x):
    return (1 - np.exp(-x)) / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def sigmoid_derivative(x):
    return x * (1 - x)

def calculate_EQM(X_train, Y, W, L):
    EQM = 0

    for i in range(X_train.shape[1]):
        x = X_train[:, i].reshape(-1, 1)
        forward(x, W, y)
        d = Y[:, i]

        EQI = 0

        for j in range(len(y[L])):
            EQI += (d[j] - y[L][j]) ** 2

        EQM += EQI

    EQM /= (2 * X_train.shape[1])

    return EQM

def forward(x, w, y):
    for j in range(len(w)):
        if j == 0:
            i = np.dot(w[j], x)
            y[j] = tanh(i)
        else:
            y_bias = np.concatenate((np.array([[-1]]), y[j - 1]))
            i = np.dot(w[j], y_bias)
            y[j] = tanh(i)
    return y

def backward(W, x, d, delta, lr):
    j = len(W) - 1

    while j >= 0:
        if j + 1 == len(W):
            delta[j] = tanh(y[j]) * (d - y[j])
            y_bias = y[j - 1]
            y_bias = np.concatenate((np.array([[-1] * y_bias.shape[1]]), y_bias), axis=0)
            W[j] = W[j] + lr * (delta[j] @ y_bias.T)  # Matrix multiplication using @
        elif j == 0:
            Wb = W[j + 1].T[:, 1:]
            delta[j] = tanh(y[j]) * (delta[j + 1] @ Wb)
            W[j] = W[j] + lr * (delta[j] @ x.T)  # Matrix multiplication using @
        else:
            Wb = W[j + 1].T[:, 1:]
            delta[j] = tanh(y[j]) * (delta[j + 1] @ Wb.T)
            y_bias = np.concatenate((np.array([[-1] * y[j - 1].shape[1]]), y[j - 1]), axis=0)
            W[j] = W[j] + lr * (delta[j] @ y_bias.T)  # Matrix multiplication using @
        j -= 1

X, Y = organize_images()
print(X)
print(Y)

L = 3
q = [6, 6, 8, 20]
lr = 0.001
max_epochs = 10
pr = 1e-15

W = [None] * (L + 1)
i = [None] * (L + 1)
y = [None] * (L + 1)
delta = [None] * (L + 1)

X_train = np.vstack((np.ones((1, X.shape[1])), X))

for i in range(L + 1):
    if i == 0:
        W[i] = np.random.random_sample((q[i], X_train.shape[0])) - 0.5
    else:
        W[i] = np.random.random_sample((q[i], q[i - 1] + 1)) - 0.5

print(W)

EQM = 1
epoch = 0

while EQM > pr and epoch < max_epochs:
    EQM = 0
    for i in range(X_train.shape[1]):
        x = X_train[:, i].reshape(-1, 1)
        forward(x, W, y)
        d = Y[:, i]
        # backward(W, x, d, delta, lr)
    EQM = calculate_EQM(X_train, Y, W, L)
    epoch += 1
    print("Epoch:", epoch, "EQM:", EQM)

print("Training completed. Final EQM:", EQM)