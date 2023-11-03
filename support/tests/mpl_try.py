import numpy as np

from organiza_imagens import organize_images


def forward():
    print("hello")

def backward():
    print("hello")

X, Y = organize_images()
print(X)
print(Y)

L = 3
q = [2, 3, 2,20]
learning_rate = 0.01
max_epochs = 10
pr = 1e-15

W = [None] * L
i = [None] * L
y = [None] * L
delta = [None] * L

X_train = np.vstack((np.ones((1, X.shape[1])), X))

for i in range(L + 1):
    if i == 0:
        W.append(np.random.random_sample((q[i], X_train.shape[0])) - 0.5)
    elif i == L:
        W.append(np.random.random_sample((q[L], q[i - 1] + 1)) - 0.5)
    else:
        W.append(np.random.random_sample((q[i], q[i - 1] + 1)) - 0.5)

print(W)

EQM = 1
epoch = 0

while EQM > error_threshold and epoch < max_epochs:
    EQM = 0
    for i in range(X_train.shape[1]):
        x = X_train[:, i].reshape(-1, 1)

        forward()

        d = Y[:, i].reshape(-1, 1)

        backward()

    # EQM = EQM()
    epoch += 1


print("Treinamento concluído.")