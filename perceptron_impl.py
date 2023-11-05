import numpy as np

# You may need to replace these imports with your actual utility functions
from util_stats import calc_confusion_matrix, calc_plot_confusion_matrix, calc_accuracy_confusion_matrix
from util import sign, divide_data, plot_results, calculate_accuracy

def perceptron(Data):
    # Split the data into training and testing sets
    X_treino, y_treino, X_teste, y_teste = divide_data(Data[:, :-1], Data[:, -1])

    # Transpose the training data
    X_treino = X_treino.T

    # Add a row of -1s as the first row in the training data
    X_treino = np.concatenate((-np.ones((1, X_treino.shape[1])), X_treino))

    # Learning rate and other hyperparameters
    LR = 0.0001
    max_epoch = 1000

    # Initialize weights with random values
    p = X_treino.shape[0] - 1
    w = np.random.rand(p + 1, 1)

    # Training loop
    for epoch in range(max_epoch):
        erro = False
        e = 0

        for t in range(X_treino.shape[1]):
            x_t = X_treino[:, t].reshape((p + 1, 1))
            u_t = (w.T @ x_t)[0, 0]

            y_t = sign(u_t)
            d_t = y_treino[t]

            e_t = d_t - y_t
            w = w + (e_t * x_t * LR) / 2

            if y_t != d_t:
                erro = True
                e += 1

        # If there are no errors, stop early
        if not erro:
            break

    accuracy = calc_accuracy_confusion_matrix(X_teste, y_teste, w)

    return accuracy, w