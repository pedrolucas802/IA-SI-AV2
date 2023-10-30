import numpy as np
import matplotlib.pyplot as plt
from util import sign, gerar_dados, divide_data

# Data = np.loadtxt('DataAV2.csv', delimiter=',')
Data = gerar_dados()

# Define the number of runs
num_runs = 100

# Lists to store accuracy results for each run
accuracy_results = []

for run in range(num_runs):
    X, y, X_teste, y_teste = divide_data(Data[:, :-1], Data[:, -1])

    N, p = X.shape
    X = np.concatenate((np.ones((N, 1)), X), axis=1)

    LR = 0.001
    erro = True
    max_epoch = 10
    epoch = 0

    w = np.zeros((p + 1, 1))

    accuracies = []  # List to store accuracies for each run

    # Train the perceptron
    while erro and epoch < max_epoch:
        erro = False
        w_anterior = w
        e = 0
        for t in range(N):
            x_t = X[t, :].reshape((p + 1, 1))
            u_t = (w.T @ x_t)[0, 0]

            y_t = sign(u_t)
            d_t = y[t]

            e_t = int(d_t - y_t)
            w = w + (e_t * x_t * LR) / 2
            if y_t != d_t:
                erro = True
                e += 1

        # Calculate accuracy on the test set
        X_teste = np.concatenate((np.ones((X_teste.shape[0], 1)), X_teste), axis=1)

        correct_predictions = np.sum((X_teste @ w > 0) == (y_teste > 0))
        accuracy = correct_predictions / len(y_teste)
        accuracies.append(accuracy)

        print(f"Run {run + 1}, epoch training: {epoch}")
        epoch += 1

    # Store the accuracy for this run
    accuracy_results.append(accuracies[-1])

# Calculate statistics for all runs
accuracy_results = np.array(accuracy_results)
accuracy_mean = np.mean(accuracy_results)
accuracy_std = np.std(accuracy_results)
max_accuracy = np.max(accuracy_results)
min_accuracy = np.min(accuracy_results)

print(f"Average Accuracy: {accuracy_mean}")
print(f"Standard Deviation of Accuracy: {accuracy_std}")
print(f"Maximum Accuracy: {max_accuracy}")
print(f"Minimum Accuracy: {min_accuracy}")