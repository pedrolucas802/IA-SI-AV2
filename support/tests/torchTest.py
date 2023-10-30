import torch
import numpy as np
import matplotlib.pyplot as plt
from util import sign, gerar_dados

def run_perceptron_with_mps_device():
    # Set the device to "mps" if available, or "cpu" if not
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Data = np.loadtxt('DataAV2.csv', delimiter=',')
    Data = gerar_dados()

    # Convert data to PyTorch tensors and send to the specified device
    X = torch.tensor(Data[:, :-1], dtype=torch.float32).to(device)
    y = torch.tensor(Data[:, -1], dtype=torch.float32).to(device)

    N, p = X.shape

    plt.scatter(X[y == 1, 0].cpu(), X[y == 1, 1].cpu(), color='blue', edgecolors='k', label='Class 1')
    plt.scatter(X[y == -1, 0].cpu(), X[y == -1, 1].cpu(), color='red', edgecolors='k', label='Class -1')

    # Add a row of ones for bias
    ones = torch.ones(N, 1, device=device)
    X = torch.cat([ones, X], dim=1)

    LR = 0.001
    max_epoch = 50

    # Initialize weights using PyTorch
    w = torch.zeros(p + 1, 1, dtype=torch.float32, device=device)

    for epoch in range(max_epoch):
        # Compute the predicted labels
        y_pred = torch.matmul(X, w)
        y_pred = torch.sign(y_pred)

        # Compute the errors
        errors = y - y_pred

        # Update the weights
        w += LR * torch.matmul(X.t(), errors)

        # Compute the classification error
        classification_error = torch.sum(errors != 0)

        print(f"Epoch: {epoch}, Classification Error: {classification_error}")

        if classification_error == 0:
            break

    x_axis = torch.linspace(-15, 8, 100, device=device)
    x2 = (-w[0] - w[1] * x_axis) / w[2]

    # Convert tensors to CPU for plotting
    x_axis = x_axis.cpu()
    x2 = x2.cpu()

    plt.plot(x_axis, x2, color='green')
    plt.show()

    print("Final weights (w):")
    print(w.cpu())

# Call the method to run the perceptron with the "mps" device
run_perceptron_with_mps_device()