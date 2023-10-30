from typing import Any
from random import random as rd
import numpy as np
import matplotlib.pyplot as plt


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