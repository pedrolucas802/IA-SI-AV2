import numpy as np
from sklearn.linear_model import Perceptron

# Define your dataset
X = np.array([
    [1, 1],
    [0, 1],
    [0, 2],
    [1, 0],
    [2, 2],
    [4, 1.5],
    [1.5, 6],
    [3, 5],
    [3, 3],
    [6, 4],
])

y = np.array([
    [1],
    [1],
    [1],
    [1],
    [1],
    [-1],
    [-1],
    [-1],
    [-1],
    [-1],
])

# Create a Perceptron model
perceptron = Perceptron(max_iter=1000, tol=1e-3)

# Train the Perceptron model
perceptron.fit(X, y)

# Make predictions
predictions = perceptron.predict(X)

# Print the predictions
print("Predictions:")
print(predictions)

# Print the accuracy of the model on the training data
accuracy = (predictions == y).mean()
print("Accuracy on the training data:", accuracy)