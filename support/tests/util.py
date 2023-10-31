import numpy as np

# Your array of arrays
array_of_arrays = [
    [[-0.066], [0.01468805], [0.01359335]],
    [[-0.066], [0.01468805], [0.01359335]],
    [[-0.066], [0.01468805], [0.01359335]]
]

# Convert the array of arrays to a NumPy array
array = np.array(array_of_arrays)

# Calculate the mean of each column
column_means = np.mean(array, axis=0)

# Convert the result back to a list
result = column_means.tolist()

print(result)