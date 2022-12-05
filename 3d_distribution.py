import numpy as np
import matplotlib.pyplot as plt

def ecdf(a):
    """Empirical CDF evaluation of a rv.

    ---- Input:
    :param a: np.ndarray (K,), realization of a rv
    ---- Outputs:
    :return: tuple, collecting the inverse eCDF and the eCDF of the rv
    """
    x, counts = np.unique(a, return_counts=True)
    cusum = np.cumsum(counts)
    return x, cusum / cusum[-1]

# Define cube length
cube_length = 1

# Define number of realizations
num_realizations = 100000

# Prepare to save positions
positions = np.zeros((3, num_realizations))

# Generate positions
positions[0] = cube_length * np.random.rand(num_realizations) - cube_length/2
positions[1] = cube_length/2 * np.random.rand(num_realizations)
positions[2] = cube_length * np.random.rand(num_realizations) - cube_length/2

# Plot
ax = plt.axes(projection='3d')

ax.scatter(positions[0], positions[1], positions[2], c=positions[2], cmap='viridis', linewidth=0.5);

# Distances
distances = np.linalg.norm(positions, axis=0)
pathloss = distances**2



fig, ax = plt.subplots(nrows=2)

x_cdf, y_cdf = ecdf(distances)
ax[0].plot(x_cdf, y_cdf)

x_cdf, y_cdf = ecdf(pathloss)
ax[1].plot(x_cdf, y_cdf)

plt.show()