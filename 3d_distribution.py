##################################################
# 3D Distribution
#
# Description: Study the distribution over users over a 3D grid.
##################################################
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rc

rc('font', **{'family': 'sans serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


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
cube_length = 20

# Define number of realizations
num_realizations = 100000

##################################################
# Method 1 - uniform at random
#
# Note that the origin is placed in (0,0,0).
##################################################

# Prepare to save positions
positions = np.zeros((3, num_realizations))

# Generate positions
positions[0] = cube_length * np.random.rand(num_realizations) - cube_length
positions[1] = cube_length/2 * np.random.rand(num_realizations)
positions[2] = cube_length * np.random.rand(num_realizations) - cube_length/2

# Plot
ax = plt.axes(projection='3d')

ax.scatter(positions[0], positions[1], positions[2], c=positions[2], cmap='viridis', linewidth=0.5)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')

plt.tight_layout()

# Distances
distances = np.linalg.norm(positions, axis=0)
elevation_angles = np.arctan2(positions[2], positions[0])   # elevation angle: x-z plane
azimuth_angles = np.arctan2(positions[1], positions[0])   # azimuth angle: x-y plane

pathloss = 10 * np.log(1 / distances**2)

fig, ax = plt.subplots(nrows=3)

x_cdf, y_cdf = ecdf(distances)
ax[0].plot(x_cdf, y_cdf)

ax[0].set_xlabel('distance [m]')
ax[0].set_ylabel('CDF')

x_cdf, y_cdf = ecdf(elevation_angles)
ax[1].plot(x_cdf, y_cdf, label='elevation')

x_cdf, y_cdf = ecdf(azimuth_angles)
ax[1].plot(x_cdf, y_cdf, label='azimuth')

ax[1].legend()
ax[1].set_xlabel('angle')
ax[1].set_ylabel('CDF')

x_cdf, y_cdf = ecdf(pathloss)
ax[2].plot(x_cdf, y_cdf)

ax[2].set_xlabel('pathloss [dB]')
ax[2].set_ylabel('CDF')

ax[0].set_title('Method 1')

plt.tight_layout()

bs_pos = np.array([[20, 0, 0]])

plt.show()