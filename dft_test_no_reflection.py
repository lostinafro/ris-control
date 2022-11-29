import numpy as np
import matplotlib.pyplot as plt

from scipy.constants import speed_of_light

# Number of antennas at the RIS
num_antennas = 100

# Define fundamental frequency
w = np.exp(-1j * 2 * np.pi / num_antennas)

# Compute DFT matrix
J, K = np.meshgrid(np.arange(num_antennas), np.arange(num_antennas))
DFT = np.power(w, J * K)
DFT = np.real(DFT)

plt.imshow(DFT)
plt.show()

# Eletromagnetic
carrier_frequency = 3e9
wavelength = speed_of_light / carrier_frequency

# Location of the antennas
el_dis = wavelength/2

# Prepare to save positions of each antenna element
el_pos = np.zeros((num_antennas, 3))

# Go through each antenna
for mm in range(num_antennas):

    index_i = mm % 10
    index_j = np.floor(mm/10)

    el_pos[mm, 0] = el_dis * index_i
    el_pos[mm, 1] = el_dis * index_j


# Create a vector of sensing locations at different azimuth angles
azimuth_angles = np.linspace(-np.pi/2, np.pi/2, 1001)

# Save array response or steering vector
steering_vector = np.zeros((len(azimuth_angles), num_antennas), dtype=np.complex_)

# Go through all azimuth angles evaluated
for aa, azimuth_angle in enumerate(azimuth_angles):

    # Define wavevector
    wavevector = 2 * np.pi / wavelength * np.array([np.cos(azimuth_angle), np.sin(azimuth_angle), 0])

    # Go through each antenna
    for mm in range(num_antennas):

        steering_vector[aa, mm] = -np.exp(1j * (wavevector * el_pos[mm]).sum())

# Compute reference steering vector
azimuth_angle = 0
ref_wavevector = 2 * np.pi / wavelength * np.array([np.cos(azimuth_angle), np.sin(azimuth_angle), 0])

# Prepare to save reference steering vector
ref_steering_vector = np.zeros((num_antennas), dtype=np.complex_)

# Go through each antenna
for mm in range(num_antennas):
    ref_steering_vector[mm] = -np.exp(1j * (ref_wavevector * el_pos[mm]).sum())


normalized_array_response = np.abs((ref_steering_vector[np.newaxis, :] * steering_vector).sum(axis=-1)) / num_antennas
normalized_array_response_db = 10 * np.log10(normalized_array_response)

fig, ax = plt.subplots()

ax.plot(azimuth_angles, normalized_array_response_db, color='black')

plt.show()