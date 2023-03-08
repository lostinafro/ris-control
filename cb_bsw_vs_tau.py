import numpy as np

import matplotlib.pyplot as plt

import scenario.common as cmn
from environment import RIS2DEnv, command_parser, NOISE_POWER_dBm, T, TAU, TX_POW_dBm

from mpl_toolkits.axes_grid1 import make_axes_locatable

##################################################
# Parameters
##################################################

# Seed random generator
np.random.seed(42)

# Define transmit power [mW]
tx_power = cmn.dbm2watt(TX_POW_dBm)

# Get noise power
noise_power = cmn.dbm2watt(NOISE_POWER_dBm)

# Define number of pilots
num_pilots = 1

# Define minimum SNR
minimum_snr_dB = 14
minimum_snr = cmn.db2lin(minimum_snr_dB)

# Total tau
total_tau = TAU

# Parameter for saving datas
prefix = 'data/cb_bsw_vs_tau'

# CB type
cb_type = 'fixed'
# cb_type = 'flexible'

# Setup option
setups = ['ob-cc', 'ib-no', 'ib-wf']

# For grid mesh
num_users = int(1e3)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # The following parser is used to impose some data without the need of changing the script (run with -h flag for
    # help) Render bool needs to be True to save the data If no arguments are given the standard value are loaded (
    # see environment) datasavedir should be used to save numpy arrays
    render, side_x, h, name, datasavedir = command_parser()
    prefix = prefix + name

    # Define length of the cube
    cube_length = side_x

    # Drop some users: the RIS is assumed to be in the middle of the bottom face of the cube.
    x = cube_length * np.random.rand(num_users, 1) - cube_length
    y = cube_length / 2 * np.random.rand(num_users, 1)
    z = cube_length * np.random.rand(num_users, 1) - cube_length / 2

    # Get position of the users and position of the BS
    ue_pos = np.hstack((x, y, z))
    bs_pos = np.array([[20, 5, 0]])

    # Plot setup
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(ue_pos[:, 0], ue_pos[:, 1], ue_pos[:, 2], marker='o', color='black', alpha=0.1, label='UE')
    ax.scatter(bs_pos[:, 0], bs_pos[:, 1], bs_pos[:, 2], marker='^', label='BS')
    ax.scatter(0, 0, 0, marker='d', label='RIS')

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')

    ax.legend()

    # Build environment
    env = RIS2DEnv(bs_position=bs_pos, ue_position=ue_pos, sides=200 * np.ones(3))
    # TODO: this sides is not being used, I am just putting a random value to ensure that the tests pass.

    ##############################
    # Generate DFT codebook of configurations
    ##############################

    # Define fundamental frequency
    fundamental_freq = np.exp(-1j * 2 * np.pi / env.ris.num_els)

    # Compute DFT matrix
    J, K = np.meshgrid(np.arange(env.ris.num_els), np.arange(env.ris.num_els))
    DFT = np.power(fundamental_freq, J * K)

    # Compute normalized DFT matrix
    DFT_norm = DFT / np.sqrt(env.ris.num_els)

    # Plot DFT codebook matrix
    fig, ax = plt.subplots()

    im = ax.imshow(DFT_norm.real)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    ax.set_title('DFT configuration matrix: $' + str(env.ris.num_els) + r'\times' + str(env.ris.num_els) + '$')

    plt.tight_layout()

    # Get the channels:
    #   ur - user-ris
    #   rb - ris-bs
    h_ur, g_rb, _ = env.return_separate_ris_channel()

    # Squeeze out
    g_rb = np.squeeze(g_rb)

    ##############################
    # Codebook-based
    ##############################

    # Codebook selection
    if cb_type == 'flexible':
        codebook = DFT_norm.copy()

        # Generate noise realizations
        noise_ = (np.random.randn(num_users, env.ris.num_els) + 1j * np.random.randn(num_users, env.ris.num_els)) / np.sqrt(2)
    else:
        index_selection = np.arange(0, 100, 3)
        codebook = DFT_norm[index_selection, :]
        num_configs = len(index_selection)

        # Generate noise realizations
        noise_ = (np.random.randn(num_users, num_configs) + 1j * np.random.randn(num_users, num_configs)) / np.sqrt(2)

    # Compute the equivalent channel
    h_eq_cb = (g_rb[np.newaxis, np.newaxis, :] * codebook[np.newaxis, :, :] * h_ur[:, np.newaxis, :]).sum(axis=-1)

    # Generate noise
    var = noise_power / num_pilots
    bsw_noise_ = np.sqrt(var) * noise_

    # Compute the measured SNR of each user when using CB scheme
    sig_pow_cb = tx_power * np.abs(h_eq_cb) ** 2
    sig_pow_noisy_cb = np.abs(np.sqrt(tx_power) * h_eq_cb + bsw_noise_) ** 2

    snr_cb = sig_pow_cb / noise_power
    snr_cb_noisy = sig_pow_noisy_cb / noise_power

    snr_cb_db = 10 * np.log10(sig_pow_cb / noise_power)
    snr_cb_noisy_db = 10 * np.log10(snr_cb_noisy)

    # Go through all users
    rate_cb_noisy = np.zeros(num_users)
    n_configurations_flex = np.zeros(num_users)

    for uu in range(num_users):

        # Get the first case in which this is true
        mask = snr_cb_noisy[uu] >= minimum_snr

        if sum(mask) == 0:
            continue

        # Get the index of the first occurrence
        index = np.argmax(mask)

        # Store results
        rate_cb_noisy[uu] = np.log2(1 + minimum_snr)
        n_configurations_flex[uu] = index + 1

    # Pre-log term
    for setup in setups:
        if setup == 'ob-cc':
            tau_setup = T
        elif setup == 'ib-no':
            tau_setup = 2 * T
        else:
            tau_setup = 3 * T

        if cb_type == 'fixed':
            tau_alg = num_configs * T
            prelog_term = 1 - (tau_setup + tau_setup + tau_alg)/total_tau
            prelog_term[prelog_term < 0] = 0

            rate_cb_bsw = prelog_term[np.newaxis].T * np.repeat(rate_cb_noisy[np.newaxis], len(total_tau), axis=0)

        else:
            # Max function is used to remove the negative values
            tau_alg = np.max(np.vstack((np.zeros_like(n_configurations_flex), (2 * n_configurations_flex - 1) * T)), axis=0)
            prelog_term = 1 - (tau_setup + tau_setup + tau_alg[np.newaxis])/np.repeat(total_tau[np.newaxis].T, num_users, axis=1)
            prelog_term[prelog_term < 0] = 0

            rate_cb_bsw = prelog_term * np.repeat(rate_cb_noisy[np.newaxis], len(total_tau), axis=0)

        ##################################################
        # Save data
        ##################################################
        np.savez(prefix + '_' + cb_type + '_' + setup + str('.npz'),
                 snr_true=snr_cb,
                 snr_esti=snr_cb_noisy,
                 rate=rate_cb_bsw)