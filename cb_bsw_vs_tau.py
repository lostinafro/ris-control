import numpy as np

import matplotlib.pyplot as plt

import scenario.common as cmn
from environment import RisProtocolEnv, command_parser, NOISE_POWER_dBm, T, TAU, TX_POW_dBm

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
minimum_snr_dB = 10
minimum_snr = cmn.db2lin(minimum_snr_dB)

# Total tau
total_tau = TAU

# Parameter for saving datas
prefix = 'data/cb_bsw_vs_tau'

# CB type
cb_types = ['fixed', 'flexi']

# Setup option
setups = ['ob-cc', 'ib-no', 'ib-wf']

# For grid mesh
num_users = int(1e3)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # The following parser is used to impose some data without the need of changing the script (run with -h flag for
    # help) Render bool needs to be True to save the data If no arguments are given the standard value are loaded (
    # see environment) datasavedir should be used to save numpy arrays
    render, side, name, datasavedir = command_parser()
    prefix = prefix + name

    # Build environment
    env = RisProtocolEnv(num_users=num_users, side=side)
    # env.plot_scenario()

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

    for cb_type in cb_types:
        try:
            minimum_snr_dB = np.load('data/cb_bsw_opt_kpi.npz')[cb_type]
        except FileNotFoundError:
            minimum_snr_dB = 10
        minimum_snr = cmn.db2lin(minimum_snr_dB)


        # Codebook selection
        if cb_type == 'flexi':
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
        snr_cb_hat = sig_pow_noisy_cb / noise_power

        snr_cb_db = 10 * np.log10(sig_pow_cb / noise_power)
        snr_cb_hat_db = 10 * np.log10(snr_cb_hat)

        # Go through all users
        se_cb = np.zeros((len(total_tau), num_users))
        n_configurations_flex = np.zeros((len(total_tau), num_users))

        for tt, tau in enumerate(total_tau):
            for uu in range(num_users):

                # Get the first case in which this is true
                mask = snr_cb_hat[uu] >= minimum_snr[tt]

                if np.sum(mask) == 0:    # No configuration satisfies the KPI
                    n_configurations_flex[tt, uu] = -1
                else:       # At least one conf satisfies the KPI
                    # Get the index of the first occurrence
                    index = np.argmax(mask)

                    # Store configuration number
                    n_configurations_flex[tt, uu] = index + 1
                    se_cb[tt, uu] = np.log2(1 + minimum_snr[tt])

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
                prelog_term = 1 - (tau_setup + tau_setup + tau_alg) / total_tau
                prelog_term[prelog_term < 0] = 0

                rate_cb_bsw = prelog_term[np.newaxis].T * se_cb

            else:
                # Max function is used to remove the negative values
                tau_alg = np.repeat((2 * n_configurations_flex[np.newaxis] - 1), len(total_tau), axis=0) * T
                prelog_term = 1 - (tau_setup + tau_setup + tau_alg) / np.repeat(total_tau[np.newaxis].T, num_users, axis=1)
                prelog_term[(prelog_term < 0) | (tau_alg < 0)] = 0

                rate_cb_bsw = prelog_term * se_cb

            ##################################################
            # Save data
            ##################################################
            np.savez(prefix + '_' + cb_type + '_' + setup + str('.npz'),
                     snr_true=snr_cb,
                     snr_esti=snr_cb_hat,
                     rate=rate_cb_bsw)