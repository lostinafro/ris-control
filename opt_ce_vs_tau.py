import numpy as np

import matplotlib.pyplot as plt

import scenario.common as cmn
from environment import RisProtocolEnv, command_parser, NOISE_POWER_dBm, T, TAU, TX_POW_dBm, NUM_PILOTS

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

# Total tau
total_tau = TAU

# Define CHEST algorithmic cost
chest_time_cost = 5

# Parameter for saving datas
prefix = 'data/opt_ce_vs_tau'

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

    # Generate noise realizations
    noise_ = (np.random.randn(num_users, env.ris.num_els) + 1j * np.random.randn(num_users, env.ris.num_els)) / np.sqrt(2)

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
    # OPT-CE
    ##############################

    # Generate estimation noise
    est_var = noise_power / env.ris.num_els / tx_power / NUM_PILOTS
    est_noise_ = np.sqrt(est_var) * noise_

    # Get estimated channel coefficients for the equivalent channel
    z_eq = h_ur * g_rb[np.newaxis, :]
    z_hat = z_eq + est_noise_

    # Get estimated/best configuration (the one that maximizes the SNR)
    Phi_true = np.exp(-1j * np.angle(z_eq))
    Phi_hat = np.exp(-1j * np.angle(z_hat))

    # Compute equivalent channel
    h_eq_chest = (Phi_true * z_eq).sum(axis=-1)
    h_eq_chest_hat = (Phi_hat * z_hat).sum(axis=-1)

    # Compute the SNR of each user when using OPT-CE
    snr_oc = tx_power / noise_power * np.abs(h_eq_chest) ** 2
    snr_oc_hat = tx_power / noise_power * np.abs(h_eq_chest_hat) ** 2

    snr_oc_db = 10 * np.log10(snr_oc)
    snr_oc_hat_db = 10 * np.log10(snr_oc_hat)

    # Compute rate
    se_oc = np.log2(1 + snr_oc)
    se_oc_hat = np.log2(1 + snr_oc_hat)
    se_oc_hat[se_oc_hat > se_oc] = 0

    # Pre-log term
    tau_alg = (env.ris.num_els + chest_time_cost) * T

    for setup in setups:
        if setup == 'ob-cc':
            tau_setup = T
        elif setup == 'ib-no':
            tau_setup = 2 * T
        else:
            tau_setup = 3 * T

        prelog_term = 1 - (tau_setup + tau_setup + tau_alg)/total_tau
        prelog_term[prelog_term < 0] = 0

        rate_opt_ce = prelog_term[np.newaxis].T * np.repeat(se_oc[np.newaxis], len(total_tau), axis=0)
        rate_opt_ce_hat = prelog_term[np.newaxis].T * np.repeat(se_oc_hat[np.newaxis], len(total_tau), axis=0)

        ##################################################
        # Save data
        ##################################################
        np.savez(prefix + '_' + setup + str('.npz'),
                 snr_true=snr_oc,
                 snr_esti=snr_oc_hat,
                 rate=rate_opt_ce,
                 rate_esti=rate_opt_ce_hat)