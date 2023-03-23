import numpy as np

import matplotlib.pyplot as plt

import scenario.common as cmn
from environment import RisProtocolEnv, command_parser, NOISE_POWER_dBm, TX_POW_dBm, T, OUTPUT_DIR, TAU, BANDWIDTH, NUM_PILOTS
import os

from matplotlib import rc


##################################################
# Parameters
##################################################

# Seed random generator
np.random.seed(42)

# Define transmit power [mW]
tx_power = cmn.dbm2watt(TX_POW_dBm)

# Get noise power
noise_power = cmn.dbm2watt(NOISE_POWER_dBm)

# For grid mesh
num_users = int(1e4)

# Setup option
setups = ['ob-cc', 'ib-wf']

# Define range of minimum SNR
minimum_snr_range_dB = np.linspace(-6, 30, 50)
minimum_snr_range = cmn.db2lin(minimum_snr_range_dB)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # The following parser is used to impose some data without the need of changing the script (run with -h flag for
    # help) Render bool needs to be True to save the data If no arguments are given the standard value are loaded (
    # see environment) datasavedir should be used to save numpy arrays
    render, side, name, datasavedir = command_parser()

    # Build environment
    env = RisProtocolEnv(num_users=num_users, side=side)

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

    # Get the channels:
    #   ur - user-ris
    #   rb - ris-bs
    h_ur, g_rb, _ = env.return_separate_ris_channel()

    # Squeeze out
    g_rb = np.squeeze(g_rb)

    ##############################
    # Codebook-based
    ##############################
    codebook_flexi = DFT_norm.copy()

    index_selection = np.arange(0, 100, 2)
    codebook_fixed = DFT_norm.copy()[index_selection, :]
    num_configs = len(index_selection)

    # Generate noise realizations
    noise_ = (np.random.randn(num_users, env.ris.num_els) + 1j * np.random.randn(num_users, env.ris.num_els)) / np.sqrt(2)

    # Compute the equivalent channel
    h_eq_cb_flexi = (g_rb.conj()[np.newaxis, np.newaxis, :] * codebook_flexi[np.newaxis, :, :] * h_ur[:, np.newaxis, :]).sum(axis=-1)
    # Compute the equivalent channel
    h_eq_cb_fixed = (g_rb.conj()[np.newaxis, np.newaxis, :] * codebook_fixed[np.newaxis, :, :] * h_ur[:, np.newaxis, :]).sum(axis=-1)

    # Generate some noise
    var = noise_power / NUM_PILOTS
    bsw_noise_ = np.sqrt(var) * noise_

    # Compute the SNR of each user when using CB scheme
    sig_pow_cb_flexi = tx_power * np.abs(h_eq_cb_flexi) ** 2
    sig_pow_noisy_cb_flexi = np.abs(np.sqrt(tx_power) * h_eq_cb_flexi + bsw_noise_) ** 2
    sig_pow_cb_fixed = tx_power * np.abs(h_eq_cb_fixed) ** 2
    sig_pow_noisy_cb_fixed = np.abs(np.sqrt(tx_power) * h_eq_cb_fixed + bsw_noise_[:, index_selection]) ** 2

    snr_cb_flexi = sig_pow_cb_flexi / noise_power
    snr_cb_hat_flexi = sig_pow_noisy_cb_flexi / noise_power
    snr_cb_fixed = sig_pow_cb_fixed / noise_power
    snr_cb_hat_fixed = sig_pow_noisy_cb_fixed / noise_power

    # Prepare to store flexible rate
    se_flex = np.zeros((minimum_snr_range.size, num_users))
    se_fixed = np.zeros((minimum_snr_range.size, num_users))
    n_configurations_flex = np.zeros((minimum_snr_range.size, num_users))

    # Go through all minimum SNR values
    for ms, minimum_snr in enumerate(minimum_snr_range):

        # Go through all users (FLEXI)
        for uu in range(num_users):

            # Get the first case in which this is true
            mask = snr_cb_hat_flexi[uu] >= minimum_snr

            if np.sum(mask) == 0:
                n_configurations_flex[ms, uu] = -1
                continue

            # Get the index of the first occurrence
            index = np.argmax(mask)

            # Store results
            se_flex[ms, uu] = np.log2(1 + minimum_snr) if snr_cb_flexi[uu, index] > minimum_snr else 0.
            n_configurations_flex[ms, uu] = index + 1

        # Go through all users (FIXED)
        for uu in range(num_users):

            # Check if there is an outage
            mask = snr_cb_hat_fixed[uu] >= minimum_snr

            if np.sum(mask) == 0:
                continue

            # Get the index of the highest snr satisfying the constraint
            index = np.argmax(snr_cb_hat_fixed[uu] * mask)

            # Store results
            se_fixed[ms, uu] = np.log2(1 + minimum_snr) if snr_cb_fixed[uu, index] > minimum_snr else 0.


    ## Varying the frame duration
    tau_plot = [7.5, 15, 30]

    opt_kpi_flexi = np.zeros((len(TAU), len(setups)))
    opt_kpi_fixed = np.zeros((len(TAU), len(setups)))


    for tt, total_tau in enumerate(TAU):
        if total_tau in tau_plot:
            # Prepare figure
            fig, axes = plt.subplots(nrows=len(setups))

        for ss, setup in enumerate(setups):
            # File name
            datafilename = 'data/opt_ce_vs_tau_' + setup + '.npz'

            # Load data
            rate = np.squeeze(np.load(datafilename)['rate'][TAU == total_tau]) * BANDWIDTH / 1e6

            if total_tau in tau_plot:
                axes[ss].plot(minimum_snr_range_dB, np.mean(rate) * np.ones_like(minimum_snr_range), linewidth=1.5, color='black', label='OPT-CE')
                axes[ss].set_title(setup.upper())

            # Define tau_setup
            if setup == 'ob-cc':
                tau_setup = T
            elif setup == 'ib-no':
                tau_setup = 2 * T
            else:
                tau_setup = 3 * T

            # Flexible structure
            # Pre-log term
            tau_alg = (2 * n_configurations_flex - 1) * T
            prelog_term = 1 - (tau_setup + tau_setup + tau_alg) / total_tau
            prelog_term[(prelog_term < 0) | (tau_alg < 0)] = 0

            rate_cb_bsw = prelog_term * se_flex
            avg_rate_cb_bsw_flexi = np.mean(rate_cb_bsw, axis=-1)
            if total_tau in tau_plot:
                axes[ss].plot(minimum_snr_range_dB, avg_rate_cb_bsw_flexi * BANDWIDTH / 1e6, linewidth=1.5, linestyle=':', label='CB-BSW: Flexible', color='red')

            opt_kpi_flexi[tt, ss] = minimum_snr_range_dB[np.argmax(avg_rate_cb_bsw_flexi)]

            # Fixed structure
            # Pre-log term
            tau_alg = num_configs * T
            prelog_term = np.maximum(1 - (tau_setup + tau_setup + tau_alg) / total_tau, 0)

            rate_cb_bsw = prelog_term * se_fixed
            avg_rate_cb_bsw_fixed = np.mean(rate_cb_bsw, axis=-1)
            if total_tau in tau_plot:
                axes[ss].plot(minimum_snr_range_dB, avg_rate_cb_bsw_fixed * BANDWIDTH / 1e6, linewidth=1.5, linestyle='--', label='CB-BSW: Fixed', color='blue')

            opt_kpi_fixed[tt, ss] = minimum_snr_range_dB[np.argmax(avg_rate_cb_bsw_fixed)]

            if total_tau in tau_plot:
                print(f'--- tau = {total_tau:.3f} ---')
                print(f'--- setup = {setup} ---')
                print('OPT-CE =', np.mean(rate))
                print('CB-BSW fixed =', np.max(avg_rate_cb_bsw_fixed))
                print(f'opt KPI: {minimum_snr_range_dB[np.argmax(avg_rate_cb_bsw_fixed)]:.3f}')
                print('CB-BSW flexi =', np.max(avg_rate_cb_bsw_flexi))
                print(f'opt KPI: {minimum_snr_range_dB[np.argmax(avg_rate_cb_bsw_flexi)]:.3f}')
                print('\n')

        # Plot for specific values
        if total_tau in tau_plot:
            cmn.printplot(fig, axes, render=render, filename=f'rate_vs_gamma0_{total_tau:.2f}', dirname=OUTPUT_DIR,
                          labels=[r'$\gamma_0$ [dB]', r'$R$ [Mbit/s]', r'$R$ [Mbit/s]'], orientation='vertical')

    # Save optimal threshold values
    opt_kpi_fixed = np.mean(opt_kpi_fixed, axis=-1)
    opt_kpi_flexi = np.mean(opt_kpi_flexi, axis=-1)


    np.savez(os.path.join(datasavedir, 'cb_bsw_opt_kpi.npz'),
             fixed=opt_kpi_fixed,
             flexi=opt_kpi_flexi)