import numpy as np

import matplotlib.pyplot as plt

import scenario.common as cmn
from environment import RIS2DEnv, command_parser, NOISE_POWER_dBm, TX_POW_dBm, N_TTIs, T, OUTPUT_DIR

from matplotlib import rc
from mpl_toolkits.axes_grid1 import make_axes_locatable

rc('font', **{'family': 'sans serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

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

# Total tau
total_tau = T * N_TTIs

# For grid mesh
num_users = int(1e3)

# Setup option
setups = ['ob-cc', 'ib-wf']

# Define range of minimum SNR
minimum_snr_range_dB = np.linspace(-6, 30, 100)
minimum_snr_range = cmn.db2lin(minimum_snr_range_dB)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # The following parser is used to impose some data without the need of changing the script (run with -h flag for
    # help) Render bool needs to be True to save the data If no arguments are given the standard value are loaded (
    # see environment) datasavedir should be used to save numpy arrays
    render, side_x, h, name, datasavedir = command_parser()

    # Define length of the cube
    cube_length = side_x

    # Drop some users: the RIS is assumed to be in the middle of the bottom face of the cube.
    x = cube_length * np.random.rand(num_users, 1) - cube_length
    y = cube_length / 2 * np.random.rand(num_users, 1)
    z = cube_length * np.random.rand(num_users, 1) - cube_length / 2

    # Get position of the users and position of the BS
    ue_pos = np.hstack((x, y, z))
    bs_pos = np.array([[20, 5, 0]])

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
    codebook = DFT_norm.copy()

    # Generate noise realizations
    noise_ = (np.random.randn(num_users, env.ris.num_els) + 1j * np.random.randn(num_users, env.ris.num_els)) / np.sqrt(2)

    # Compute the equivalent channel
    h_eq_cb = (g_rb.conj()[np.newaxis, np.newaxis, :] * codebook[np.newaxis, :, :] * h_ur[:, np.newaxis, :]).sum(axis=-1)

    # Generate some noise
    var = noise_power / num_pilots
    bsw_noise_ = np.sqrt(var) * noise_

    # Compute the SNR of each user when using CB scheme
    sig_pow_cb = tx_power * np.abs(h_eq_cb) ** 2
    sig_pow_noisy_cb = np.abs(np.sqrt(tx_power) * h_eq_cb + bsw_noise_) ** 2

    snr_cb = sig_pow_cb / noise_power
    snr_cb_noisy = sig_pow_noisy_cb / noise_power

    snr_cb_db = 10 * np.log10(sig_pow_cb / noise_power)
    snr_cb_noisy_db = 10 * np.log10(snr_cb_noisy)

    # Compute rates
    rate_cb = np.log2(1 + snr_cb)
    rate_cb_noisy = np.log2(1 + snr_cb_noisy)

    # Prepare to store flexible rate
    rate_flex = np.zeros((minimum_snr_range.size, num_users))
    n_configurations_flex = np.zeros((minimum_snr_range.size, num_users))

    # Go through all minimum SNR values
    for ms, minimum_snr in enumerate(minimum_snr_range):

        # Go through all users
        for uu in range(num_users):

            # Get the first case in which this is true
            mask = snr_cb_noisy[uu] >= minimum_snr

            if np.sum(mask) == 0:
                continue

            # Get the index of the first occurrence
            index = np.argmax(mask)

            # Store results
            rate_flex[ms, uu] = np.log2(1 + minimum_snr)
            n_configurations_flex[ms, uu] = index + 1

    # Prepare figure
    fig, axes = plt.subplots(nrows=len(setups))

    for ss, setup in enumerate(setups):

        print('----- setup: ' + str(setup) + ' -----')

        # File name
        datafilename = 'data/opt_ce_' + setup + '.npz'

        # Load data
        rate = np.load(datafilename)['rate']

        axes[ss].plot(minimum_snr_range_dB, np.mean(rate) * np.ones_like(minimum_snr_range), linewidth=1.5, color='black', label='OPT-CE')
        axes[ss].set_title(setup.upper())

        # Define tau_setup
        if setup == 'ob-cc':
            tau_setup = T
        elif setup == 'ib-no':
            tau_setup = 2 * T
        else:
            tau_setup = 3 * T

        # Pre-log term
        tau_alg = np.max(np.vstack((np.zeros_like(n_configurations_flex), (2 * n_configurations_flex - 1) * T)), axis=0)
        prelog_term = 1 - (tau_setup + tau_setup + tau_alg) / total_tau
        prelog_term[prelog_term < 0] = 0

        rate_cb_bsw = prelog_term * rate_flex
        avg_rate_cb_bsw_flexi = np.mean(rate_cb_bsw, axis=-1)

        axes[ss].plot(minimum_snr_range_dB, avg_rate_cb_bsw_flexi, linewidth=1.5, linestyle=':', label='CB-BSW: Flexible')

        # Pre-log term
        index_selection = np.arange(0, env.ris.num_els, 1)
        num_configs = len(index_selection)
        tau_alg = num_configs * T
        prelog_term = 1 - (tau_setup + tau_setup + tau_alg) / total_tau

        rate_cb_bsw = prelog_term * rate_flex
        avg_rate_cb_bsw_fixed = np.mean(rate_cb_bsw, axis=-1)

        axes[ss].plot(minimum_snr_range_dB, avg_rate_cb_bsw_fixed, linewidth=1.5, linestyle='--', label='CB-BSW: Fixed')


        print('OPT-CE =', np.mean(rate))
        print('CB-BSW fixed =', np.max(avg_rate_cb_bsw_fixed))
        print(minimum_snr_range_dB[np.argmax(avg_rate_cb_bsw_fixed)])
        print('CB-BSW flexi =', np.max(avg_rate_cb_bsw_flexi))
        print(minimum_snr_range_dB[np.argmax(avg_rate_cb_bsw_flexi)])

    cmn.printplot(fig, axes, render=True, filename='rate_vs_gamma0', dirname=OUTPUT_DIR,
                  labels=[r'$\gamma_0$ [dB]', r'$R$ [bit/Hz/s]', r'$R$ [bit/Hz/s]'], orientation='vertical')
