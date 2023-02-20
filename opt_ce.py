try:
    import cupy as np
except ImportError:
    import numpy as np

import matplotlib.pyplot as plt

import scenario.common as cmn
from environment import RIS2DEnv, command_parser, ecdf, OUTPUT_DIR, NOISE_POWER_dBm

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
tx_power = 100

# Get noise power
noise_power = cmn.dbm2watt(NOISE_POWER_dBm)

# Define number of pilots
num_pilots = 1

# Period
T = 1/14

# Number of TTIs
n_ttis = 200

# Total tau
total_tau = T * n_ttis

# Define CHEST algorithmic cost
chest_time_cost = 5

# Parameter for saving datas
prefix = 'data/opt_ce'

# Setup option
setup = 'ob-cc'
#setup = 'ib-no'
#setup = 'ib-wf'

if setup == 'ob-cc':
    tau_setup = T
elif setup == 'ib-no':
    tau_setup = 2 * T
else:
    tau_setup = 3 * T

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

    # Generate noise realizations
    noise_ = (np.random.randn(num_users, env.ris.num_els) + 1j * np.random.randn(num_users, env.ris.num_els))

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
    est_var = noise_power / env.ris.num_els / tx_power / num_pilots / 2
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
    sig_pow_oc = tx_power * np.abs(h_eq_chest) ** 2
    sig_pow_oc_hat = tx_power * np.abs(h_eq_chest_hat) ** 2

    snr_oc = sig_pow_oc / noise_power
    snr_oc_hat = sig_pow_oc_hat / noise_power

    snr_oc_db = 10 * np.log10(snr_oc)
    snr_oc_hat_db = 10 * np.log10(snr_oc_hat)

    # Compute rate
    rate_oc = np.log2(1 + snr_oc)
    rate_oc_hat = np.log2(1 + snr_oc_hat)

    # Pre-log term
    tau_alg = (env.ris.num_els + chest_time_cost) * T
    prelog_term = 1 - (tau_setup + tau_setup + tau_alg)/total_tau

    rate_opt_ce = prelog_term * rate_oc_hat

    ##################################################
    # Save data
    ##################################################
    np.savez(prefix + '_' + setup + str('.npz'),
             snr=snr_oc_hat,
             rate=rate_opt_ce
             )

    ##################################################
    # Plot
    ##################################################
    fig, axes = plt.subplots(nrows=2)

    # Get CDF
    x_cdf_oc_db, y_cdf_oc_db = ecdf(snr_oc_db)
    x_cdf_oc_hat_db, y_cdf_oc_hat_db = ecdf(snr_oc_hat_db)

    axes[0].plot(x_cdf_oc_db, y_cdf_oc_db, linewidth=1.5, color='black', label=r'OPT-CE: true')
    axes[0].plot(x_cdf_oc_hat_db, y_cdf_oc_hat_db, linewidth=1.5, linestyle='--', color='black',
                 label=r'OPT-CE: estimated')

    axes[0].set_xlabel('SNR [dB]')
    axes[0].set_ylabel('CDF')

    axes[0].legend()

    # Get CDF
    x_cdf_oc_db, y_cdf_oc_db = ecdf(rate_oc)
    x_cdf_oc_hat_db, y_cdf_oc_hat_db = ecdf(rate_oc_hat)

    axes[1].plot(x_cdf_oc_db, y_cdf_oc_db, linewidth=1.5, color='black', label=r'OPT-CE: true')
    axes[1].plot(x_cdf_oc_hat_db, y_cdf_oc_hat_db, linewidth=1.5, linestyle='--', color='black',
                 label=r'OPT-CE: estimated')

    axes[1].set_xlabel('Rate [bit/Hz/s]')
    axes[1].set_ylabel('CDF')

    plt.tight_layout()


    #plt.show()