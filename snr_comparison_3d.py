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

# Define transmit power [mW]
tx_power = 100

# Get noise power
noise_power = cmn.dbm2watt(NOISE_POWER_dBm)

# Define number of pilots
num_pilots = 1

# Parameter for saving datas
prefix = '3D_'

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
    x = cube_length * np.random.rand(num_users, 1) - cube_length / 2
    y = cube_length / 2 * np.random.rand(num_users, 1)
    z = cube_length * np.random.rand(num_users, 1) - cube_length / 2

    # Get position of the users and position of the BS
    ue_pos = np.hstack((x, y, z))
    bs_pos = np.array([[20, 0, 0]])

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

    # Prepare figure
    fig, ax = plt.subplots()

    cmap = plt.get_cmap("tab10")

    ##############################
    # OPT-CE
    ##############################

    # Generate estimation noise
    est_var = noise_power / env.ris.num_els / tx_power / num_pilots / 2
    est_noise_ = np.sqrt(est_var) * noise_

    # Get estimated channel coefficients for the equivalent channel
    z_eq = h_ur.conj() * g_rb[np.newaxis, :]
    z_hat = z_eq + est_noise_

    # Get estimated/best configuration (the one that maximizes the SNR)
    Phi_hat = np.exp(-1j*np.angle(z_hat))

    # Compute equivalent channel
    h_eq_chest = ((g_rb.conj()[np.newaxis, :] * Phi_hat) * h_ur).sum(axis=-1)
    h_eq_chest_hat = (Phi_hat * z_hat).sum(axis=-1)

    # Compute the SNR of each user when using OPT-CE
    sig_pow_oc = tx_power * np.abs(h_eq_chest) ** 2
    sig_pow_oc_hat = tx_power * np.abs(h_eq_chest_hat) ** 2

    snr_oc_db = 10 * np.log10(sig_pow_oc / noise_power)
    snr_oc_hat_db = 10 * np.log10(sig_pow_oc_hat / noise_power)

    # Get CDF
    x_cdf_oc_db, y_cdf_oc_db = ecdf(snr_oc_db)
    x_cdf_oc_hat_db, y_cdf_oc_hat_db = ecdf(snr_oc_hat_db)

    ax.plot(x_cdf_oc_db, y_cdf_oc_db, linewidth=1.5, color='black', label=r'OPT-CE: true')
    ax.plot(x_cdf_oc_hat_db, y_cdf_oc_hat_db, linewidth=1.5, linestyle='--', color='black', label=r'OPT-CE: estimated')

    ##############################
    # Codebook-based
    ##############################

    # Codebook selection
    codebook = DFT_norm.copy()

    # Compute the equivalent channel
    h_eq_cb = (g_rb.conj()[np.newaxis, np.newaxis, :] * codebook[np.newaxis, :, :] * h_ur[:, np.newaxis, :]).sum(axis=-1)

    # Generate some noise
    var = noise_power / num_pilots / 2
    bsw_noise_ = np.sqrt(var) * noise_

    # Compute the SNR of each user when using CB scheme
    sig_pow_cb = tx_power * np.abs(h_eq_cb) ** 2
    sig_pow_noisy_cb = tx_power * np.abs(h_eq_cb + bsw_noise_) ** 2

    snr_cb_db = 10 * np.log10(sig_pow_cb / noise_power)
    snr_cb_noisy_db = 10 * np.log10(sig_pow_noisy_cb / noise_power)

    # Get CDF
    x_cdf_cb_db, y_cdf_cb_db = ecdf(snr_cb_db)
    x_cdf_cb_noisy_db, y_cdf_cb_noisy_db = ecdf(snr_cb_noisy_db)

    ax.plot(x_cdf_cb_db, y_cdf_cb_db, linewidth=1.5, color='tab:blue', label='CB-BSW: true, $N=' + str(env.ris.num_els) + '$')
    ax.plot(x_cdf_cb_noisy_db, y_cdf_cb_noisy_db, linewidth=1.5, color='tab:blue', linestyle='--',
            label=r'CB-BSW: noisy, $C_{\mathrm{CB}}=' + str(env.ris.num_els) + '$')

    ax.set_xlabel('SNR [dB]')
    ax.set_ylabel('CDF')

    ax.legend()

    plt.tight_layout()

    plt.show()

    # # This is a wrap function to show or save plots conveniently
    # # If render == True this will save the data in ris-protocol/plots/{dateoftoday}/
    # cmn.printplot(fig, axes, render, filename=f'{prefix}' + 'oc_vs_cb', dirname=OUTPUT_DIR,
    #               labels=['SNR over noise floor [dB]', 'norm. SNR over noise floor [dB]', 'ECDF'],
    #               orientation='horizontal')
