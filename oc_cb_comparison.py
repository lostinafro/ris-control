try:
    import cupy as np
except ImportError:
    import numpy as np

import matplotlib.pyplot as plt

from os import path

import scenario.common as cmn
from environment import RIS2DEnv, command_parser, ecdf

NOISE_POWER_dbm = -94               # [dBm]
NOISE_POWER = 10**(NOISE_POWER_dbm / 10)

# Parameter for saving datas
prefix = 'random_users_'

# For grid mesh
num_users = int(1e4)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # The following parser is used to impose some data without the need of changing the script (run with -h flag for help)
    # Render bool needs to be True to save the data
    # If no arguments are given the standard value are loaded (see environment)
    render, side_x, h, name, dirname = command_parser()
    side_y = side_x
    prefix = prefix + name

    if render:
        # Load previous data if they exist
        filename = path.join(dirname, f'{prefix}_hris.npz')
    else:
        filename = ''

    # Generate users
    x = side_x * np.random.rand(num_users, 1)
    y = side_y * np.random.rand(num_users, 1)

    ue_pos = np.hstack((x, y, np.zeros((num_users, 1))))

    # Build environment
    env = RIS2DEnv(ue_position=ue_pos, sides=np.array([side_x, side_y, h]))
    num_conf = env.ris.num_std_configs

    # Pre-allocation of variables
    h_ur = np.zeros((num_users, env.ris.num_els), dtype=complex)
    g_rb = np.zeros((env.ris.num_els), dtype=complex)
    Phi = np.zeros((num_conf, env.ris.num_els, env.ris.num_els), dtype=complex)

    # Looping through the configurations (progressbar is only a nice progressbar)
    for index in cmn.std_progressbar(range(num_conf)):

        # Load the appropriate config
        actual_conf = env.set_std_conf_2D(index)

        # Compute array factor
        env.compute_array_factor()

        # Compute the channel for each user
        if index == 0: 
            h_ur, g_rb, Phi[index] = env.return_separate_ris_channel()
        else:
            _, _, Phi[index] = env.return_separate_ris_channel()        

    # Squeeze out
    g_rb = np.squeeze(g_rb)

    # Compute the equivalent channel 
    h_eq_cb = np.matmul(np.matmul(g_rb.conj().T[np.newaxis, :], Phi)[:, np.newaxis, :, :], h_ur[np.newaxis, :, :, np.newaxis])
    h_eq_cb = h_eq_cb.squeeze()

    # Compute the SNR of each user when using beam sweeping
    sig_pow_cb = np.abs(h_eq_cb) ** 2
    snr_cb_db = 10 * np.log10(sig_pow_cb) - NOISE_POWER_dbm

    # Generate estimation noise
    est_var = (NOISE_POWER/2) / env.ris.num_els
    est_noise_ = np.sqrt(est_var) * (np.random.randn(num_users, env.ris.num_els) + 1j * np.random.randn(num_users, env.ris.num_els))

    # Get estimated channel coefficients
    z_eq = h_ur.conj() * g_rb[np.newaxis, :]
    z_hat = z_eq + est_noise_

    # Get estimated Phi
    Phi_hat = np.exp(np.angle(-z_hat))

    # Compute equivalent channel
    h_eq_chest = ((g_rb.conj()[np.newaxis] * Phi_hat) * h_ur).sum(axis=-1)

    # Compute the SNR of each user when using CHEST
    sig_pow_chest = np.abs(h_eq_chest)**2
    snr_chest_db = 10 * np.log10(sig_pow_chest) - NOISE_POWER_dbm

    ##############################
    # Plot
    ##############################
    fig, axes = plt.subplots(ncols=2, sharey='all')

    # Get CDF for CB
    x_cdf_cb_db, y_cdf_cb_db = ecdf(snr_cb_db)
    axes[0].plot(x_cdf_cb_db, y_cdf_cb_db, color='black', label='CB')

    # Get CDF for CHEST
    x_cdf_chest_db, y_cdf_chest_db = ecdf(snr_chest_db)
    axes[0].plot(x_cdf_chest_db, y_cdf_chest_db, label=r'CHEST')

    axes[0].set_xlabel('SNR over noise floor [dB]')
    axes[0].set_ylabel('ECDF')     #x_cdf_cb_db, y_cdf_cb_db = ecdf(snr_cb_db/Phi.shape[0])

    axes[0].legend()

    axes[1].plot(x_cdf_cb_db / Phi.shape[0], y_cdf_cb_db, color='black')
    axes[1].plot(x_cdf_chest_db / env.ris.num_els, y_cdf_chest_db)

    axes[1].set_xlabel('norm. SNR over noise floor [dB]')

    plt.show()


    # saving data
    if render:
        np.savez(filename,  h_ris=h_ris)
    print('\t...DONE')

