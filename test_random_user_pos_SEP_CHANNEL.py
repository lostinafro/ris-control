try:
    import cupy as np
except ImportError:
    import numpy as np

import matplotlib.pyplot as plt

from os import path

import scenario.common as cmn
from environment import RIS2DEnv, command_parser, ecdf

NOISE_POWER = -94               # [dBm]

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

    g_rb = np.squeeze(g_rb)

    
    # Compute the equivalent channel 
    h_eq = np.matmul(np.matmul(g_rb.conj().T[np.newaxis, :], Phi)[:, np.newaxis, :, :], h_ur[np.newaxis, :, :, np.newaxis])
    h_eq = h_eq.squeeze()

    # Compute the SNR of each user when using beam sweeping
    snr_bsw_db = 10 * np.log10(np.abs(h_eq)**2) - NOISE_POWER

    # Get angles 
    g_rb_angles = np.angle(g_rb)

    # Estimation qualities
    est_qualities_ = np.array([0, 0.01, 0.25, 0.5, 0.75, 1])

    # Generate estimation noise
    est_noise_ = (1/np.sqrt(2)) * (np.random.randn(num_users, env.ris.num_els) + 1j * np.random.randn(num_users, env.ris.num_els))

    # Prepare to save the equivalent channel
    h_eq_chest = np.zeros((len(est_qualities_), num_users), dtype=complex)
    h_eq_chest_hat = np.zeros((len(est_qualities_), num_users), dtype=complex)

    # Go through all the qualities 
    for qq, quality in enumerate(est_qualities_):

        # Store estimates
        hat_h_ur = np.sqrt((1 - quality**2)) * h_ur +  quality * est_noise_

        # Get angles
        hat_h_ur_angles = np.angle(hat_h_ur)

        # Go through each user 
        for kk in range(num_users):

            # Store estimated Phi
            hat_Phi = np.diag(np.exp(1j * (hat_h_ur_angles[kk] - g_rb_angles)))

            h_eq_chest[qq, kk] = np.matmul(np.matmul(g_rb.conj().T, hat_Phi), h_ur[kk, :, np.newaxis])

    # Compute the SNR of each user when using beam sweeping
    snr_chest_db = 10 * np.log10(np.abs(h_eq_chest)**2) - NOISE_POWER


    fig, ax = plt.subplots()

    # Get CDF
    x_cdf_bsw_db, y_cdf_bsw_db = ecdf(snr_bsw_db)

    ax.plot(x_cdf_bsw_db, y_cdf_bsw_db, color='black', label='beam sweeping')

    # Go through all the qualities 
    for qq, quality in enumerate(est_qualities_):

        # Get CDF
        x_cdf_chest_db, y_cdf_chest_db = ecdf(snr_chest_db[qq])

        ax.plot(x_cdf_chest_db, y_cdf_chest_db, label=(r'CHEST: $\alpha=' + str(quality) + '$'))

    ax.set_xlabel('SNR over noise floor [dB]')
    ax.set_ylabel('ECDF') 

    ax.legend()

    plt.show()


    # saving data
    if render:
        np.savez(filename,  h_ris=h_ris)
    print('\t...DONE')

