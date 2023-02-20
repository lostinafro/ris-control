try:
    import cupy as np
except ImportError:
    import numpy as np

import matplotlib.pyplot as plt

import scenario.common as cmn
from environment import RIS2DEnv, command_parser, ecdf, NOISE_POWER_dBm, OUTPUT_DIR

noise_power = cmn.dbm2watt(NOISE_POWER_dBm)
tx_power = 1     # Transmit power

# Parameter for saving data
prefix = '2D_'

# For grid mesh
num_users = int(1e4)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # The following parser is used to impose some data without the need of changing the script (run with -h flag for
    # help) Render bool needs to be True to save the data If no arguments are given the standard value are loaded (
    # see environment) datasavedir should be used to save numpy arrays
    render, side_x, h, name, datasavedir = command_parser()
    side_y = side_x
    prefix = prefix + name

    # Generate users
    x = side_x * np.random.rand(num_users, 1)
    y = side_y * np.random.rand(num_users, 1)

    ue_pos = np.hstack((x, y, np.zeros((num_users, 1))))

    # Build environment
    env = RIS2DEnv(ue_position=ue_pos, sides=np.array([side_x, side_y, h]))
    num_conf = env.ris.num_std_configs

    # Pre-allocation of variables
    h_ur = np.zeros((num_users, env.ris.num_els), dtype=complex)
    g_rb = np.zeros(env.ris.num_els, dtype=complex)
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
    h_eq_cb = np.matmul(np.matmul(g_rb.conj().T[np.newaxis, :], Phi)[:, np.newaxis, :, :], h_ur[np.newaxis, :, :, np.newaxis]).squeeze()

    # Generate some noise
    var = noise_power / 2
    noise_ = np.sqrt(var) * (np.random.randn(num_users) + 1j * np.random.randn(num_users))

    # Compute the SNR of each user when using CB scheme
    sig_pow_cb = tx_power * np.abs(h_eq_cb) ** 2
    sig_pow_noisy_cb = tx_power * np.abs(h_eq_cb + noise_) ** 2

    snr_cb_db = 10 * np.log10(sig_pow_cb / noise_power)
    snr_cb_noisy_db = 10 * np.log10(sig_pow_noisy_cb / noise_power)

    # Generate estimation noise
    est_var = noise_power / 2 / env.ris.num_els
    est_noise_ = np.sqrt(est_var) * (np.random.randn(num_users, env.ris.num_els) + 1j * np.random.randn(num_users, env.ris.num_els))

    # Get estimated channel coefficients
    z_eq = h_ur.conj() * g_rb[np.newaxis, :]
    z_hat = z_eq + est_noise_

    # Get estimated Phi
    Phi_hat = np.exp(np.angle(-z_hat))

    # Compute equivalent channel
    h_eq_chest = ((g_rb.conj()[np.newaxis] * Phi_hat) * h_ur).sum(axis=-1)
    h_eq_chest_hat = (Phi_hat * z_hat).sum(axis=-1)

    # Compute the SNR of each user when using OC
    sig_pow_oc = tx_power * np.abs(h_eq_chest) ** 2
    sig_pow_oc_hat = tx_power * np.abs(h_eq_chest_hat) ** 2

    snr_oc_db = 10 * np.log10(sig_pow_oc / noise_power)
    snr_oc_hat_db = 10 * np.log10(sig_pow_oc_hat / noise_power)

    ##############################
    # Plot
    ##############################
    x_cdf_cb_db, y_cdf_cb_db = ecdf(snr_cb_db)
    x_cdf_cb_noisy_db, y_cdf_cb_noisy_db = ecdf(snr_cb_noisy_db)
    x_cdf_oc_db, y_cdf_oc_db = ecdf(snr_oc_db)
    x_cdf_oc_hat_db, y_cdf_oc_hat_db = ecdf(snr_oc_hat_db)

    try:    # explicit conversion to numpy for compatibility reason with cupy
        x_cdf_cb_db = np.asnumpy(x_cdf_cb_db)
        y_cdf_cb_db = np.asnumpy(y_cdf_cb_db)
        x_cdf_cb_noisy_db = np.asnumpy(x_cdf_cb_noisy_db)
        y_cdf_cb_noisy_db = np.asnumpy(y_cdf_cb_noisy_db)
        x_cdf_oc_db = np.asnumpy(x_cdf_oc_db)
        y_cdf_oc_db = np.asnumpy(y_cdf_oc_db)
        x_cdf_oc_hat_db = np.asnumpy(x_cdf_oc_hat_db)
        y_cdf_oc_hat_db = np.asnumpy(y_cdf_oc_hat_db)
    except AttributeError:
        pass

    fig, axes = plt.subplots(ncols=2, sharey='all', figsize=(12, 4))
    axes[0].plot(x_cdf_cb_db, y_cdf_cb_db, linewidth=1.5, color='black', label='CB: true')
    axes[0].plot(x_cdf_cb_noisy_db, y_cdf_cb_noisy_db, linewidth=1.5, linestyle='--', color='black', label='CB: noisy')
    axes[0].plot(x_cdf_oc_db, y_cdf_oc_db, linewidth=1.5, color='tab:blue', label=r'OC: true')
    axes[0].plot(x_cdf_oc_hat_db, y_cdf_oc_hat_db, linewidth=1.5, linestyle='--', color='tab:blue',  label=r'OC: estimated')

    axes[1].plot(x_cdf_cb_db / Phi.shape[0], y_cdf_cb_db, color='black')
    axes[1].plot(x_cdf_oc_db / env.ris.num_els, y_cdf_oc_db)

    axes[0].legend()

    plt.show()


    # This is a wrap function to show or save plots conveniently
    # If render == True this will save the data in ris-protocol/plots/{dateoftoday}/
    #cmn.printplot(fig, axes, render, filename=f'{prefix}' + 'oc_vs_cb', dirname=OUTPUT_DIR,
                  #labels=['SNR over noise floor [dB]', 'norm. SNR over noise floor [dB]', 'ECDF'],
                  #orientation='horizontal')

