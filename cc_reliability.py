import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


import scenario.common as cmn
from environment import command_parser, TAU, BANDWIDTH, OUTPUT_DIR, T

mpl.rc('font', **{'family': 'sans serif', 'serif': ['Computer Modern']})
mpl.rc('text', usetex=True)

# Setup option
setups = ['ob-cc', 'ib-wf']
setups_labels = ['OB-CC', 'IB-CC']
colors = ['black', 'blue']
styles = ['-', '--']
labels = ['OPT-CE', 'CB-BSW']

# Conf
C_ce = 100
N = 100
C_cb = 34

# Guard period
ratio = 10
tau_s = T / ratio

# Bandwidth
B_cc = BANDWIDTH * 5

# Bit number
b_id = 8
b_frame = 16
b_guard = 16
b_conf = 8
b_se = 6
b_quant = 2

# The followings are a vector representing [SET-U, ACK-U, SET-R, ACK-R]
b_opt = b_id + 1 + np.array([b_frame + b_guard + b_conf, b_se, b_frame + C_ce * b_conf, N * b_quant])
b_bsw = b_id + 1 + np.array([b_frame + b_guard + b_conf, 0, b_frame + C_cb * b_conf, b_conf])
t_opt = np.array([T - tau_s, T, T, T]) * 1e-3
t_bsw = np.array([T - tau_s, T - tau_s, T, T]) * 1e-3

# Define the mean snr parameter
snr_db = np.arange(0, 30.1, 0.1)
snr = cmn.db2lin(snr_db)

# Target pcc
target_pcc = 0.99

if __name__ == '__main__':
    render, _, _, _ = command_parser()

    # Compute the prob of each packet
    pkt_err_prob_opt = 1 - np.exp(- (2 ** (b_opt / t_opt / B_cc) - 1) / np.repeat(snr[np.newaxis], len(b_opt), axis=0).T)
    pkt_err_prob_bsw = 1 - np.exp(- (2 ** (b_bsw / t_bsw / B_cc) - 1) / np.repeat(snr[np.newaxis], len(b_bsw), axis=0).T)

    # OBCC
    pcc_opt_obcc = np.prod(1 - pkt_err_prob_opt[:, :2], axis=-1)
    pcc_bsw_obcc = np.prod(1 - pkt_err_prob_bsw[:, :2], axis=-1)
    # Take the first value
    ind_opt = np.argmax(pcc_opt_obcc >= target_pcc)
    x_target_opt = snr_db[ind_opt]
    y_target_opt = pcc_opt_obcc[ind_opt]
    ind_bsw = np.argmax(pcc_bsw_obcc >= target_pcc)
    x_target_bsw = snr_db[ind_opt]
    y_target_bsw = pcc_bsw_obcc[ind_opt]
    # Plotting values
    fig, ax = plt.subplots()
    ax.plot(snr_db, pcc_opt_obcc, ls=styles[0], color=colors[0], label=labels[0])
    ax.plot(snr_db, pcc_bsw_obcc, ls=styles[1], color=colors[1], label=labels[1])
    ax.vlines(x_target_opt, np.min(pcc_opt_obcc), y_target_opt, linestyle=':', color=colors[0])
    ax.vlines(x_target_bsw, np.min(pcc_opt_obcc), y_target_bsw, linestyle=':', color=colors[1])
    ax.hlines(target_pcc, snr_db[0], snr_db[-1], linestyles=':', color='gray')

    cmn.printplot(fig, ax, render, filename='obcc_pcc_vs_snr', dirname=OUTPUT_DIR,
                  labels=['$\lambda_u$ [dB]','$p_\mathrm{cc}$'])

    # IBCC
    pcc_opt_ibcc = np.zeros((len(snr), len(snr)))
    pcc_bsw_ibcc = np.zeros((len(snr), len(snr)))
    for uu, snr_u in enumerate(snr):
        for rr, snr_r in enumerate(snr):
            pcc_opt_ibcc[uu, rr] = (1 - pkt_err_prob_opt[uu, 0]) * (1 - pkt_err_prob_opt[uu, 1]) * (1 - pkt_err_prob_opt[rr, 2]) * (1 - pkt_err_prob_opt[rr, 3])
            pcc_bsw_ibcc[uu, rr] = (1 - pkt_err_prob_bsw[uu, 0]) * (1 - pkt_err_prob_bsw[uu, 1]) * (1 - pkt_err_prob_bsw[rr, 2]) * (1 - pkt_err_prob_bsw[rr, 3])

    # Plot definitions

    masked_opt_ibcc = np.ma.masked_where(pcc_opt_ibcc < target_pcc, pcc_bsw_ibcc)
    masked_bsw_ibcc = np.ma.masked_where(pcc_bsw_ibcc < target_pcc, pcc_bsw_ibcc)

    fig, ax = plt.subplots(1, 2)

    cmap = mpl.colormaps['Oranges']
    cmap.set_bad(color='white')
    ax[0].imshow(masked_opt_ibcc, cmap=cmap, vmin=target_pcc, vmax=1, extent=[min(snr_db), max(snr_db), min(snr_db), max(snr_db)],
                 interpolation='nearest', origin='lower', aspect='auto')
    ax[0].axis('scaled')
    tmp = ax[1].imshow(masked_bsw_ibcc, cmap=cmap, vmin=target_pcc, vmax=1, extent=[min(snr_db), max(snr_db), min(snr_db), max(snr_db)],
                       interpolation='nearest', origin='lower', aspect='auto')
    ax[1].axis('scaled')

    fig.colorbar(tmp, ax=ax, label=r'$p_\mathrm{cc}$')
    cmn.printplot(fig, ax, render,
                  f'pcc_heatmap', OUTPUT_DIR, orientation='horizontal',
                  title=labels, labels=[r'$\lambda_r$ [dB]', r'$\lambda_r$ [dB]', r'$\lambda_u$ [dB]'], grid=False, tight_layout=False)

