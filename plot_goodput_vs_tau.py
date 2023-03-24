import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

from environment import command_parser, ecdf, OUTPUT_DIR, TAU, BANDWIDTH
import scenario.common as cmn

rc('font', **{'family': 'sans serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# Define setups
setups = ['ob-cc', 'ib-wf']
setups_labels = ['OB-CC', 'IB-CC']

# Define paradigms
paradigms = ['opt_ce_vs_tau_', 'cb_bsw_vs_tau_fixed_', 'cb_bsw_vs_tau_flexi_']
colors = ['black', 'blue', 'red']
styles = ['-', '--', ':']
labels = ['OPT-CE', 'CB-BSW: Fixed', 'CB-BSW: Flexible']

colors_snr = ['blue', 'red']
labels_snr = ['OPT-CE', 'CB-BSW']

# Define specific taus to plot
taus = [55.,  60., 65.]


if __name__ == '__main__':
    render, _, _, _ = command_parser()

    # define the axes for the specific taus
    plot_list = [plt.subplots(nrows=len(setups_labels)) for _ in taus]

    # plots varying tau
    fig_me, axes_me = plt.subplots(nrows=len(setups_labels))
    fig_snr, axes_snr = plt.subplots(figsize=(5, 2.5))
    fig_kpi, axes_kpi = plt.subplots()

    # Optimal KPI for BSW
    kpi_flexi = np.load('data/cb_bsw_opt_kpi.npz')['flexi']
    kpi_fixed = np.load('data/cb_bsw_opt_kpi.npz')['fixed']

    axes_kpi.plot(TAU, kpi_fixed, linewidth=1.5, linestyle=styles[1], label=labels[1], color=colors[1])
    axes_kpi.plot(TAU, kpi_flexi, linewidth=1.5, linestyle=styles[2], label=labels[2], color=colors[2])

    # Analysis for different CC and paradigms
    for ss, setup in enumerate(setups):

        for pp, paradigm in enumerate(paradigms):

            # File name
            datafilename = 'data/' + paradigm + setup + '.npz'

            # SNR
            if ss == 0 and pp < 2:
                snr_true = np.load(datafilename)['snr_true']
                snr_esti = np.load(datafilename)['snr_esti']
                x_, y_ = ecdf(snr_true)
                axes_snr.plot(10*np.log10(x_[::10]), y_[::10], linewidth=1.5, linestyle='-', label=f'{labels_snr[pp]}: true', color=colors_snr[pp])
                x_, y_ = ecdf(snr_esti)
                axes_snr.plot(10*np.log10(x_[::10]), y_[::10], linewidth=1.5, linestyle='--', label=f'{labels_snr[pp]}', color=colors_snr[pp])


            ## RATE (FAKE)
            # Load data
            goodput = BANDWIDTH * np.load(datafilename)['rate_real'] / 1e6

            mean_goodput_vs_tau =  np.mean(goodput, axis=1)
            axes_me[ss].plot(TAU, mean_goodput_vs_tau, linewidth=1.5, linestyle=styles[pp], label=labels[pp], color=colors[pp])

            # Get CDF for the specific taus
            for tt, tau in enumerate(taus):
                x_, y_ = ecdf(goodput[TAU == tau])
                plot_list[tt][1][ss].plot(x_, y_, linewidth=1.5, linestyle=styles[pp], label=labels[pp], color=colors[pp])
                plot_list[tt][1][ss].set_title(setups_labels[ss])

    # Printing plots

    # Rate CDF
    for tt, tau in enumerate(taus):
        plot_list[tt][0].suptitle(r'CDF with $\tau =' + f'{tau:.1f}$ [ms]', fontsize=12)
        cmn.printplot(plot_list[tt][0], plot_list[tt][1], render, filename=f'rate_cdf_tau{tau:.1f}', dirname=OUTPUT_DIR,
                      labels=['$R$ [Mbit/s]', 'CDF', 'CDF', 'CDF'], orientation='vertical')

    # Average rate vs tau
    fig_me.suptitle(r'Average rate vs $\tau$', fontsize=12)
    cmn.printplot(fig_me, axes_me, render, filename='mean_rate', dirname=OUTPUT_DIR,
                  labels=[r'$\tau$ [ms]', '$R$ [Mbit/s]', '$R$ [Mbit/s]', '$R$ [Mbit/s]'], orientation='vertical')

    # SNR CDF
    cmn.printplot(fig_snr, axes_snr, render, filename='snr_cdf', dirname=OUTPUT_DIR,
                  labels=['$\gamma$ [dB]', 'CDF'])

    # Opt KPI for BSW
    cmn.printplot(fig_kpi, axes_kpi, render, filename='kpi_vs_tau', dirname=OUTPUT_DIR,
                  labels=[r'$\tau$ [ms]', '$\gamma_0$ [dB]'])