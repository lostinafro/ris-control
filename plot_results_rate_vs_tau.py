import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

from environment import ecdf, OUTPUT_DIR, TAU
import scenario.common as cmn

rc('font', **{'family': 'sans serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# Define setups
setups = ['ob-cc', 'ib-wf']
labels_setups = ['OB-CC', 'IB-CC']

# Define paradigms
paradigms = ['opt_ce_vs_tau_', 'cb_bsw_vs_tau_fixed_', 'cb_bsw_vs_tau_flexible_']
styles = ['-', '--', ':']
labels = ['OPT-CE', 'CB-BSW: Fixed', 'CB-BSW: Flexible']

colors_snr = ['blue', 'red']
labels_snr = ['OPT-CE', 'CB-BSW']

render = False

# generate plots with different tau
fig_lo, axes_lo = plt.subplots(nrows=len(labels_setups))
fig_hi, axes_hi = plt.subplots(nrows=len(labels_setups))
fig_me, axes_me = plt.subplots(nrows=len(labels_setups))
fig_snr, axes_snr = plt.subplots(figsize=(5, 2.5))
fig_kpi, axes_kpi = plt.subplots()

kpi_flexi = np.load('data/cb_bsw_opt_kpi.npz')['flexi']
kpi_fixed = np.load('data/cb_bsw_opt_kpi.npz')['fixed']

axes_kpi.plot(TAU, kpi_fixed, linewidth=1.5, linestyle=':', label=f'CB-BSW: Fixed', color='red')
axes_kpi.plot(TAU, kpi_flexi, linewidth=1.5, linestyle='--', label=f'CB-BSW: Flexible', color='blue')


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


        ## RATE
        # Load data
        rate = np.load(datafilename)['rate']

        mean_rate_vs_tau = np.mean(rate, axis=1)
        axes_me[ss].plot(TAU, mean_rate_vs_tau, linewidth=1.5, linestyle=styles[pp], label=labels[pp])

        # Get CDF for tau = 7.5 ms
        x_, y_ = ecdf(rate[TAU == 8.125])
        axes_lo[ss].plot(x_, y_, linewidth=1.5, linestyle=styles[pp], label=labels[pp])

        # Get CDF for tau = 15 ms
        x_, y_ = ecdf(rate[TAU == 10.])
        axes_hi[ss].plot(x_, y_, linewidth=1.5, linestyle=styles[pp], label=labels[pp])


    axes_me[ss].set_title(labels_setups[ss])
    axes_lo[ss].set_title(labels_setups[ss])
    axes_hi[ss].set_title(labels_setups[ss])

fig_lo.suptitle(r'CDF with $\tau = 8.125$ ms', fontsize=12)
fig_hi.suptitle(r'CDF with $\tau = 10$ ms', fontsize=12)
fig_me.suptitle(r'Average rate vs $\tau$', fontsize=12)


cmn.printplot(fig_me, axes_me, render, filename='mean_rate', dirname=OUTPUT_DIR,
              labels=[r'$\tau$ [ms]', '$R$ [bit/Hz/s]', '$R$ [bit/Hz/s]', '$R$ [bit/Hz/s]'], orientation='vertical')
cmn.printplot(fig_lo, axes_lo, render, filename='rate_cdf_lo', dirname=OUTPUT_DIR,
              labels=['$R$ [bit/Hz/s]', 'CDF', 'CDF', 'CDF'], orientation='vertical')
cmn.printplot(fig_hi, axes_hi, render, filename='rate_cdf_hi', dirname=OUTPUT_DIR,
              labels=['$R$ [bit/Hz/s]', 'CDF', 'CDF', 'CDF'], orientation='vertical')

cmn.printplot(fig_snr, axes_snr, render, filename='snr_cdf', dirname=OUTPUT_DIR,
              labels=['$\gamma$ [dB]', 'CDF'])

cmn.printplot(fig_kpi, axes_kpi, render, filename='kpi_vs_tau', dirname=OUTPUT_DIR,
              labels=[r'$\tau$ [ms]', '$\gamma_0$ [dB]'])