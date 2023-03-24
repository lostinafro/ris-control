import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

import scenario.common as cmn
from environment import command_parser, TAU, BANDWIDTH, OUTPUT_DIR

rc('font', **{'family': 'sans serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# Setup option
setups = ['ob-cc', 'ib-wf']
setups_labels = ['OB-CC', 'IB-CC']
paradigms = ['opt_ce_vs_tau_', 'cb_bsw_vs_tau_fixed_', 'cb_bsw_vs_tau_flexi_']
colors = ['black', 'blue', 'red']
styles = ['-', '--', ':']
labels = ['OPT-CE', 'CB-BSW: Fixed', 'CB-BSW: Flexible']

# point for the probability
num_points = 50
perr = np.logspace(-3, 0, base=10, num=num_points)

# Define specific taus to plot
taus = [55., 60., 65.]
plot_list = [plt.subplots(nrows=len(setups_labels)) for _ in taus]

if __name__ == '__main__':
    # The following parser is used to impose some data without the need of changing the script (run with -h flag for
    # help) Render bool needs to be True to save the data If no arguments are given the standard value are loaded (
    # see environment) datasavedir should be used to save numpy arrays
    render, _, _, _ = command_parser()


    # Cycling through different paradigms
    for pp, paradigm in enumerate(paradigms):
        # Cycling through different ccs
        for ss, setup in enumerate(setups):

            ## Load data obtained
            datafilename = 'data/' + paradigm + setup + '.npz'

            # Obtain the average goodput in Mbit/s evaluated
            rate = np.mean(BANDWIDTH * np.load(datafilename)['rate_real'] / 1e6, axis=1)

            # Obtain the utility
            utility = (1 - perr) * np.repeat(rate[np.newaxis].T, num_points, axis=1)

            # Plot for some values
            for tt, tau in enumerate(taus):
                plot_list[tt][1][ss].semilogx(perr, np.squeeze(utility[TAU == tau]), linewidth=1.5, linestyle=styles[pp], label=labels[pp], color=colors[pp])
                plot_list[tt][1][ss].set_title(setups_labels[ss])

    for tt, tau in enumerate(taus):
        plot_list[tt][0].suptitle(r'Utility with $\tau =' + f'{tau:.1f}$ [ms]', fontsize=12)
        cmn.printplot(plot_list[tt][0], plot_list[tt][1], render, filename=f'utility_tau{tau:.1f}', dirname=OUTPUT_DIR,
                      labels=['$1-p_\mathrm{cc}$', '$R$ [Mbit/s]', '$R$ [Mbit/s]'], orientation='vertical')