import numpy as np

import matplotlib.pyplot as plt

from matplotlib import rc

from environment import ecdf

rc('font', **{'family': 'sans serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# Define setups
setups = ['ob-cc', 'ib-no', 'ib-wf']
labels_setups = ['OB-CC', 'IB-no', 'IB-wf']

# Define paradigms
paradigms = ['opt_ce_', 'cb_bsw_fixed_', 'cb_bsw_flexible_']
styles = ['-', '--', ':']
labels = ['OPT-CE', 'CB-BSW: Fixed', 'CB-BSW: Flexible']

fig, axes = plt.subplots(nrows=3)

for ss, setup in enumerate(setups):

    print('----- setup: ' + str(setup) + ' -----')

    for pp, paradigm in enumerate(paradigms):

        # File name
        datafilename = 'data/' + paradigm + setup + '.npz'

        # Load data
        rate = np.load(datafilename)['rate']

        # Get CDF
        x_, y_ = ecdf(rate)

        axes[ss].plot(x_, y_, linewidth=1.5, linestyle=styles[pp], label=labels[pp])

        print(labels[pp] + ' = ' + str(np.mean(rate)))

    axes[ss].set_title(labels_setups[ss])

    axes[ss].set_xlabel('rate [bit/Hz/s]')
    axes[ss].set_ylabel('CDF')

axes[0].legend()

plt.tight_layout(h_pad=.15)

plt.show()