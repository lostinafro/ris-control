try:
    import cupy as np
except ImportError:
    import numpy as np
from os import path

import scenario.common as cmn
from environment import RIS2DEnv, command_parser

# Paramter for saving datas
prefix = 'random_users_'

# For grid mesh
num_users = int(1e3)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # The following parser is used to impose some data without the need of changing the script (run with -h flag for help)
    # Render bool needs gto be True to save the data
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

    ue_pos = np.hstack((x,y, np.zeros((num_users, 1))))

    # Build environment
    env = RIS2DEnv(ue_position=ue_pos, sides=np.array([side_x, side_y, h]))
    num_conf = env.ris.num_std_configs

    # Pre-allocation of variables
    h_ris = np.zeros((num_conf, num_users))

    # Looping through the configurations (progressbar is only a nice progressbar)
    for index in cmn.std_progressbar(range(num_conf)):
        # load the appropriate config
        actual_conf = env.set_std_conf_2D(index)
        # Compute the channel for each user
        h_ris[index] = env.build_ris_channel()
        # TODO: do the average and other stuff

    # saving data
    if render:
        np.savez(filename,  h_ris=h_ris)
    print('\t...DONE')

