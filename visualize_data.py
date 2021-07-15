import os, sys
import time
import numpy as np
from options.export_config import DataVizOptions
from data_handlers.data_viz import DataViz


if __name__ == "__main__":
    np.random.seed(21)
    opts = DataVizOptions().parse()
    start_time = time.time()

    data_viz = DataViz(opts.data_dir)
    data_viz.convert_pickle_to_df(overwrite=True)
    data_viz.plot_volumes()

    if opts.plot_mu:
        data_viz.plot_eccentricities()

    if opts.plot_Am:
        data_viz.plot_orientation_angles()

    if opts.plot_bs:
        data_viz.plot_basal_slanting()



