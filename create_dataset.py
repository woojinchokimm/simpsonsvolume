import logging
import os, sys
import time
import numpy as np
from options.export_config import BaseOptions
from data_handlers.vtkslicer import VTKSlicer
import copy


if __name__ == "__main__":
    np.random.seed(21)
    opts = BaseOptions().parse()
    start_time = time.time()
    output_dir = opts.output_dir

    # VTK subsection
    for dataset in opts.dataset_type:
        for typ in opts.simpson_type:
            for view_name in opts.view_name:
                lmk_start_time = time.time()

                new_opts = copy.deepcopy(opts)
                new_opts.dataset_type = dataset
                new_opts.simpson_type = typ
                new_opts.view_name = view_name

                vtk_slicer = VTKSlicer(new_opts)
                vtk_slicer.create_slices()
                logging.info(f'TIMER: VTK landmark generation processing took {lmk_start_time - start_time:.2f} seconds')


