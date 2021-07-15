import argparse

class BaseOptions:
    """This class defines options used for the vtk pipeline.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initialized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options."""
        # basic parameters
        parser.add_argument("--vtk_mesh_dir", required=True, type=str, help="path to input vtk meshes.")
        parser.add_argument("--output_dir", required=True, type=str, help="path to series of outputs for storing.")
        parser.add_argument('--nodelabels_dir', required=True, type=str, help='path to csv file containing epi/endo/inner_rim/outer_rim node indices.')
        parser.add_argument('--rv_path', default=None, type=str, help='path to HFC rv direction meta-data folder, default x_axis')
        parser.add_argument('--num_disks', required=True, type=int, default=20, help='number of Simpsons disks.')
        parser.add_argument('--simpson_type', required=True, type=int, nargs="+", help='Simpson method type: 0 for BOD, 1 for TBC, and 2 for SBR.')
        parser.add_argument('--view_name', required=True, type=str, nargs="+", default='a3ch', help='Viewing angle of secondary apical view: a3ch for A4C-A3C and a2ch for A4C-A2C.')
        parser.add_argument('--a4c_offset', type=int, default=20, help='degree of offset from the RV_dir to get the A4C.')
        parser.add_argument('--dataset_type', required=True, type=str, nargs="+", default='UKBB', help='Dataset type: UKBB (n=4113), YHC (n=225) or HFC (n=50).')
        parser.add_argument('--ptsz_disp', required=True, type=int, default=1, help='Size of points when in display mode.')
        parser.add_argument('--enclosed', type=int, default=0, help='Whether or not mesh is closed at the base. Default False.')


        # not required aguments
        parser.add_argument('--anomaly', type=int, nargs='*', action='append', default=[], help='list of anomaly cases to remove from study.')
        parser.add_argument('--manual_apex', type=int, nargs='*', action='append', default=[], help='list of cases that require manual endo and apex nodes .')

        # arguments for how pipeline flows
        parser.add_argument('--exclude_landmarks', default=False, action='store_true', help='Does not run the landmark part of the population. If false, loads the landmarks from dataframe.')
        parser.add_argument('--exclude_volume', default=False, action='store_true', help='Does not run the volume estimation part of the population. If false, loads the volume data from dataframe.')
        parser.add_argument('--exclude_mesh_traits', default=False, action='store_true', help='Does not run the eccentricity and orientation angle part of the population. If false, loads the stat data from dataframe.')
        parser.add_argument('--exclude_eccent_profile', default=False, action='store_true', help='Does not run the eccentricity profiles.')

        # debugging
        parser.add_argument('--debug', default=False, action='store_true', help='Debug mode with aid of visualization.')
        parser.add_argument("--verbose", default=False, action="store_true", help="additional output information")
        parser.add_argument("--export_pkl", default=False, action="store_true", help="whether or not to save output")

        # visualization
        parser.add_argument('--data_viz', default=False, action='store_true', help='Data Visualization.')



        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional models-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in models and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options
        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        self.print_options(opt)
        self.opt = opt
        return self.opt



class DataVizOptions:
    """This class defines options used for the data visualization pipeline.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initialized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options."""
        # basic parameters
        parser.add_argument("--data_dir", required=True, type=str,
                            help="path to the outputs stored during vtk pipeline.")

        # debugging
        parser.add_argument('--plot_mu', default=False, action='store_true', help='Plot eccentricities histogram.')
        parser.add_argument("--plot_Am", default=False, action="store_true", help='Plot orientation angles histogram.')
        parser.add_argument("--plot_bs", default=False, action="store_true",
                            help='Plot basal slantings histogram.')

        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional models-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in models and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options
        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        self.print_options(opt)
        self.opt = opt
        return self.opt
