# imports
#from .WIP import linfit2d
from . import fitting, manual, speclines, utils, plots

# set up options dictionary
def set_fit_options() -> None:
    
    options = {}

    # set directories
    options['data_dir'] = ""
    options['line_dir'] = "" # use this later maybe, to make more general
    options['rslt_dir'] = ""
    options['plot_dir'] = ""

    # verbose
    options['verbose'] = True
    options['save_plots'] = False
    options['display_auto_plots'] = True

    # plotting mode for automation
    options['fit_mode'] = None
    options['open'] = False

    # stacked gaussian profile velocities
    options['vel_barrier'] = 200 # km s-1

    # window size
    options['window_minus'] = 500.
    options['window_plus']  = 500.

    # select relevant windows
    options['manual_windows'] = False

    # set emission lines
    return options, speclines.Generate_Features()

# initialise
options, lines = set_fit_options()
