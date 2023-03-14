# imports
from . import fitting, manual, utils, plots
import numpy as np

# set up options dictionary
def set_fit_options(options:dict) -> None:
    
    # set directories
    options['data_dir'] = ""
    options['line_dir'] = "" # use this later maybe, to make more general
    options['rslt_dir'] = ""
    options['plot_dir'] = ""

    # verbose
    options['verbose'] = True
    options['save_plots'] = False
    options['display_auto_plots'] = True

    # generate line data [adapt to later use lots of lines]
    options['O[III]λ5007']    = [5007, 'O[III]_5007']
    options['H[β]']           = [4861, 'H_Beta']
    options['Ne[III]λ3869']   = [3869, 'Ne[III]_3869']
    options['H[γ]']           = [4340, 'H_Gamma']
    options['O[III]λ4363']    = [4363, 'O[III]_4363']
    options['O[II]λλ3727,29'] = [3727, 3729, 'O[II]_3727,29']
    options['O[III]λ4960'] = [4960, 'O[III]_4960']
    options['He[II]λ4686'] = [4686, 'He[II]_4686']

    # plotting mode for automation
    options['fit_mode'] = None
    options['open'] = False

    # stacked gaussian profile velocities
    options['vel_barrier'] = 200 # km s-1

# initialise
options = {}
set_fit_options(options)
