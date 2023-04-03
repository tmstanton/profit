# relevant imports
import numpy as np
import matplotlib.pyplot as plt
import profit
from profit import utils, manual, auto, plots
from typing import Callable as func
import sys
import astropy.modeling as apm
from uncertainties import ufloat, umath

# -=-=-=- Fitting methods for different Lines -=-=-=-

def Fit(name:str, specpath:str, outpath:str, redshift:float, zerr:float=0.05, mode:str='manual', readmethod:func = utils.kmos1D_n) -> None:
    
    valid_types = {
        '1':'single',
        '2':'double',
        '3':'stacked',
        '4':'lorentzian',
        '5': 'SKIP'
    }
    if mode == 'auto':
        fit_type = profit.options['fit_mode']

    # read in data
    wavelengths, fluxes, errors = readmethod(name, specpath) #utils.kmos1D(name, specpath)

    # calculate approximate location of the line
    approx_wl = np.mean(profit.options['line_wl']) * (1. + redshift)
    profit.options['redshift'] = float(redshift)

    # mask out all but relevant region
    mask = (wavelengths >= approx_wl - 75.) & (wavelengths <= approx_wl + 75.)
    wl_fit, flux_fit, errs_fit = wavelengths[mask], fluxes[mask], errors[mask]

    # set initial loop parameters
    accepted = False
    run_continuum = True
    run_fit = True

    plots.InitialPlot(wl_fit, flux_fit, errs_fit, approx_wl, '', verbose=False)
    profit.options['open'] = True

    if profit.options['fit_mode'] is None:
        fit_input = input('1: Single Gaussian\n2: Double Gaussian\n3: Stacked Gaussian\n4: Lorentzian\n5: SKIP\n-> [profit]: Select Type: ')
        if fit_input not in valid_types.keys():
            print('-> [profit]: Invalid fitting type provided. Exiting...')
            sys.exit()
        elif fit_input == '5':
            # skipping conditions
            accepted = True
            run_continuum = False
            run_fit = False
            fit_type = 'SKIP'
        else:
            fit_type = valid_types[fit_input]
            if mode == 'auto':
                profit.options['fit_mode'] = valid_types[fit_input]

    if profit.options['open']: utils.ClosePlot()

    # give continuum normalisation option
    while run_fit:

        plots.InitialPlot(wl_fit, flux_fit, errs_fit, approx_wl, fit_type, verbose=False)
        profit.options['open'] = True

        # check continuum normalisation
        if run_continuum:
            continuum = True if input('-> [profit]: Continuum Normalisation required (y/n): ').lower() == 'y' else False
            run_continuum = False 

        if not continuum:
            break

        # get linear points
        print('\n-> [profit]: normalising by a linear continuum: ')
        lower_region = input('   -> Lower Wavelength Range (x, y): ').split(' ')
        upper_region = input('   -> Upper Wavelength Range (x, y): ').split(' ')

        # close oritinal plot
        if profit.options['open']: plt.close()

        # masking and average points
        lowermask = (wl_fit >= int(lower_region[0])) & (wl_fit <= int(lower_region[1]))
        uppermask = (wl_fit >= int(upper_region[0])) & (wl_fit <= int(upper_region[1]))

        continuum_fit_wl = np.concatenate((wl_fit[lowermask], wl_fit[uppermask]))
        continuum_fit_fl = np.concatenate((flux_fit[lowermask], flux_fit[uppermask]))

        # linear fit 
        cont_fit_params = np.polyfit(continuum_fit_wl, continuum_fit_fl, deg=1)

        # check continuum fit:
        plot_fit = np.copy(flux_fit)
        good_fit = plots.ContinuumPlot(wl_fit, plot_fit, errs_fit, approx_wl, fit_type, p=cont_fit_params)
        plt.close()

        # if good fit, apply and break:
        if good_fit:
            flux_fit -= cont_fit_params[0]*wl_fit + cont_fit_params[1]
            break
        print('-> [profit]: re-running continuum normalisation...\n')

    if profit.options['open'] == True: utils.ClosePlot()

    # run fitting
    while not accepted:

        print(f'-> [profit]: running {fit_type} fitting of {profit.options["line_name"]}...')

        # show graph if manual
        if mode == 'manual': 
            plots.InitialPlot(wl_fit, flux_fit, errs_fit, approx_wl, fit_type)
            profit.options['open'] = True

        if mode == 'auto':
            print('-> [profit]: Automated codes not fully implemented.')
            sys.exit()

        if fit_type == 'single':
            cen = profit.options['line_wl'][0]
            if mode == 'manual':
                params, accepted = manual.GaussianFit(wl=wl_fit, flux=flux_fit, errs=errs_fit,
                                                   cwl=cen, z=redshift, zerr=zerr)
            elif mode == 'auto': # TODO: check this is up to date
                params, accepted = auto.GaussianFit(wl=wl_fit, flux=flux_fit, errs=errs_fit,
                                                   cwl=cen, z=redshift, zerr=zerr)
        if fit_type == 'double':
            cen = profit.options['line_wl']
            if mode == 'manual': 
                params, accepted = manual.DoubleGaussianFit(wl=wl_fit, flux=flux_fit, errs=errs_fit,
                                                   wl1=cen[0], wl2=cen[1], z=redshift, zerr=zerr)
            elif mode == 'auto': # TODO: check this is up to date
                params, accepted = auto.DoubleGaussianFit(wl=wl_fit, flux=flux_fit, errs=errs_fit,
                                                   wl1=cen[0], wl2=cen[1], z=redshift, zerr=zerr)
        if fit_type == 'lorentzian':
            cen = profit.options['line_wl'][0]
            if mode == 'manual':
                params, accepted = manual.LorentzianFit(wl=wl_fit, flux=flux_fit, errs=errs_fit,
                                                   cwl=cen, z=redshift, zerr=zerr)
            elif mode == 'auto': # TODO: check this is up to date
                params, accepted = auto.LorentzianFit(wl=wl_fit, flux=flux_fit, errs=errs_fit,
                                                   cwl=cen, z=redshift, zerr=zerr)
        if fit_type == 'stacked':
            cen = profit.options['line_wl'][0]
            if mode == 'manual': 
                params, accepted = manual.StackedGaussianFit(wl=wl_fit, flux=flux_fit, errs=errs_fit,
                                                    cwl=cen, z=redshift, zerr=zerr)
            elif mode == 'auto': # TODO: check this is up to date
                params, accepted = auto.StackedGaussianFit(wl=wl_fit, flux=flux_fit, errs=errs_fit,
                                                    cwl=cen, z=redshift, zerr=zerr)  

    # create final dictionary
    fit_params = {}

    if fit_type == 'single':

        # general info
        fit_params['type']        = fit_type
        fit_params['success']     = params['success']
        if params['success']:
            fit_params['comment'] = 'Single Gaussian Fit'
        else:
            fit_params['comment'] = 'Failed fit'
        # data
        fit_params['z']           = params['z'][0]
        fit_params['z_err']       = max(params['z'][1], params['z'][2])
        fit_params['log_amp']     = params['amp'][0]
        fit_params['sig']         = params['sig'][0]
        fit_params['sig_err']     = max(params['sig'][1], params['sig'][2])
        fit_params['fwhm']        = params['fwhm'][0]
        fit_params['flux']        = params['flux'][0]
        fit_params['flux_err']    = max(params['flux'][1], params['flux'][2])
        fit_params['fwhm']        = params['fwhm'][0]
        fit_params['fwhm_err']    = max(params['fwhm'][1], params['fwhm'][2])
        fit_params['snr']          = fit_params['flux'] / fit_params['flux_err']

    elif fit_type == 'double':

        # general info
        fit_params['type']          = fit_type
        fit_params['success']       = params['success']
        if params['success']:
            fit_params['comment']   = 'Double Gaussian Fit'
        else:
            fit_params['comment']   = 'Failed fit'
        fit_params['line']          = profit.options['line_name']
        # data
        fit_params['z']             = params['z'][0]
        fit_params['z_err']         = max(params['z'][1], params['z'][2])
        fit_params['low_log_amp']   = params['amp1'][0]
        fit_params['low_sig']       = params['sig'][0]
        fit_params['low_flux']      = params['flux_1'][0]
        fit_params['low_flux_err']  = max(params['flux_1'][1], params['flux_1'][2])
        fit_params['high_log_amp']  = params['amp2'][0]
        fit_params['high_sig']      = params['sig'][0]
        fit_params['high_flux']     = params['flux_2'][0]
        fit_params['high_flux_err'] = max(params['flux_2'][1], params['flux_2'][2])
        fit_params['sig_err']       = max(params['sig'][1], params['sig'][2])
        fit_params['fwhm']          = params['fwhm'][0]
        fit_params['fwhm_err']      = max(params['fwhm'][1], params['fwhm'][2])

        # handle combination
        left = ufloat(fit_params['high_flux'], fit_params['low_flux_err'])
        right = ufloat(fit_params['flow_flux'], fit_params['high_flux_err'])
        fit_params['flux']        = (left + right).n
        fit_params['flux_err']    = (left + right).s
        fit_params['snr']         = fit_params['flux'] / fit_params['flux_err']   

    elif fit_type == 'lorentzian':

        # general info
        fit_params['type']        = fit_type
        fit_params['success']     = params['success']
        if params['success']:
            fit_params['comment'] = 'Lorentzian Fit'
        else:
            fit_params['comment'] = 'Failed fit'
        fit_params['line']        = profit.options['line_name']
        # data
        fit_params['z']           = params['z'][0]
        fit_params['z_err']       = max(params['z'][1], params['z'][2])
        fit_params['amp']         = params['amp'][0]
        fit_params['gamma']       = params['gamma'][0]
        fit_params['flux']        = params['flux'][0]
        fit_params['flux_err']    = max(params['flux'][1], params['flux'][2])
        fit_params['fwhm']        = params['fwhm'][0]
        fit_params['fwhm_err']    = max(params['fwhm'][1], params['fwhm'][2])
        fit_params['snr']         = fit_params['flux'] / fit_params['flux_err']

    elif fit_type == 'stacked':

        # general info
        fit_params['type']        = fit_type
        fit_params['success']     = params['success']
        if params['success']:
            fit_params['comment'] = 'Stacked Gaussian Fit'
        else:
            fit_params['comment'] = 'Failed fit'
        fit_params['line']        = profit.options['line_name']
        # data
        fit_params['z']           = params['z'][0]
        fit_params['z_err']       = max(params['z'][1], params['z'][2])
        fit_params['amp_n']       = params['amp_n'][0]
        fit_params['amp_b']       = params['amp_b'][0]
        fit_params['sig_n']       = params['sig_n'][0]
        fit_params['sig_b']       = params['sig_b'][0]
        fit_params['fwhm_n']      = params['fwhm_n'][0]
        fit_params['fwhm_n_err']  = max(params['fwhm_n'][1], params['fwhm_n'][2])
        fit_params['fwhm_b']      = params['fwhm_b'][0]
        fit_params['fwhm_b_err']  = max(params['fwhm_b'][1], params['fwhm_b'][2])
        fit_params['flux_n']      = params['flux_n'][0]
        fit_params['flux_n_err']  = max(params['flux_n'][1], params['flux_b'][2])
        fit_params['flux_b']      = params['flux_b'][0]
        fit_params['flux_b_err']  = max(params['flux_b'][1], params['flux_b'][2])
        
        # handle combination
        narrow = ufloat(fit_params['flux_n'], fit_params['flux_n_err'])
        broad = ufloat(fit_params['flux_b'], fit_params['flux_b_err'])
        fit_params['flux']        = (narrow + broad).n
        fit_params['flux_err']    = (narrow + broad).s
        fit_params['snr']         = fit_params['flux'] / fit_params['flux_err']

    elif fit_type == 'SKIP':
        # general info
        fit_params['type']     = fit_type
        fit_params['success']  = False
        fit_params['comment']  = 'Skipped'
        # data
        fit_params['z']        = -999
        fit_params['z_err']    = -999
        fit_params['log_amp']  = -999
        fit_params['sig']      = -999
        fit_params['fwhm']     = -999
        fit_params['flux']     = -999
        fit_params['flux_err'] = -999
        fit_params['fwhm']     = -999
        fit_params['fwhm_err'] = -999
        fit_params['snr']      = -999

    # save pickle file
    if len(profit.options['rslt_dir']) < 1:
        print(f'-> [profit]: saving fit data to {outpath}\n')
        utils.PklSave(filepath=f'{outpath}/{name}_{mode[0]}.pkl', results=fit_params)
    else:
        print(f'-> [profit]: saving fit data to {profit.options["rslt_dir"]}\n')
        utils.PklSave(filepath=f'{profit.options["rslt_dir"]}/{profit.options["line_path"]}_{mode[0]}.pkl', results=fit_params)

    return None