# relevant imports
import numpy as np
import matplotlib.pyplot as plt
import profit # type: ignore
from profit import utils
from typing import Callable as func
import sys
import astropy.modeling as apm
from uncertainties import ufloat, umath

# -=-=-=- Fitting methods for different Lines -=-=-=-

def Fit(name:str, specpath:str, outpath:str="", redshift:float=0.0, zerr:float=0.05, 
        mode:str='manual', readmethod:func = utils.kmos1D, check_continuum:bool=True) -> None:
    
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
    wavelengths, fluxes, errors = readmethod(name, specpath) 
    if len(wavelengths) == len(fluxes) + 1: # TODO: figure out why, temporary fix
        wavelengths = wavelengths[:-1]

    if name == 'XL110239':
        fluxes *= -1.0

    # calculate approximate location of the line    
    approx_wl = np.mean(profit.Line.wls) * (1. + redshift)

    # in case of insecure / photometric redshift
    if profit.options['manual_windows']:
        
        # mask out all but relevant region
        wl_fit, flux_fit, errs_fit = wavelengths, fluxes, errors
        profit.plots.InitialPlot(wl_fit, flux_fit, errs_fit, approx_wl, '', verbose=False, xlow=None, xupp=None)
        profit.options['open'] = True

        while True:

            # check bounds
            accept = 'x'
            while not (accept in ['y', 'n']):
                accept = input('-> Manually selecting windows: happy with windows? (y/n): ').lower()

            if accept == 'y':
                
                # get wavelength of line to set fitting redshift
                approx_wl = utils.UserInput('Centroid Wavelength')
                utils.ClosePlot()

                # set masks
                profit.options['redshift'] = (approx_wl / np.mean(profit.Line.wls)) - 1.
                redshift = profit.options['redshift']
                mask = (wavelengths >= lower) & (wavelengths <= upper)
                wl_fit, flux_fit, errs_fit = wavelengths[mask], fluxes[mask], errors[mask]
                break

            else:

                lower = utils.UserInput('Lower Wavelength Bound')
                upper = utils.UserInput('Upper Wavelength Bound')
                utils.ClosePlot()

                mask = (wavelengths >= lower) & (wavelengths <= upper) # WIDENED
                wl_fit, flux_fit, errs_fit = wavelengths[mask], fluxes[mask], errors[mask]
                profit.plots.InitialPlot(wl_fit, flux_fit, errs_fit, approx_wl, '', verbose=False, xlow=lower, xupp=upper)
                profit.options['open'] = True

    # otherwise run
    else:

        profit.options['redshift'] = float(redshift)

        # mask out all but relevant region
        mask = (wavelengths >= approx_wl - profit.options['window_minus']) & (wavelengths <= approx_wl + profit.options['window_plus']) # WIDENED
        wl_fit, flux_fit, errs_fit = wavelengths[mask], fluxes[mask], errors[mask]

    # set initial loop parameters
    accepted = False
    run_continuum = True
    run_fit = True
    continuum = True
    continuum_normalised = False

    profit.plots.InitialPlot(wl_fit, flux_fit, errs_fit, approx_wl, '', verbose=False)
    profit.options['open'] = True

    # check for skip first
    fit = 'x' if input('\t-> Enter "x" to skip line profile: ').lower() == 'y' else 'n'
    if fit == 'x':
        accepted = True
        run_continuum = False
        run_fit = False
        fit_type = 'SKIP'

    # check for emission or absorption line
    if profit.Line.emission : 

        # handle different line types
        if profit.Line.type == 'singlet':
            
            print('-> [profit]: Fitting single gaussian profile')

            # handle potential stacked gaussian case
            fit_type  = 'stacked' if input('\t-> Enter "y" to fit for stacked gaussian profile: ').lower() == 'y' else 'single'

        elif profit.Line.type == 'doublet':

            print('-> [profit]: Fitting single gaussian profile')
            fit_type  = 'double'

        elif profit.Line.type == 'triplet':

            print('-> [profit]: Fitting single gaussian profile')
            fit_type  = 'triple'

        else: raise ValueError('-> [profit]: Unknown Line type presented - check pkl files...')
    
    else: raise ValueError('-> [profit]: ABSORPTION LINES TO BE ADDED IN A FUTURE UPDATE, EXITING')
        
    if profit.options['open']: profit.utils.ClosePlot()

    # give continuum normalisation option
    while run_fit:

        profit.plots.InitialPlot(wl_fit, flux_fit, errs_fit, approx_wl, fit_type, verbose=False)
        profit.options['open'] = True

        # check continuum normalisation
        if run_continuum & check_continuum:
            continuum = True if input('-> [profit]: Continuum Normalisation required (y/n): ').lower() == 'y' else False
            run_continuum = False 

        if not continuum:
            break

        # get linear points
        print('\n-> [profit]: normalising by a linear continuum: ')
        while True:
            try:
                lower_region = input('   -> Lower Wavelength Range (x, y): ').split(' ')
                upper_region = input('   -> Upper Wavelength Range (x, y): ').split(' ')
                for low, upp in zip(lower_region, upper_region):
                    check = int(low[0])
                    check = int(upp[0])
                break
            except:
                print('-> [ERROR]: Invalid Ranges given. ')

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
        good_fit = profit.plots.ContinuumPlot(wl_fit, plot_fit, errs_fit, approx_wl, fit_type, p=cont_fit_params)
        plt.close()

        # if good fit, apply and break:
        if good_fit:
            flux_fit -= cont_fit_params[0]*wl_fit + cont_fit_params[1]
            continuum_normalised = True
            break
        print('-> [profit]: re-running continuum normalisation...\n')

    if profit.options['open'] == True: profit.utils.ClosePlot()

    # run fitting
    while not accepted:

        print(f'-> [profit]: running {fit_type} fitting of {profit.Line.id}...')

        # show graph if manual
        if mode == 'manual': 
            profit.plots.InitialPlot(wl_fit, flux_fit, errs_fit, approx_wl, fit_type)
            profit.options['open'] = True

        if mode == 'auto':
            print('-> [profit]: Automated codes not fully implemented.')
            sys.exit()

        if fit_type == 'single':
            cen = profit.Line.wls[0]
            if mode == 'manual':
                params, accepted = profit.manual.GaussianFit(wl=wl_fit, flux=flux_fit, errs=errs_fit,
                                                   cwl=cen, z=redshift, zerr=zerr)
            elif mode == 'auto': # TODO: check this is up to date
                params, accepted = profit.auto.GaussianFit(wl=wl_fit, flux=flux_fit, errs=errs_fit,
                                                   cwl=cen, z=redshift, zerr=zerr)
        if fit_type == 'double':
            cen = profit.Line.wls
            if mode == 'manual': 
                params, accepted = profit.manual.DoubleGaussianFit(wl=wl_fit, flux=flux_fit, errs=errs_fit,
                                                   wl1=cen[0], wl2=cen[1], z=redshift, zerr=zerr)
            elif mode == 'auto': # TODO: check this is up to date
                params, accepted = profit.auto.DoubleGaussianFit(wl=wl_fit, flux=flux_fit, errs=errs_fit,
                                                   wl1=cen[0], wl2=cen[1], z=redshift, zerr=zerr)
                
        if fit_type == 'triple':
            cen = profit.Line.wls
            if mode == 'manual': 
                params, accepted = profit.manual.TripleGaussianFit(wl=wl_fit, flux=flux_fit, errs=errs_fit,
                                                   wl1=cen[0], wl2=cen[1], wl3=cen[2], z=redshift, zerr=zerr)
            elif mode == 'auto': # TODO: check this is up to date
                params, accepted = profit.auto.TripleGaussianFit(wl=wl_fit, flux=flux_fit, errs=errs_fit,
                                                   wl1=cen[0], wl2=cen[1], wl3=cen[2], z=redshift, zerr=zerr)

        if fit_type == 'lorentzian':
            cen = profit.Line.wls[0]
            if mode == 'manual':
                params, accepted = profit.manual.LorentzianFit(wl=wl_fit, flux=flux_fit, errs=errs_fit,
                                                   cwl=cen, z=redshift, zerr=zerr)
            elif mode == 'auto': # TODO: check this is up to date
                params, accepted = profit.auto.LorentzianFit(wl=wl_fit, flux=flux_fit, errs=errs_fit,
                                                   cwl=cen, z=redshift, zerr=zerr)
        if fit_type == 'stacked':
            cen = profit.Line.wls[0]
            if mode == 'manual': 
                params, accepted = profit.manual.StackedGaussianFit(wl=wl_fit, flux=flux_fit, errs=errs_fit,
                                                    cwl=cen, z=redshift, zerr=zerr)
            elif mode == 'auto': # TODO: check this is up to date
                params, accepted = profit.auto.StackedGaussianFit(wl=wl_fit, flux=flux_fit, errs=errs_fit,
                                                    cwl=cen, z=redshift, zerr=zerr)  

    # create final dictionary
    fit_params = {}
    singlet_params = {}

    if continuum_normalised:
        fit_params['normalisation_params'] = cont_fit_params
        print(fit_params)

    if fit_type == 'single':

        # general info
        singlet_params['type']        = fit_type
        singlet_params['success']     = params['success']
        if params['success']:
            singlet_params['comment'] = 'Single Gaussian Fit'
        else:
            singlet_params['comment'] = 'Failed fit'
        # data
        singlet_params['z']           = params['z'][0]
        singlet_params['z_err']       = max(params['z'][1], params['z'][2])
        singlet_params['log_amp']     = params['amp'][0]
        singlet_params['log_amp_err'] = max(params['amp'][1], params['amp'][2])
        singlet_params['sig']         = params['sig'][0]
        singlet_params['sig_err']     = max(params['sig'][1], params['sig'][2])
        singlet_params['fwhm']        = params['fwhm'][0]
        singlet_params['flux']        = params['flux'][0]
        singlet_params['flux_err']    = max(params['flux'][1], params['flux'][2])
        singlet_params['fwhm']        = params['fwhm'][0]
        singlet_params['fwhm_err']    = max(params['fwhm'][1], params['fwhm'][2])
        singlet_params['snr']         = singlet_params['flux'] / singlet_params['flux_err']

        fit_params[profit.Line.components[0]] = singlet_params

    elif fit_type == 'double':

        for _n, n in enumerate(np.arange(profit.Line.nprofiles) + 1):

            # general info
            singlet_params['type']          = fit_type
            singlet_params['success']       = params['success']
            if params['success']:
                singlet_params['comment']   = 'Double Gaussian Fit'
            else:
                singlet_params['comment']   = 'Failed fit'
            singlet_params['line']          = profit.Line.id
            # data
            singlet_params['z']            = params['z'][0]
            singlet_params['z_err']        = max(params['z'][1], params['z'][2])
            singlet_params['log_amp']      = params[f'amp{n}'][0]
            singlet_params['log_amp_err']  = max(params[f'amp{n}'][1], params[f'amp{n}'][2])
            singlet_params['sig']          = params['sig'][0]
            singlet_params['sig_err']      = max(params['sig'][1], params['sig'][2])
            singlet_params['flux']         = params[f'flux_{n}'][0]
            singlet_params['flux_err']     = max(params[f'flux_{n}'][1], params[f'flux_{n}'][2])            
            singlet_params['fwhm']         = params['fwhm'][0]
            singlet_params['fwhm_err']     = max(params['fwhm'][1], params['fwhm'][2])
            singlet_params['snr']          = singlet_params['flux'] / singlet_params['flux_err']

            # add singlet param dict to the final 
            fit_params[profit.Line.components[_n]] = singlet_params

    elif fit_type == 'triple':

        for _n, n in enumerate(np.arange(profit.Line.nprofiles) + 1):

            # general info
            singlet_params['type']          = fit_type
            singlet_params['success']       = params['success']
            if params['success']:
                singlet_params['comment']   = 'Triple Gaussian Fit'
            else:
                singlet_params['comment']   = 'Failed fit'
            singlet_params['line']          = profit.Line.id
            # data
            singlet_params['z']            = params['z'][0]
            singlet_params['z_err']        = max(params['z'][1], params['z'][2])
            singlet_params['log_amp']      = params[f'amp{n}'][0]
            singlet_params['log_amp_err']  = max(params[f'amp{n}'][1], params[f'amp{n}'][2])
            singlet_params['sig']          = params['sig'][0]
            singlet_params['sig_err']      = max(params['sig'][1], params['sig'][2])
            singlet_params['flux']         = params[f'flux_{n}'][0]
            singlet_params['flux_err']     = max(params[f'flux_{n}'][1], params[f'flux_{n}'][2])            
            singlet_params['fwhm']         = params['fwhm'][0]
            singlet_params['fwhm_err']     = max(params['fwhm'][1], params['fwhm'][2])
            singlet_params['snr']          = singlet_params['flux'] / singlet_params['flux_err']

            # add singlet param dict to the final 
            fit_params[profit.Line.components[_n]] = singlet_params

    elif fit_type == 'lorentzian':

        # general info
        singlet_params['type']        = fit_type
        singlet_params['success']     = params['success']
        if params['success']:
            singlet_params['comment'] = 'Lorentzian Fit'
        else:
            singlet_params['comment'] = 'Failed fit'
        singlet_params['line']        = profit.Line.id
        # data
        singlet_params['z']           = params['z'][0]
        singlet_params['z_err']       = max(params['z'][1], params['z'][2])
        singlet_params['amp']         = params['amp'][0]
        singlet_params['amp_err']     = max(params['amp'][1], params['amp'][2])
        singlet_params['gamma']       = params['gamma'][0]
        singlet_params['gamma_err']   = max(params['gamma'][1], params['gamma'][2])
        singlet_params['flux']        = params['flux'][0]
        singlet_params['flux_err']    = max(params['flux'][1], params['flux'][2])
        singlet_params['fwhm']        = params['fwhm'][0]
        singlet_params['fwhm_err']    = max(params['fwhm'][1], params['fwhm'][2])
        singlet_params['snr']         = singlet_params['flux'] / singlet_params['flux_err']

        # add to final params
        fit_params[profit.Line.components[0]] = singlet_params

    elif fit_type == 'stacked':

        # general info
        singlet_params['type']        = fit_type
        singlet_params['success']     = params['success']
        if params['success']:
            singlet_params['comment'] = 'Stacked Gaussian Fit'
        else:
            singlet_params['comment'] = 'Failed fit'
        singlet_params['line']        = profit.Line.id
        # data
        singlet_params['z']           = params['z'][0]
        singlet_params['z_err']       = max(params['z'][1], params['z'][2])
        singlet_params['amp_n']       = params['amp_n'][0]
        singlet_params['amp_n_err']   = max(params['amp_n'][1], params['amp_n'][2])
        singlet_params['amp_b']       = params['amp_b'][0]
        singlet_params['amp_b_err']   = max(params['amp_b'][1], params['amp_b'][2])
        singlet_params['sig_n']       = params['sig_n'][0]
        singlet_params['sig_n_err']   = max(params['sig_n'][1], params['sig_n'][2])
        singlet_params['sig_b']       = params['sig_b'][0]
        singlet_params['sig_b_err']   = max(params['sig_b'][1], params['sig_b'][2])
        singlet_params['vel_n']       = params['vel_n'][0]
        singlet_params['vel_n_err']   = max(params['vel_n'][1], params['vel_n'][2])
        singlet_params['vel_b']       = params['vel_b'][0]
        singlet_params['vel_b_err']   = max(params['vel_b'][1], params['vel_b'][2])
        singlet_params['fwhm_n']      = params['fwhm_n'][0]
        singlet_params['fwhm_n_err']  = max(params['fwhm_n'][1], params['fwhm_n'][2])
        singlet_params['fwhm_b']      = params['fwhm_b'][0]
        singlet_params['fwhm_b_err']  = max(params['fwhm_b'][1], params['fwhm_b'][2])
        singlet_params['flux_n']      = params['flux_n'][0]
        singlet_params['flux_n_err']  = max(params['flux_n'][1], params['flux_b'][2])
        singlet_params['flux_b']      = params['flux_b'][0]
        singlet_params['flux_b_err']  = max(params['flux_b'][1], params['flux_b'][2])
        
        # handle combination
        narrow = ufloat(fit_params['flux_n'], fit_params['flux_n_err'])
        broad = ufloat(fit_params['flux_b'], fit_params['flux_b_err'])
        singlet_params['flux']        = (narrow + broad).n
        singlet_params['flux_err']    = (narrow + broad).s
        singlet_params['snr']         = singlet_params['flux'] / singlet_params['flux_err']
        
        # add to final params
        fit_params[profit.Line.components[0]] = singlet_params
    
    elif fit_type == 'limit':
        # general info
        singlet_params['type']        = fit_type
        singlet_params['success']     = True
        singlet_params['comment']     = 'successful limit'
        singlet_params['line']        = profit.Line.id
        # data
        singlet_params['two_sigma_lim'] = params['two_sigma_lim']
        singlet_params['z']        = -999
        singlet_params['z_err']    = -999
        singlet_params['log_amp']  = -999
        singlet_params['sig']      = -999
        singlet_params['fwhm']     = -999
        singlet_params['flux']     = -999
        singlet_params['flux_err'] = -999
        singlet_params['fwhm']     = -999
        singlet_params['fwhm_err'] = -999
        singlet_params['snr']      = -999      

        # add to final params
        fit_params[profit.Line.components[0]] = singlet_params 

    elif fit_type == 'SKIP':

        for _n, n in enumerate(np.arange(profit.Line.nprofiles) + 1):

            # general info
            singlet_params['type']     = fit_type
            singlet_params['success']  = False
            singlet_params['comment']  = 'Skipped'
            # data
            singlet_params['z']        = -999
            singlet_params['z_err']    = -999
            singlet_params['log_amp']  = -999
            singlet_params['sig']      = -999
            singlet_params['fwhm']     = -999
            singlet_params['flux']     = -999
            singlet_params['flux_err'] = -999
            singlet_params['fwhm']     = -999
            singlet_params['fwhm_err'] = -999
            singlet_params['snr']      = -999
    
            # add to final params
            fit_params[profit.Line.components[_n]] = singlet_params

    # save pickle file
    if len(profit.options['rslt_dir']) < 1:
        print(f'-> [profit]: saving fit data to {outpath}\n')
        profit.utils.PklSave(filepath=f'{outpath}/{name}.pkl', results=fit_params)
    else:
        print(f'-> [profit]: saving fit data to {profit.options["rslt_dir"]}\n')
        for component in profit.Line.components:
            profit.utils.PklSave(filepath=f'{profit.options["rslt_dir"]}/{name}.{component}.pkl', results=fit_params[component])

    return None