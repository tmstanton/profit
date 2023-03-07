# relevant imports
import profit
from profit import utils, plots
import numpy as np
from scipy import stats
from scipy.stats import linregress
import matplotlib.pyplot as plt
from astropy.io import fits
import dynesty
from dynesty.utils import quantile as _quantile
from dynesty import utils as dyfunc
from typing import Tuple as tpl
import sys

# -=-=-=- methods -=-=-=-

def DoubleGaussianFit(wl:object, flux:object, errs:object, wl1:float, wl2:float, z:float, zerr=0.1) -> tpl[dict, bool]:
    pass

def GaussianFit(wl:list, flux:list, errs:list, cwl:float, z:float, zerr:float=0.05) -> None: 
    
    # select maxmimum flux within the window
    amp_lims = [0, np.log10(np.max(flux))]
    sig_lims = [0, 20]
    rst_lims = [z-zerr, z+zerr]

    if profit.options['open']: utils.ClosePlot()

    # define log likelihood and prior transform functions
    def logl_single(u):
        
        # extract parameters
        log_amp, sig, rs = u
        # determine log likelihood
        if sig <= 0:
            return -np.inf
        model = utils.Gaussian(x=wl, amp=10**log_amp, xc=cwl*(1. + rs), sig=sig)
        return -0.5 * np.sum((np.power((model - flux) / errs, 2) + 2 * np.log(errs) ))

    def ptform_single(p): 

        # unpack prior parameters
        plog_amp, psig, pz = p
        # uniform priors
        log_amp = amp_lims[0] + plog_amp * (amp_lims[1] - amp_lims[0])
        sig     = sig_lims[0] + psig     * (sig_lims[1] - sig_lims[0])
        red     = rst_lims[0] + pz       * (rst_lims[1] - rst_lims[0]) 

        return log_amp, sig, red 

    # run sampling
    sampler = dynesty.NestedSampler(loglikelihood=logl_single, prior_transform=ptform_single,
                                    ndim=3)
    sampler.run_nested(print_progress=profit.options['verbose'])

    # extract results 
    res = sampler.results
    samples, weights = res['samples'], np.exp(res.logwt - res.logz[-1])
    samples_equal = dyfunc.resample_equal(samples, weights)

    # add data to dictionary
    keys = ['amp', 'sig', 'z']
    params = {}
    for p, par in enumerate(samples.T):
        params[keys[p]] = utils.ExtractParamValues(par, weights=weights)

    # plot best fit model
    plots.BestFitPlot(wl=wl, fluxes=flux, errors=errs, params=params, cen=(cwl), mode='single')
    # plot corner plot
    plots.CornerPlot(results=res, mode='single')

    # use samples_equal to get fluxes under gaussians, and fwhm 
    fluxes, fwhms = np.empty(samples_equal.size), np.empty(samples_equal.size)

    for s, samp in enumerate(samples_equal):
        fluxes[s] = utils.FluxUnderGaussian(amp=np.power(10,samp[0]), sig=samp[1])
        fwhms[s]  = utils.FWHM(samp[1]) 

    # generate distributions in these parameters
    genkeys = ['flux', 'fwhm']
    for gp, genparam in enumerate(genkeys):
        extracted_values = utils.ExtractParamValues(genparam, weights=None)
        params[genkeys[gp]] = extracted_values

    flux_vals = _quantile(fluxes, [0.16, 0.50, 0.84], weights=None)
    fwhm_vals = _quantile(fwhms, [0.16, 0.50, 0.84], weights=None)

    # add to params
    params['flux'] = (flux_vals[1], flux_vals[1]-flux_vals[0], flux_vals[2]-flux_vals[1])
    params['fwhm'] = (fwhm_vals[1], fwhm_vals[1]-fwhm_vals[0], fwhm_vals[2]-fwhm_vals[1])

    # check to see if happy
    return utils.ValidatePlot(params)

def LorentzianFit(wl:object, flux:object, errs:object, cwl:float, z:float, zerr:float=0.05) -> tpl[dict, bool]:

    # take in parameter guesses (visual)
    amp_lims   = [0, np.log10(np.max(flux))]
    gamma_lims = [0, 30] 
    rst_lims   = [z-zerr, z+zerr]

    if profit.options['open']: utils.ClosePlot()

    # define log likelihood and prior transform functions
    def logl_lor(u):
        
        # extract parameters
        log_amp, gma, rs = u
        # determine log likelihood
        if gma <= 0:
            return -np.inf
        model = utils.Lorentzian(x=wl, amp=10**log_amp, gamma=gma, xc = cwl*(1. + rs))
        return -0.5 * np.sum((np.power((model[errs>0] - flux[errs>0]) / errs[errs>0], 2) + 2 * np.log(errs[errs>0]) ))

    def ptform_lor(p): 

        # unpack prior parameters
        plog_amp, pgamma, pz = p
        log_amp = amp_lims[0]   + plog_amp * (amp_lims[1]   - amp_lims[0])
        gam     = gamma_lims[0] + pgamma   * (gamma_lims[1] - gamma_lims[0])
        red     = rst_lims[0]   + pz       * (rst_lims[1]   - rst_lims[0])

        return log_amp, gam, red 

    # run sampling
    sampler = dynesty.NestedSampler(loglikelihood=logl_lor, prior_transform=ptform_lor,
                                    ndim=3)
    sampler.run_nested(print_progress=profit.options['verbose'])

    # extract results 
    res = sampler.results
    samples, weights = res['samples'], np.exp(res.logwt - res.logz[-1])
    samples_equal = dyfunc.resample_equal(samples, weights)

    # add data to dictionary
    keys = ['amp', 'gamma', 'z']
    params = {}
    for p, par in enumerate(samples.T):
        params[keys[p]] = utils.ExtractParamValues(par, weights=weights)

    # plot best fit model
    plots.BestFitPlot(wl=wl, fluxes=flux, errors=errs, params=params, cen=(cwl), mode='lorentzian')
    # plot corner plot
    plots.CornerPlot(results=res, mode='lorentzian')

    # use samples_equal to get fluxes under gaussians, and fwhm 
    fluxes, fwhms = np.empty(samples_equal.size), np.empty(samples_equal.size)

    for s, samp in enumerate(samples_equal):
        fluxes[s] = utils.FluxUnderLorentzian(amp=np.power(10,samp[0]), sig=samp[1])
        fwhms[s]  = 2 * samp[1] # FWHM is 2 * gamma

    # generate distributions in these parameters
    genkeys = ['flux', 'fwhm']
    flux_fwhm = [fluxes, fwhms]
    for gp, genparam in enumerate(genkeys):
        extracted_values = utils.ExtractParamValues(flux_fwhm[gp], weights=None)
        params[genkeys[gp]] = extracted_values

    # check to see if happy
    return utils.ValidatePlot(params)

def StackedGaussianFit(wl:object, flux:object, errs:object, cwl:float, z:float, zerr:float=0.05) -> tpl[dict, bool]:

    # automated guesses
    amp_lims   = [0, np.log10(np.max(flux))]
    vel_n_lims = [0, profit.options['vel_barrier']]
    vel_b_lims = [profit.options['vel_barrier'], 2*profit.options['vel_barrier']]
    rst_lims   = [z-zerr, z+zerr]

    if profit.options['open']: utils.ClosePlot()

    # define log likelihood and prior transform functions
    def logl_stacked(u:tuple) -> tuple:
        
        # extract parameters
        log_amp_n, log_amp_b, vel_n, vel_b, rs = u
        sig_n = utils.Vel_To_Sigma(vel_n)
        sig_b = utils.Vel_To_Sigma(vel_b)

        # determine log likelihood
        if min(sig_n, sig_b) <= 0:
            return -np.inf
        model = utils.StackedGaussian(x=wl, xc=cwl*(1+rs), amp1=10**log_amp_n, 
                                      amp2=10**log_amp_b, sig1=sig_n, sig2=sig_b)
        return -0.5 * np.sum((np.power((model - flux) / errs, 2) + 2 * np.log(errs) ))

    def ptform_stacked(p:tuple) -> tuple: 

        # unpack prior parameters
        plog_amp_n, plog_amp_b, pvel_n, pvel_b, pz = p

        # use stats.norm to define the parameter estimates around the guesses
        log_amp_n = amp_lims[0] + plog_amp_n * (amp_lims[1] - amp_lims[0])
        log_amp_b = amp_lims[0] + plog_amp_b * (amp_lims[1] - amp_lims[0]) / 2

        # generate limits? (0 to barrier value, barrier value to maximum possible width from guess value?)
        vel_n = vel_n_lims[0] + pvel_n * (vel_n_lims[1] - vel_n_lims[0])
        vel_b = vel_b_lims[0] + pvel_b * (vel_b_lims[1] - vel_b_lims[0])

        # redshift
        red = rst_lims[0] + pz * (rst_lims[1] - rst_lims[0])

        return log_amp_n, log_amp_b, vel_n, vel_b, red 

    # run sampling
    sampler = dynesty.NestedSampler(loglikelihood=logl_stacked, prior_transform=ptform_stacked,
                                    ndim=5)
    sampler.run_nested(print_progress=profit.options['verbose'])

    # extract results 
    res = sampler.results
    samples, weights = res['samples'], np.exp(res.logwt - res.logz[-1])
    samples_equal = dyfunc.resample_equal(samples, weights)

    # add data to dictionary
    keys = ['amp_n', 'amp_b', 'vel_n', 'vel_b', 'z']
    params = {}
    for p, par in enumerate(samples.T):
        params[keys[p]] = utils.ExtractParamValues(par, weights=weights)

    if profit.options['display_auto_plots']:
        # plot best fit model
        plots.BestFitPlot(wl=wl, fluxes=flux, errors=errs, params=params, cen=(cwl), mode='stacked')
        # plot corner plot
        plots.CornerPlot(results=res, mode='stacked')

    # TODO: would need to think about how best to handle this, as fwhm and other values for this would likely be useful. 
    """
    # use samples_equal to get fluxes under gaussians, and fwhm 
    fluxes_n, fwhms_n = np.empty(samples_equal.size), np.empty(samples_equal.size)

    for s, samp in enumerate(samples_equal):
        fluxes_n[s] = utils.FluxUnderGaussian(amp=np.power(10,samp[0]), sig=samp[1])
        fwhms_n[s]  = utils.FWHM(samp[1]) 

    # generate distributions in these parameters
    genkeys = ['flux', 'fwhm']
    flux_fwhm = [fluxes_n, fwhms_n]
    for gp, genparam in enumerate(genkeys):
        extracted_values = utils.ExtractParamValues(flux_fwhm[gp], weights=None)
        params[genkeys[gp]] = extracted_values
    """
    # check to see if happy
    return utils.ValidatePlot(params)