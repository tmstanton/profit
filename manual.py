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

# -=-=-=- Fitting Methods -=-=-=-

def DoubleGaussianFit(wl:list, flux:list, errs:list, wl1:float, wl2:float, z:float, zerr=0.1) -> tpl[dict, bool]:
    
    # take in parameter guesses (visual)
    a1_guess = utils.UserInput('Amplitude (L)') #float(input('Width Guess: '))
    a2_guess = utils.UserInput('Amplitude (R)') #float(input('Width Guess: '))
    sig_guess = utils.UserInput('Width') #float(input('Width Guess: '))

    if profit.options['open']: utils.ClosePlot()

    # define log likelihood and prior transform functions
    def logl_double(u: tuple) -> float:
        
        # unpack parameters
        log_amp1, log_amp2, sig, rs = u

        # determine log likelihood
        if sig <= 0:
            return -np.inf

        # work out two centers
        c1, c2 = wl1 * (1. + rs), wl2 * (1. + rs)
        model = utils.DoubleGaussian(x=wl, amp1=10**log_amp1, xc1=c1, amp2=10**log_amp2, xc2=c2, sig=sig)
        return -0.5 * np.sum((np.power((model - flux) / errs, 2) + 2 * np.log(errs) ))

    def ptform_double(p:tuple) -> tuple:
        
        # unpack values
        plog_amp1, plog_amp2, psig, pz = p

        # generate prior distributions
        log_amp1 = stats.norm.ppf(plog_amp1, loc=np.log10(a1_guess), scale=0.3)
        log_amp2 = stats.norm.ppf(plog_amp2, loc=np.log10(a2_guess), scale=0.3)
        sig = stats.norm.ppf(psig, loc=sig_guess, scale=2*sig_guess)
        #red = stats.norm.ppf(pz, loc=z, scale=0.1)
        red = stats.uniform.ppf(pz, loc=z-zerr, scale=2*zerr)

        return log_amp1, log_amp2, sig, red

    # run sampling
    sampler = dynesty.NestedSampler(loglikelihood=logl_double, prior_transform=ptform_double,
                                    ndim=4)
    sampler.run_nested(print_progress=profit.options['verbose'])

    # extract results 
    res = sampler.results
    samples, weights = res['samples'], np.exp(res.logwt - res.logz[-1])
    samples_equal = dyfunc.resample_equal(samples, weights)
    
    # add data to dictionary
    keys = ['amp1', 'amp2', 'sig', 'z']
    params = {}
    for p, par in enumerate(samples.T):
        params[keys[p]] = utils.ExtractParamValues(par, weights=weights)

    # plot best fit model
    plots.BestFitPlot(wl=wl, fluxes=flux, errors=errs, params=params, cen=(wl1, wl2), mode='double')
    # plot corner plot
    plots.CornerPlot(results=res, mode='double')

    # use samples_equal to get fluxes under gaussians, and fwhm 
    fluxes1, fluxes2, fwhms = np.empty(samples_equal.shape[0]), np.empty(samples_equal.shape[0]), np.empty(samples_equal.shape[0])

    for s, samp in enumerate(samples_equal):
        fluxes1[s] = utils.FluxUnderGaussian(amp=np.power(10,samp[0]), sig=samp[2])
        fluxes2[s] = utils.FluxUnderGaussian(amp=np.power(10,samp[1]), sig=samp[2])
        fwhms[s]   = 2. * np.sqrt(2. * np.log(2.)) * samp[2] 

    # generate distributions in these parameters
    params['flux_1'] = utils.ExtractUnweightedParams(fluxes1)
    params['flux_2'] = utils.ExtractUnweightedParams(fluxes2)
    params['fwhm']   = utils.ExtractUnweightedParams(fwhms)

    # check to see if happy
    return utils.ValidatePlot(params)

def GaussianFit(wl:list, flux:list, errs:list, cwl:float, z:float, zerr:float=0.05) -> tpl[dict, bool]:

    # take in parameter guesses (visual)
    amp_guess = utils.UserInput('Amplitude') #float(input('Amplitude Guess: '))
    sig_guess = utils.UserInput('Width') #float(input('Width Guess: '))

    if profit.options['open']: utils.ClosePlot()

    # define log likelihood and prior transform functions
    def logl_single(u:tuple) -> float:
        
        # extract parameters
        log_amp, sig, rs = u
        # determine log likelihood
        if sig <= 0:
            return -np.inf
        model = utils.Gaussian(x=wl, amp=10**log_amp, xc=cwl*(1. + rs), sig=sig)
        return -0.5 * np.sum((np.power((model - flux) / errs, 2) + 2 * np.log(errs) ))

    def ptform_single(p:tuple) -> tuple: 

        # unpack prior parameters
        plog_amp, psig, pz = p
        # use stats.norm to define the parameter estimates around the guesses
        log_amp = stats.norm.ppf(plog_amp, loc=np.log10(amp_guess), scale=0.05)
        sig = stats.norm.ppf(psig, loc=sig_guess, scale=2*sig_guess)
        red = stats.uniform.ppf(pz, loc=z-zerr, scale=2*zerr)

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

    if profit.options['open']: utils.ClosePlot()

    # plot best fit model
    plots.BestFitPlot(wl=wl, fluxes=flux, errors=errs, params=params, cen=(cwl), mode='single')
    # plot corner plot
    plots.CornerPlot(results=res, mode='single')

    # use samples_equal to get fluxes under gaussians, and fwhm 
    fluxes, fwhms = np.empty(samples_equal.shape[0]), np.empty(samples_equal.shape[0])

    for s, samp in enumerate(samples_equal):
        fluxes[s] = utils.FluxUnderGaussian(amp=np.power(10,samp[0]), sig=samp[1])
        fwhms[s]  = 2. * np.sqrt(2. * np.log(2.)) * samp[1] 

    params['flux'] = utils.ExtractUnweightedParams(fluxes)
    params['fwhm'] = utils.ExtractUnweightedParams(fwhms)
    
    # check to see if happy
    return utils.ValidatePlot(params)

def LorentzianFit(wl:list, flux:list, errs:list, cwl:float, z:float, zerr:float=0.05) -> tpl[dict, bool]:

    # take in parameter guesses (visual)
    amp_guess = float(input('Amplitude Guess: '))
    gamma_guess = float(input('FWHM Guess: '))

    if profit.options['open']: utils.ClosePlot()

    # define log likelihood and prior transform functions
    def logl_lor(u:tuple) -> float:
        
        # extract parameters
        log_amp, gma, rs = u
        # determine log likelihood
        if gma <= 0: # or gma >= 10000:
            return -np.inf
        model = utils.Lorentzian(x=wl, amp=10**log_amp, gamma=gma, xc = cwl*(1. + rs))
        return -0.5 * np.sum((np.power((model - flux) / errs, 2) + 2 * np.log(errs) ))

    def ptform_lor(p:tuple) -> float: 

        # unpack prior parameters
        plog_amp, pgamma, pz = p
        log_amp = stats.norm.ppf(plog_amp, loc=np.log10(amp_guess), scale=0.3)
        gam = stats.norm.ppf(pgamma, loc=gamma_guess, scale=1)
        #gam = pgamma * 1.5 * gamma_guess
        red = stats.uniform.ppf(pz, loc=z-zerr, scale=2*zerr)

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
    fluxes, fwhms = np.empty(samples_equal.shape[0]), np.empty(samples_equal.shape[0])

    for s, samp in enumerate(samples_equal):
        fluxes[s] = utils.FluxUnderLorentzian(amp=np.power(10,samp[0]), gam=samp[1])
        fwhms[s] = 2 * samp[1]

    # generate distributions in these parameters
    genkeys = ['flux', 'fwhm']
    flux_fwhm = [fluxes, fwhms]
    for gp, genparam in enumerate(genkeys):
        extracted_values = utils.ExtractParamValues(flux_fwhm[gp], weights=None)
        params[genparam] = extracted_values

    # check to see if happy
    return utils.ValidatePlot(params)

def StackedGaussianFit(wl:list, flux:list, errs:list, cwl:float, z:float, zerr:float=0.05) -> tpl[dict, bool]:
    
    # take in parameter guesses (visual) (for narrow peak? and maybe width for wide peak? (e.g. max to set upper limit?))
    amp_guess = float(input('Amplitude Guess: '))
    #sig_guess = float(input('Width Guess: '))

    if profit.options['open']: utils.ClosePlot()

    # convert to velocity and set velocity limit
    vel_n_lims = [0, profit.options['vel_barrier']]
    vel_b_lims = [profit.options['vel_barrier'], 5 * profit.options['vel_barrier']]

    # amplitude limits
    log_amp_guess = np.log10(amp_guess)
    log_amp_n_lims = [0.5 * log_amp_guess, log_amp_guess]
    log_amp_b_lims = [0, 0.5 * log_amp_guess]

    # define log likelihood and prior transform functions
    def logl_stacked(u:tuple) -> float:
        
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
        log_amp_n = log_amp_n_lims[0] + plog_amp_n * (log_amp_n_lims[0] - log_amp_n_lims[-1])
        log_amp_b = log_amp_b_lims[0] + plog_amp_b * (log_amp_b_lims[0] - log_amp_b_lims[-1])

        # generate limits? 
        vel_n = vel_n_lims[0] + pvel_n * (vel_n_lims[0] - vel_n_lims[-1])
        vel_b = vel_b_lims[0] + pvel_b * (vel_b_lims[0] - vel_b_lims[-1])

        # redshift
        red = stats.uniform.ppf(pz, loc=z-zerr, scale=z+zerr)

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
    keys = ['amp_n', 'amp_b', 'sig_n', 'sig_b', 'z']
    params = {}
    for p, par in enumerate(samples.T):
        params[keys[p]] = utils.ExtractParamValues(par, weights=weights)

    # plot best fit model
    plots.BestFitPlot(wl=wl, fluxes=flux, errors=errs, params=params, cen=(cwl), mode='stacked')
    # plot corner plot
    plots.CornerPlot(results=res, mode='stacked')

    # TODO: would need to think about how best to handle this, as fwhm and other values for this would likely be useful. 
    """
    # use samples_equal to get fluxes under gaussians, and fwhm 
    fluxes_n, fwhms_n = np.empty(samples_equal.shape[0]), np.empty(samples_equal.shape[0])

    for s, samp in enumerate(samples_equal):
        fluxes_n[s] = utils.FluxUnderGaussian(amp=np.power(10,samp[0]), sig=samp[1])
        fwhms_n[s]  = 2. * np.sqrt(2. * np.log(2.)) * samp[1] 

    # generate distributions in these parameters
    genkeys = ['flux', 'fwhm']
    flux_fwhm = [fluxes_n, fwhms_n]
    for gp, genparam in enumerate(genkeys):
        extracted_values = utils.ExtractParamValues(flux_fwhm[gp], weights=None)
        params[genkeys[gp]] = extracted_values
    """
    # check to see if happy
    return utils.ValidatePlot(params)

def TwoSigmaLimit(wl:list, flux:list, err:list, cwl:float, z:float) -> tpl[dict, bool]:
    # a 2 sigma limit if the full line cannot be resolved)
    # TODO: figure this out 
    pass

