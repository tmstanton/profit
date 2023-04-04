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
from multiprocessing import Pool
from dynesty import plotting as dyplot

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
        sig = stats.norm.ppf(psig, loc=sig_guess, scale=4) # changed from 2 * sig guess
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

    # extract derived parameters
    genkeys = ['flux_1', 'flux_2', 'fwhm']
    flux_fwhm = [fluxes1, fluxes2, fwhms]
    for key, array in zip(genkeys, flux_fwhm):
        params[key] = utils.ExtractParamValues(array, weights=None)

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
        if sig >= 5 * sig_guess:
            return -np.inf
        model = utils.Gaussian(x=wl, amp=10**log_amp, xc=cwl*(1. + rs), sig=sig)
        return -0.5 * np.sum((np.power((model - flux) / errs, 2) + 2 * np.log(errs) ))

    def ptform_single(p:tuple) -> tuple: 

        # unpack prior parameters
        plog_amp, psig, pz = p
        # use stats.norm to define the parameter estimates around the guesses
        log_amp = stats.norm.ppf(plog_amp, loc=np.log10(amp_guess), scale=0.05)
        sig = stats.norm.ppf(psig, loc=sig_guess, scale=5)
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
    
    genkeys = ['flux', 'fwhm']
    flux_fwhm = [fluxes, fwhms]
    for key, array in zip(genkeys, flux_fwhm):
        params[key] = utils.ExtractParamValues(array, weights=None)

    # check to see if happy
    return utils.ValidatePlot(params)

def LorentzianFit(wl:list, flux:list, errs:list, cwl:float, z:float, zerr:float=0.05) -> tpl[dict, bool]:

    # take in parameter guesses (visual)
    amp_guess = utils.UserInput('Amplitude') 
    gamma_guess = utils.UserInput('FWHM')

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
    for key, array in zip(genkeys, flux_fwhm):
        params[key] = utils.ExtractParamValues(array, weights=None)
    """
    for gp, genparam in enumerate(genkeys):
        extracted_values = utils.ExtractParamValues(flux_fwhm[gp], weights=None)
        params[genparam] = extracted_values
    """
    # check to see if happy
    return utils.ValidatePlot(params)

def StackedGaussianFit(wl:list, flux:list, errs:list, cwl:float, z:float, zerr:float=0.05) -> tpl[dict, bool]:

    def GaussianStack(x:np.ndarray, namp:float, bamp:float, nsig:float, bsig:float, xc:float) -> np.ndarray:
        narrow = Gaussian(x, namp, nsig, xc)
        broad  = Gaussian(x, bamp, bsig, xc)
        return broad + narrow

    def Gaussian(x:np.ndarray, amp:float, sig:float, xc:float) -> np.ndarray:
        return amp * np.exp(-np.power(x-xc, 2) / (2 * np.power(sig, 2)))

    def InvertFWHM(fwhm:float) -> float:
        return fwhm / (2. * np.sqrt(2 * np.log(2.)))

    def Sigma(vel:float, cwl:float, z:float) -> float:
        fwhm = (cwl * (1. + z)) * vel / 3e5
        return InvertFWHM(fwhm)

    # define log likelihood function
    def logL(u:tuple) -> float:
        
        # unpack parameters
        namp, bamp, nvel, bvel, z = u
        # convert vels to sigma
        nsig = Sigma(nvel, cwl, z)
        bsig = Sigma(bvel, cwl, z)

        # make model flux
        if min(nsig, bsig) <= 0:
            return -np.inf
        model = GaussianStack(wl, namp, bamp, nsig, bsig, cwl * (1. + z))
        return -0.5 * np.sum((np.power((model - flux) / errs, 2) + 2 * np.log(errs) ))

    def unpack(lims:list, par:float) -> object:
        return lims[0] + (par * (lims[1] - lims[0]))

    # define prior transform function
    def ptform(p:tuple) -> float:
        
        # unpack parameters
        pnamp, pbamp, pnvel, pbvel, pz = p

        # determine ptform
        red = unpack(redshift_lims, pz)
        nvel = unpack(narrow_vel_lims, pnvel)
        bvel = unpack(broad_vel_lims,  pbvel)
        namp = unpack(narrow_amp_lims, pnamp)
        bamp = unpack(broad_amp_lims,  pbamp)

        return namp, bamp, nvel, bvel, red
    
    # run dynesty fitting
    amp_guess = utils.UserInput('Amplitude')
    broad_amp_lims  = [0, 0.5 * amp_guess]
    narrow_amp_lims = [0.5 * amp_guess, 2.5 * amp_guess]
    narrow_vel_lims  = [profit.options['vel_barrier']/4, 1.75 * profit.options['vel_barrier']] 
    broad_vel_lims = [profit.options['vel_barrier'], 5*profit.options['vel_barrier']]
    redshift_lims = [z - 0.05, z + 0.05]

    # close plot
    if profit.options['open']: utils.ClosePlot()

    # run fitting
    sampler = dynesty.NestedSampler(loglikelihood=logL, prior_transform=ptform,
                                    ndim=5) #, pool=Pool(), queue_size=5)
    sampler.run_nested(print_progress=profit.options['verbose'])

    # extract results 
    res = sampler.results
    samples, weights = res['samples'], np.exp(res.logwt - res.logz[-1])
    samples_equal = dyfunc.resample_equal(samples, weights)

    # log amplitude samples
    for amp in range(2):
        samples.T[amp]       = np.log10(samples.T[amp])
        samples_equal.T[amp] = np.log10(samples_equal.T[amp])

    # add data to dictionary
    keys = ['amp_n', 'amp_b', 'vel_n', 'vel_b', 'z']
    params = {}
    graph_params = []
    for p, par in enumerate(samples.T):
        params[keys[p]] = utils.ExtractParamValues(par, weights=weights)
        graph_params.append(params[keys[p]][0])
        print(keys[p], utils.ExtractParamValues(par, weights=weights))

    # generate samples in terms of sig
    sig_n_samples = Sigma(samples.T[2], cwl, samples.T[4])
    sig_b_samples = Sigma(samples.T[3], cwl, samples.T[4])
    params['sig_b'] = utils.ExtractParamValues(sig_b_samples, weights=weights)
    params['sig_n'] = utils.ExtractParamValues(sig_n_samples, weights=weights)

    # plot best fit model
    plots.BestFitPlot(wl=wl, fluxes=flux, errors=errs, params=params, cen=(cwl), mode='stacked')
    # plot corner plot
    plots.CornerPlot(results=res, mode='stacked')

    # work out fluxes
    flux_n, flux_b = np.empty(samples_equal.shape[0]), np.empty(samples_equal.shape[0])
    fwhm_n, fwhm_b = np.empty(samples_equal.shape[0]), np.empty(samples_equal.shape[0])
    for s, samp in enumerate(samples_equal):
            flux_n[s] = utils.FluxUnderGaussian(amp=np.power(10,samp[0]), sig=Sigma(samp[2], cwl, samp[-1]))
            flux_b[s] = utils.FluxUnderGaussian(amp=np.power(10,samp[1]), sig=Sigma(samp[3], cwl, samp[-1]))
            fwhm_n[s] = 2. * np.sqrt(2. * np.log(2.)) * Sigma(samp[2], cwl, samp[-1])
            fwhm_b[s] = 2. * np.sqrt(2. * np.log(2.)) * Sigma(samp[3], cwl, samp[-1])


    # generate distributions in these parameters
    genkeys = ['flux_n', 'flux_b', 'fwhm_n', 'fwhm_b']
    flux_fwhm = [flux_n, flux_b, fwhm_n, fwhm_b]
    for key, array in zip(genkeys, flux_fwhm):
        params[key] = utils.ExtractParamValues(array, weights=None)

    return utils.ValidatePlot(params)

def TwoSigmaLimit(wl:list, flux:list, err:list, cwl:float, z:float) -> tpl[dict, bool]:
    # a 2 sigma limit if the full line cannot be resolved)
    # TODO: figure this out 
    pass

