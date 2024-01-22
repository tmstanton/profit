# relevant imports
import profit # type: ignore
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
    a1_guess = profit.utils.UserInput('Amplitude (L)') #float(input('Width Guess: '))
    a2_guess = profit.utils.UserInput('Amplitude (R)') #float(input('Width Guess: '))
    sig_guess = profit.utils.UserInput('Width') #float(input('Width Guess: '))

    if profit.options['open']: profit.utils.ClosePlot()

    # define log likelihood and prior transform functions
    def logl_double(u: tuple) -> float:
        
        # unpack parameters
        log_amp1, log_amp2, sig, rs = u

        # determine log likelihood
        if sig <= 0:
            return -np.inf
        if sig >= 2.5 * sig_guess:
            return -np.inf
        # work out two centers
        c1, c2 = wl1 * (1. + rs), wl2 * (1. + rs)
        model = profit.utils.DoubleGaussian(x=wl, amp1=10**log_amp1, xc1=c1, amp2=10**log_amp2, xc2=c2, sig=sig)
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
        params[keys[p]] = profit.utils.ExtractParamValues(par, weights=weights)

    # plot best fit model
    profit.plots.BestFitPlot(wl=wl, fluxes=flux, errors=errs, params=params, cen=(wl1, wl2), mode='double')
    # plot corner plot
    profit.plots.CornerPlot(results=res, mode='double')

    # use samples_equal to get fluxes under gaussians, and fwhm 
    fluxes1, fluxes2, fwhms = np.empty(samples_equal.shape[0]), np.empty(samples_equal.shape[0]), np.empty(samples_equal.shape[0])

    for s, samp in enumerate(samples_equal):
        fluxes1[s] = profit.utils.FluxUnderGaussian(amp=np.power(10,samp[0]), sig=samp[2])
        fluxes2[s] = profit.utils.FluxUnderGaussian(amp=np.power(10,samp[1]), sig=samp[2])
        fwhms[s]   = 2. * np.sqrt(2. * np.log(2.)) * samp[2] 

    # extract derived parameters
    genkeys = ['flux_1', 'flux_2', 'fwhm']
    flux_fwhm = [fluxes1, fluxes2, fwhms]
    for key, array in zip(genkeys, flux_fwhm):
        params[key] = profit.utils.ExtractParamValues(array, weights=None)

    # check to see if happy
    return profit.utils.ValidatePlot(params)

def TripleGaussianFit(wl:list, flux:list, errs:list, wl1:float, wl2:float, wl3:float, z:float, zerr=0.1) -> tpl[dict, bool]:
    
    # take in parameter guesses (visual)
    a1_guess = profit.utils.UserInput('Amplitude (Left)') #float(input('Width Guess: '))
    a2_guess = profit.utils.UserInput('Amplitude (Centre)') #float(input('Width Guess: '))
    a3_guess = profit.utils.UserInput('Amplitude (Right)') #float(input('Width Guess: '))
    sig_guess = profit.utils.UserInput('Width') #float(input('Width Guess: '))

    if profit.options['open']: profit.utils.ClosePlot()

    # define log likelihood and prior transform functions
    def logl_triple(u: tuple) -> float:
        
        # unpack parameters
        log_amp1, log_amp2, log_amp3, sig, rs = u

        # determine log likelihood
        if sig <= 0:
            return -np.inf
        if sig >= 2.5 * sig_guess:
            return -np.inf
        # work out two centers
        c1, c2, c3 = wl1 * (1. + rs), wl2 * (1. + rs), wl3 * (1. + rs)
        model = profit.utils.TripleGaussian(x=wl, sig=sig,
                                            amp1=10**log_amp1, xc1=c1, 
                                            amp2=10**log_amp2, xc2=c2, 
                                            amp3=10**log_amp3, xc3=c3
                                            )
        return -0.5 * np.sum((np.power((model - flux) / errs, 2) + 2 * np.log(errs) ))

    def ptform_triple(p:tuple) -> tuple:
        
        # unpack values
        plog_amp1, plog_amp2, plog_amp3, psig, pz = p

        # generate prior distributions
        log_amp1 = stats.norm.ppf(plog_amp1, loc=np.log10(a1_guess), scale=0.3)
        log_amp2 = stats.norm.ppf(plog_amp2, loc=np.log10(a2_guess), scale=0.3)
        log_amp3 = stats.norm.ppf(plog_amp3, loc=np.log10(a3_guess), scale=0.3)
        sig = stats.norm.ppf(psig, loc=sig_guess, scale=4) # changed from 2 * sig guess
        red = stats.uniform.ppf(pz, loc=z-zerr, scale=2*zerr)

        return log_amp1, log_amp2, log_amp3, sig, red

    # run sampling
    sampler = dynesty.NestedSampler(loglikelihood=logl_triple, prior_transform=ptform_triple,
                                    ndim=5)
    sampler.run_nested(print_progress=profit.options['verbose'])

    # extract results 
    res = sampler.results
    samples, weights = res['samples'], np.exp(res.logwt - res.logz[-1])
    samples_equal = dyfunc.resample_equal(samples, weights)
    
    # add data to dictionary
    keys = ['amp1', 'amp2', 'amp3', 'sig', 'z']
    params = {}
    for p, par in enumerate(samples.T):
        params[keys[p]] = profit.utils.ExtractParamValues(par, weights=weights)

    # plot best fit model
    profit.plots.BestFitPlot(wl=wl, fluxes=flux, errors=errs, params=params, cen=(wl1, wl2, wl3), mode='triple')
    # plot corner plot
    profit.plots.CornerPlot(results=res, mode='triple')

    # use samples_equal to get fluxes under gaussians, and fwhm 
    fluxes1, fluxes2, fluxes3 = np.empty(samples_equal.shape[0]), np.empty(samples_equal.shape[0]), np.empty(samples_equal.shape[0])
    fwhms = np.empty(samples_equal.shape[0])

    for s, samp in enumerate(samples_equal):
        fluxes1[s] = profit.utils.FluxUnderGaussian(amp=np.power(10,samp[0]), sig=samp[3])
        fluxes2[s] = profit.utils.FluxUnderGaussian(amp=np.power(10,samp[1]), sig=samp[3])
        fluxes3[s] = profit.utils.FluxUnderGaussian(amp=np.power(10,samp[2]), sig=samp[3])
        fwhms[s]   = 2. * np.sqrt(2. * np.log(2.)) * samp[3] 

    # extract derived parameters
    genkeys = ['flux_1', 'flux_2', 'flux_3', 'fwhm']
    flux_fwhm = [fluxes1, fluxes2, fluxes3, fwhms]
    for key, array in zip(genkeys, flux_fwhm):
        params[key] = profit.utils.ExtractParamValues(array, weights=None)

    # check to see if happy
    return profit.utils.ValidatePlot(params)

def GaussianFit(wl:list, flux:list, errs:list, cwl:float, z:float, zerr:float=0.05) -> tpl[dict, bool]:

    # take in parameter guesses (visual)
    amp_guess = profit.utils.UserInput('Amplitude') #float(input('Amplitude Guess: '))
    sig_guess = profit.utils.UserInput('Width') #float(input('Width Guess: '))

    if profit.options['open']: profit.utils.ClosePlot()

    # define log likelihood and prior transform functions
    def logl_single(u:tuple) -> float:
        
        # extract parameters
        log_amp, sig, rs = u
        # determine log likelihood
        if sig <= 0:
            return -np.inf
        if sig >= 2.5 * sig_guess:
            return -np.inf
        if sig <= (sig_guess / 2):
            return -np.inf
        model = profit.utils.Gaussian(x=wl, amp=10**log_amp, xc=cwl*(1. + rs), sig=sig)
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
        params[keys[p]] = profit.utils.ExtractParamValues(par, weights=weights)

    if profit.options['open']: profit.utils.ClosePlot()

    # plot best fit model
    profit.plots.BestFitPlot(wl=wl, fluxes=flux, errors=errs, params=params, cen=(cwl), mode='single')
    # plot corner plot
    profit.plots.CornerPlot(results=res, mode='single')

    # use samples_equal to get fluxes under gaussians, and fwhm 
    fluxes, fwhms = np.empty(samples_equal.shape[0]), np.empty(samples_equal.shape[0])

    for s, samp in enumerate(samples_equal):
        fluxes[s] = profit.utils.FluxUnderGaussian(amp=np.power(10,samp[0]), sig=samp[1])
        fwhms[s]  = 2. * np.sqrt(2. * np.log(2.)) * samp[1] 
    
    genkeys = ['flux', 'fwhm']
    flux_fwhm = [fluxes, fwhms]
    for key, array in zip(genkeys, flux_fwhm):
        params[key] = profit.utils.ExtractParamValues(array, weights=None)

    # check to see if happy
    return profit.utils.ValidatePlot(params)

def LorentzianFit(wl:list, flux:list, errs:list, cwl:float, z:float, zerr:float=0.05) -> tpl[dict, bool]:

    # take in parameter guesses (visual)
    amp_guess = profit.utils.UserInput('Amplitude') 
    gamma_guess = profit.utils.UserInput('FWHM')

    if profit.options['open']: profit.utils.ClosePlot()

    # define log likelihood and prior transform functions
    def logl_lor(u:tuple) -> float:
        
        # extract parameters
        log_amp, gma, rs = u
        # determine log likelihood
        if gma <= 0: # or gma >= 10000:
            return -np.inf
        model = profit.utils.Lorentzian(x=wl, amp=10**log_amp, gamma=gma, xc = cwl*(1. + rs))
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
        params[keys[p]] = profit.utils.ExtractParamValues(par, weights=weights)

    # plot best fit model
    profit.plots.BestFitPlot(wl=wl, fluxes=flux, errors=errs, params=params, cen=(cwl), mode='lorentzian')
    # plot corner plot
    profit.plots.CornerPlot(results=res, mode='lorentzian')

    # use samples_equal to get fluxes under gaussians, and fwhm 
    fluxes, fwhms = np.empty(samples_equal.shape[0]), np.empty(samples_equal.shape[0])

    for s, samp in enumerate(samples_equal):
        fluxes[s] = profit.utils.FluxUnderLorentzian(amp=np.power(10,samp[0]), gam=samp[1])
        fwhms[s] = 2 * samp[1]

    # generate distributions in these parameters
    genkeys = ['flux', 'fwhm']
    flux_fwhm = [fluxes, fwhms]
    for key, array in zip(genkeys, flux_fwhm):
        params[key] = profit.utils.ExtractParamValues(array, weights=None)
    """
    for gp, genparam in enumerate(genkeys):
        extracted_values = profit.utils.ExtractParamValues(flux_fwhm[gp], weights=None)
        params[genparam] = extracted_values
    """
    # check to see if happy
    return profit.utils.ValidatePlot(params)

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
    amp_guess = profit.utils.UserInput('Amplitude')
    broad_amp_lims  = [0, 0.5 * amp_guess]
    narrow_amp_lims = [0.5 * amp_guess, 2.5 * amp_guess]
    narrow_vel_lims  = [profit.options['vel_barrier']/4, 1.75 * profit.options['vel_barrier']] 
    broad_vel_lims = [profit.options['vel_barrier'], 5*profit.options['vel_barrier']]
    redshift_lims = [z - 0.05, z + 0.05]

    # close plot
    if profit.options['open']: profit.utils.ClosePlot()

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
        params[keys[p]] = profit.utils.ExtractParamValues(par, weights=weights)
        graph_params.append(params[keys[p]][0])
        print(keys[p], profit.utils.ExtractParamValues(par, weights=weights))

    # generate samples in terms of sig
    sig_n_samples = Sigma(samples.T[2], cwl, samples.T[4])
    sig_b_samples = Sigma(samples.T[3], cwl, samples.T[4])
    params['sig_b'] = profit.utils.ExtractParamValues(sig_b_samples, weights=weights)
    params['sig_n'] = profit.utils.ExtractParamValues(sig_n_samples, weights=weights)

    # plot best fit model
    profit.plots.BestFitPlot(wl=wl, fluxes=flux, errors=errs, params=params, cen=(cwl), mode='stacked')
    # plot corner plot
    profit.plots.CornerPlot(results=res, mode='stacked')

    # work out fluxes
    flux_n, flux_b = np.empty(samples_equal.shape[0]), np.empty(samples_equal.shape[0])
    fwhm_n, fwhm_b = np.empty(samples_equal.shape[0]), np.empty(samples_equal.shape[0])
    for s, samp in enumerate(samples_equal):
            flux_n[s] = profit.utils.FluxUnderGaussian(amp=np.power(10,samp[0]), sig=Sigma(samp[2], cwl, samp[-1]))
            flux_b[s] = profit.utils.FluxUnderGaussian(amp=np.power(10,samp[1]), sig=Sigma(samp[3], cwl, samp[-1]))
            fwhm_n[s] = 2. * np.sqrt(2. * np.log(2.)) * Sigma(samp[2], cwl, samp[-1])
            fwhm_b[s] = 2. * np.sqrt(2. * np.log(2.)) * Sigma(samp[3], cwl, samp[-1])


    # generate distributions in these parameters
    genkeys = ['flux_n', 'flux_b', 'fwhm_n', 'fwhm_b']
    flux_fwhm = [flux_n, flux_b, fwhm_n, fwhm_b]
    for key, array in zip(genkeys, flux_fwhm):
        params[key] = profit.utils.ExtractParamValues(array, weights=None)

    return profit.utils.ValidatePlot(params)

def TwoSigmaLimit(wl:list, flux:list, err:list, cwl:float, z:float) -> tpl[dict, bool]:
    """ Guess the lower limits of the flux of an emission line """

    # take in initial guesses
    min_flux = profit.utils.UserInput('Lowest Flux Value')
    sigma    = profit.utils.UserInput('Sigma')
    n_mcmc   = profit.utils.UserInput('Number of MCMC')

    # generate input flxes
    factors = np.array([1., 1.5, 2., 3., 4., 5.])
    input_fluxes = factors * min_flux

    # storage arrays
    retrieved_fluxes = np.empty_like(input_fluxes)
    retrieved_errors = np.empty_like(input_fluxes)

    # generate model
    Model = lambda x: x ** 2
    gmodel = Model(profit.utils.Gaussian)

    # loop 
    for i, input_flux in enumerate(input_fluxes):

        iamp = input_flux / (sigma * np.sqrt(2 * np.pi))
        model_flux = profit.utils.Gaussian(x=wl, amp=iamp, xc=cwl*(1. + z), sig=sigma)
        out_fluxes = np.empty(n_mcmc)

        # loop through mcmc simulations
        for n in range(n_mcmc):

            flux_fit = model_flux + np.random.normal(loc=0.0, scale=err)

            params = gmodel.make_params(x=wl, amp=iamp, xc=cwl*(1. + z), sig=sigma)
            params['sig'].min = 0.0

            fit = gmodel.fit(flux_fit, params, x=wl)
            out_fluxes[n] = fit.params['amp'].value * fit.params['sig'].value * np.sqrt(2 * np.pi)

        # mask out bad fluxes
        valid = (out_fluxes < 1.e-15) & (out_fluxes > 1.e-20)
        retrieved_fluxes[i] = np.mean(retrieved_fluxes[valid])
        retrieved_errors[i] = np.std(retrieved_fluxes[valid])

        # plotting
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # plot non-scaled
        ax[0].errorbar(input_fluxes / 1e-19, retrieved_fluxes / 1e-19, 
                       xerr=None, yerr = retrieved_errors / 1e-19,
                       marker='o', c='k', ms=8, capsize=3)
        # plot scaled
        ax[1].scatter(input_fluxes / 1e-19, retrieved_fluxes  / retrieved_errors,
                      marker='o', s=50, color='k')
        
        # work out lin-regress
        _x = np.linspace(0.0, 1.1 * np.max(input_fluxes) / 1e-19, 1000)
        ax[0].plot(_x, _x, ls='-.', color='mediumseagreen', lw=.75, zorder=-1)

        # linregress
        m, c, rv, pv, std_dev = linregress(input_fluxes / 1e-19, retrieved_fluxes  / retrieved_errors)
        ax[1].plot(_x, _x * m + c, ls='-.', lw=.75, c='orchid', zorder=-1)

        # calculte two-sigma limit
        twosiglim = (2. - c) / m
        print(f'-> [profit]: 2-sigma limit: {twosiglim * 1e-19}')
        ax[1].axvline(twosiglim, c='crimson', ls='dashed')

        ax[0].set_xlabel(
            r'Input Flux / 10$^{-19}$ erg/s/cm$^2$/$\rm{\AA}$', fontsize=15)
        ax[0].set_ylabel(
            r'Retrieved Flux / 10$^{-19}$ erg/s/cm$^2$/$\rm{\AA}$', fontsize=15)

        ax[1].set_xlabel(
            r'Input Flux / 10$^{-19}$ erg/s/cm$^2$/$\rm{\AA}$', fontsize=15)
        ax[1].set_ylabel('S/N', fontsize=15)

        plt.show()         

        params = {
            'two_sigma_limit': twosiglim
        }

        return profit.utils.ValidatePlot(params)
