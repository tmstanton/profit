# relevant imports
import profit
from profit import utils
import numpy as np
from scipy import stats
from scipy.stats import linregress
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'sans serif'
plt.rcParams['font.size'] = 16
from matplotlib.ticker import AutoMinorLocator
from dynesty.utils import quantile as _quantile
from dynesty import utils as dyfunc
from dynesty import plotting as dyplot
from typing import Tuple as tpl
import sys
import astropy.modeling as apm

# plot set methods
def _format_axes(ax:object):
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='both',
        bottom=True, top=True,
        left=True, right=True)
    ax.tick_params(which='both',
        labelbottom=True, labeltop=False,
        labelleft=True, labelright=False)
    ax.tick_params(which='both', direction='in')
    return ax
# -=-=-=- Plotting Methods -=-=-=-

def InitialPlot(wl_fit:list, flux_fit:list, errs_fit:list, approx_wl:float, fit_type:str, verbose:bool=True, xlow:float=None, xupp:float=None) -> None:

    # fitting wl
    wl_apg = np.linspace(wl_fit[0], wl_fit[-1], 1000)

    # show the spectrum
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axhline(0.0, ls='-', lw=.5, color='grey')
    ax.plot(wl_fit, flux_fit, drawstyle='steps', c='black', lw=1., label='Data')
    ax.plot(wl_fit, errs_fit, ls='-.', c='crimson', lw=.5, drawstyle='steps', label='Error')

    # astropy trial fit
    if fit_type == 'single':
        g_init = apm.models.Gaussian1D(np.max(flux_fit), mean=approx_wl, stddev=8.)
        fit_g = apm.fitting.LevMarLSQFitter()
        ap_fit = fit_g(g_init, wl_fit, flux_fit)
        ax.plot(wl_apg, ap_fit(wl_apg), c='mediumseagreen', lw=.5, label='Astropy Guess')
        if verbose:
            print(f'\n-> [profit]: Astropy Initial Guess Fit: \nAmplitude: {ap_fit.amplitude.value:.2e}')
            print(f'Central Wavelength: {ap_fit.mean.value:.2f} Å  \nWidth: {ap_fit.stddev.value:.2f}')

    elif fit_type == 'double':
        g1_init = apm.models.Gaussian1D(.5 * np.max(flux_fit), 
                    mean=profit.Line.wls[0] * (1. + profit.options['redshift']), stddev=8.)
        g2_init = apm.models.Gaussian1D(.5 * np.max(flux_fit),
                    mean=profit.Line.wls[1] * (1. + profit.options['redshift']), stddev=8.)
        fit_g = apm.fitting.LevMarLSQFitter()
        ap_fit1 = fit_g(g1_init, wl_fit, flux_fit)
        ap_fit2 = fit_g(g2_init, wl_fit, flux_fit)
        ax.plot(wl_apg, ap_fit1(wl_apg), c='mediumseagreen', lw=.3)
        ax.plot(wl_apg, ap_fit2(wl_apg), c='royalblue', lw=.3, label='Astropy Guess')
        #ap_doublet = ap_fit1(wl_apg) + ap_fit2(wl_apg)
        #ax.plot(wl_apg, ap_doublet, c='green', lw=.5, ls='dashed', label='Astropy Guess')
        if verbose:
            print(f'\n-> [profit]: Astropy Initial Guess Fit (L): \nAmplitude: {ap_fit1.amplitude.value:.2e}')
            print(f'Central Wavelength: {ap_fit1.mean.value:.2f} Å  \nWidth: {ap_fit1.stddev.value/2:.2f}')     
            print(f'-> [profit]: Astropy Initial Guess Fit (R): \nAmplitude: {ap_fit2.amplitude.value:.2e}')
            print(f'Central Wavelength: {ap_fit2.mean.value:.2f} Å  \nWidth: {ap_fit2.stddev.value/2:.2f}')    

    elif fit_type == 'triple':
            g1_init = apm.models.Gaussian1D(.5 * np.max(flux_fit), 
                        mean=profit.Line.wls[0] * (1. + profit.options['redshift']), stddev=8.)
            g2_init = apm.models.Gaussian1D(.5 * np.max(flux_fit),
                        mean=profit.Line.wls[1] * (1. + profit.options['redshift']), stddev=8.)
            g3_init = apm.models.Gaussian1D(.5 * np.max(flux_fit),
                        mean=profit.Line.wls[2] * (1. + profit.options['redshift']), stddev=8.)
            fit_g = apm.fitting.LevMarLSQFitter()
            ap_fit1 = fit_g(g1_init, wl_fit, flux_fit)
            ap_fit2 = fit_g(g2_init, wl_fit, flux_fit)
            ap_fit3 = fit_g(g3_init, wl_fit, flux_fit)
            ax.plot(wl_apg, ap_fit1(wl_apg), c='mediumseagreen', lw=.3)
            ax.plot(wl_apg, ap_fit2(wl_apg), c='royalblue', lw=.3, label='Astropy Guess')
            ax.plot(wl_apg, ap_fit3(wl_apg), c='gold', lw=.3, label='Astropy Guess')
            #ap_doublet = ap_fit1(wl_apg) + ap_fit2(wl_apg)
            #ax.plot(wl_apg, ap_doublet, c='green', lw=.5, ls='dashed', label='Astropy Guess')
            if verbose:
                print(f'\n-> [profit]: Astropy Initial Guess Fit (L): \nAmplitude: {ap_fit1.amplitude.value:.2e}')
                print(f'Central Wavelength: {ap_fit1.mean.value:.2f} Å  \nWidth: {ap_fit1.stddev.value/2:.2f}')     
                print(f'-> [profit]: Astropy Initial Guess Fit (C): \nAmplitude: {ap_fit2.amplitude.value:.2e}')
                print(f'Central Wavelength: {ap_fit2.mean.value:.2f} Å  \nWidth: {ap_fit2.stddev.value/2:.2f}')   
                print(f'-> [profit]: Astropy Initial Guess Fit (R): \nAmplitude: {ap_fit3.amplitude.value:.2e}')
                print(f'Central Wavelength: {ap_fit3.mean.value:.2f} Å  \nWidth: {ap_fit3.stddev.value/2:.2f}') 

    elif fit_type == 'stacked':
        g_init = apm.models.Gaussian1D(np.max(flux_fit), mean=approx_wl, stddev=8.)
        fit_g = apm.fitting.LevMarLSQFitter()
        ap_fit = fit_g(g_init, wl_fit, flux_fit)
        ax.plot(wl_apg, ap_fit(wl_apg), c='mediumseagreen', lw=.5, label='Astropy Guess')
        if verbose:
            print(f'\n-> [profit]: Astropy Initial Guess Fit: \nAmplitude: {ap_fit.amplitude.value:.2e}')
            print(f'Central Wavelength: {ap_fit.mean.value:.2f} Å  \nWidth: {ap_fit.stddev.value:.2f}')
        """
        gn_init = apm.models.Gaussian1D(np.max(flux_fit), mean=approx_wl, stddev=utils.Vel_To_Sigma(250))
        gb_init = apm.models.Gaussian1D(0.25 * np.max(flux_fit), mean=approx_wl, stddev=utils.Vel_To_Sigma(750))
        fit_g = apm.fitting.LevMarLSQFitter()
        ap_n_fit = fit_g(gn_init, wl_fit, flux_fit)
        ap_b_fit = fit_g(gb_init, wl_fit, flux_fit)
        ax.plot(wl_apg, ap_n_fit(wl_apg), c='green', lw=.5, label='Astropy Guess')
        ax.plot(wl_apg, ap_b_fit(wl_apg), c='green', lw=.5)
        if verbose:
            print(f'\n-> [profit]: Astropy Initial Guess Fit: \nMax Amplitude: {ap_n_fit.amplitude.value:.2e}')
            print(f'Central Wavelength: {ap_n_fit.mean.value:.2f} Å  \nNarrow Width: {ap_n_fit.stddev.value:.2f}')
            print(f'Central Wavelength: {ap_b_fit.mean.value:.2f} Å  \nBroad Width: {ap_b_fit.stddev.value:.2f}')
        """
            
    elif fit_type == 'lorentzian':
        l_init = apm.models.Lorentz1D(np.max(flux_fit), x_0 = approx_wl, fwhm = 20.)
        fit_l = apm.fitting.LevMarLSQFitter()
        ap_fit = fit_l(l_init, wl_fit, flux_fit)
        ax.plot(wl_apg, ap_fit(wl_apg), c='mediumseagreen', lw=.5, label='Astropy Guess')
        if verbose:
            print(f'\n-> [profit]: Astropy Initial Guess Fit: \nAmplitude: {ap_fit.amplitude.value:.2e}')
            print(f'Central Wavelength: {ap_fit.x_0.value:.2f} Å  \nFWHM: {ap_fit.fwhm.value:.2f}')      

    ax.axvline(approx_wl, ls=':', c='grey', lw=.5)
    # potential further y lim
    ax.set_xlabel(r'Wavelength $\lambda$ / Å')
    ax.set_ylabel(r'Flux F$_\lambda$ / erg s$^{-1}$ Å$^{-1}$ cm$^{-2}$')
    #ax.legend(loc='upper right')
    ax.set_title(profit.Line.title)
    #if ax.get_ylim()[1] > 2e-18:
    #    ax.set_ylim(ax.get_ylim()[0], 2e-18)
    #ax.set_ylim(factor * 1.1 * np.min(flux_fit), 1.5 * np.max(flux_fit))
    
    # set xlims if necessary
    if (xlow is not None) & (xupp is not None):
        ax.set_xlim(xlow, xupp)
    
    ax = _format_axes(ax)
    plt.show(block=False)

def CornerPlot(results:object, mode:str) -> None:
    # works for single or double gaussian modes
    if mode == 'single':
        labels = ['$\log \mathrm{amp_1}$', '$\sigma$', '$z$']
    elif mode == 'double':
        labels = ['$\log \mathrm{amp_1}$', '$\log \mathrm{amp_2}$', '$\sigma$', '$z$']
    elif mode == 'triple':
        labels = ['$\log \mathrm{amp_{1}}$', '$\log \mathrm{amp_2}$', '$\log \mathrm{amp_3}$', '$\sigma$', '$z$']
    elif mode == 'lorentzian':
        labels = ['$\log \mathrm{amp}$', '$\Gamma$', '$z$']
    elif mode == 'stacked':
        labels = ['$\log \mathrm{amp_b}$', '$\log \mathrm{amp_n}$', '$\sigma_\mathrm{b}$', '$\sigma_\mathrm{n}$', '$z$']
    else:
        print('-> [profit]: Incorrect plotting mode used in CornerPlot. Exiting...')
        sys.exit()
    cfig, cax = dyplot.cornerplot(results, 
                                  color='k', 
                                  labels=labels,
                                  show_titles=True,
                                  quantiles=(0.16, 0.50, 0.84),
                                  title_quantiles=(0.16, 0.50, 0.84),
                                  title_fmt='.2f')
    #plt.show()
    if profit.options['save_plots']:
        plt.savefig(f'{profit.options["plot_dir"]}/{profit.Line.id}.corner.png',
                    dpi=cfig.dpi)
    plt.close()

def BestFitPlot(wl:object, fluxes:object, errors:object, params:dict, cen:object, mode:str) -> None:
    
    # general plot set up
    windowfig, ax = plt.subplots(figsize=(8,6))
    ax.axhline(0., ls='-', lw=.5, color='royalblue')
    ax.set_title(profit.Line.title)
    ax.set_xlabel(r'Wavelength $\lambda$ / Å')
    ax.set_ylabel(r'Flux F$_\lambda$ / erg s$^{-1}$ Å$^{-1}$ cm$^{-2}$')

    # plot data
    ax.errorbar(wl, fluxes, yerr=errors, fmt='.', ecolor='k', c='k', lw=.5, label='Data')

    # fit data
    gaussx = np.linspace(wl[0], wl[-1], 1000)
    ax.axvline(np.mean(profit.Line.wls) * (1. + profit.options['redshift']), ls=':', c='k', lw=.5)

    if mode == 'single':
    
        # plot centers
        c1 = cen
        ax.axvline(c1 * (1. + params['z'][0]), ls=':', c='grey', lw=.5)

        # plot single gaussian
        ax.plot(gaussx, utils.Gaussian(x=gaussx, amp=10**params['amp'][0], 
                                   xc=cen * (1. + params['z'][0]), sig=params['sig'][0]),
                                   c='darkorchid', lw=1., ls='-', label='Best Fit')
        
        # output results
        strparams = [10**params['amp'][0], params['z'][0], params['sig'][0]]
        print(f'-> [profit]: Best Fit Parameters:\n -> Amplitude: {strparams[0]:.4e}\n -> Redshift: {strparams[1]:.4f}\n -> Width: {strparams[2]:.4f}')

    elif mode == 'double':

        # plot center
        c1, c2 = cen
        ax.axvline(c1 * (1. + params['z'][0]), ls=':', c='grey', lw=.5)
        ax.axvline(c2 * (1. + params['z'][0]), ls=':', c='grey', lw=.5)

        # plot double gaussian
        ax.plot(gaussx, utils.DoubleGaussian(x=gaussx, sig=params['sig'][0],
                                         amp1=10**params['amp1'][0], xc1=c1 * (1. + params['z'][0]),
                                         amp2=10**params['amp2'][0], xc2=c2 * (1. + params['z'][0])),
                                         c='darkorchid', lw=1.5, ls='-')

        # plot single gaussians
        ax.plot(gaussx, utils.Gaussian(x=gaussx, amp=10**params['amp1'][0], 
                                   xc=c1 * (1. + params['z'][0]), sig=params['sig'][0]),
                                   c='orchid', lw=1., ls='-', alpha=.75)

        ax.plot(gaussx, utils.Gaussian(x=gaussx, amp=10**params['amp2'][0], 
                                   xc=c2 * (1. + params['z'][0]), sig=params['sig'][0]),
                                   c='orchid', lw=1., ls='-', alpha=.75)

        # output results
        strparams = [10**params['amp1'][0], 10**params['amp2'][0], params['z'][0], params['sig'][0]]
        print(f'-> [profit]: Best Fit Parameters:\n -> Amplitude 1: {strparams[0]:.4e}\n -> Amplitude 2: {strparams[1]:.4e}')
        print(f'-> Redshift: {strparams[2]:.4f}\n -> Sigma: {strparams[3]:.4f}')

    elif mode == 'triple':

        # plot center
        c1, c2, c3 = cen
        ax.axvline(c1 * (1. + params['z'][0]), ls=':', c='grey', lw=.5)
        ax.axvline(c2 * (1. + params['z'][0]), ls=':', c='grey', lw=.5)
        ax.axvline(c3 * (1. + params['z'][0]), ls=':', c='grey', lw=.5)

        # plot double gaussian
        ax.plot(gaussx, utils.TripleGaussian(x=gaussx, sig=params['sig'][0],
                                         amp1=10**params['amp1'][0], xc1=c1 * (1. + params['z'][0]),
                                         amp2=10**params['amp2'][0], xc2=c2 * (1. + params['z'][0]),
                                         amp3=10**params['amp3'][0], xc3=c3 * (1. + params['z'][0])),
                                         c='darkorchid', lw=1.5, ls='-')

        # plot single gaussians
        ax.plot(gaussx, utils.Gaussian(x=gaussx, amp=10**params['amp1'][0], 
                                   xc=c1 * (1. + params['z'][0]), sig=params['sig'][0]),
                                   c='orchid', lw=1., ls='-', alpha=.75)

        ax.plot(gaussx, utils.Gaussian(x=gaussx, amp=10**params['amp2'][0], 
                                   xc=c2 * (1. + params['z'][0]), sig=params['sig'][0]),
                                   c='orchid', lw=1., ls='-', alpha=.75)

        ax.plot(gaussx, utils.Gaussian(x=gaussx, amp=10**params['amp3'][0], 
                                   xc=c3 * (1. + params['z'][0]), sig=params['sig'][0]),
                                   c='orchid', lw=1., ls='-', alpha=.75)

        # output results
        strparams = [10**params['amp1'][0], 10**params['amp2'][0], 10**params['amp3'][0], params['z'][0], params['sig'][0]]
        print(f'-> [profit]: Best Fit Parameters:\n -> Amplitude 1: {strparams[0]:.4e}\n -> Amplitude 2: {strparams[1]:.4e}\n -> Amplitude 3: {strparams[2]:.4e}')
        print(f'-> Redshift: {strparams[3]:.4f}\n -> Sigma: {strparams[4]:.4f}')

    elif mode == 'stacked':
        
        # plot centers
        c1 = cen
        ax.axvline(c1 * (1. + params['z'][0]), ls=':', c='grey', lw=.5)

        # plot narrow
        narrow = utils.Gaussian(x=gaussx, amp=10**params['amp_n'][0], 
                                   xc=c1 * (1. + params['z'][0]), sig=params['sig_n'][0])
        ax.plot(gaussx, narrow, c='darkorchid', lw=1., ls='--', label='narrow')

        # plot broad
        broad = utils.Gaussian(x=gaussx, amp=10**params['amp_b'][0], 
                                   xc=c1 * (1. + params['z'][0]), sig=params['sig_b'][0])
        ax.plot(gaussx, broad, c='orchid', lw=1., ls='--', label='broad')

        # plot combination
        #utils.GaussianStack(x=gaussx, namp=10**params['amp_n'][0], bamp=10**params['amp_b'][0],
                                          #nsig=params['sig_n'][0], bsig=params['sig_b'][0],
                                          #xc=c1 * (1. + params['z'][0]))
        ax.plot(gaussx, (narrow + broad), c='green', lw=1.5, ls='-', label='combined')

        # output results
        strparams = [10**params['amp_n'][0], 10**params['amp_b'][0], params['z'][0], params['sig_n'][0], params['sig_b'][0]]
        print(f'-> [profit]: Best Fit Parameters:\n -> Narrow Amp: {strparams[0]:.4e}\n -> Broad Amp: {strparams[1]:.4e}')
        print(f'-> Redshift: {strparams[2]:.4f}\n -> Narrow Sigma: {strparams[3]:.4f}\n -> Broad Sigma: {strparams[4]:.4f}')

    elif mode == 'lorentzian':

        # plot centers
        c1 = cen
        ax.axvline(c1 * (1. + params['z'][0]), ls=':', c='grey', lw=.5)

        # plot lorentzian
        ax.plot(gaussx, utils.Lorentzian(x=gaussx, gamma=params['gamma'][0], 
                                   xc=cen * (1. + params['z'][0]), amp=10**params['amp'][0]),
                                   c='darkorchid', lw=1., ls='-', label='Best Fit')

        strparams = [10**params['amp'][0], params['z'][0], params['gamma'][0]]
        print(f'-> [profit]: Best Fit Parameters:\n -> Amplitude: {strparams[0]:.4e}\n -> Redshift: {strparams[1]:.4f}\n -> FWHM: {strparams[2]:.4f}')

    else:
        print('-> [profit]: Incorrect plotting mode used in BestFitPlot. Exiting...')
        sys.exit()

    ax = _format_axes(ax)

    #plt.legend(loc='upper right')
    if np.min(fluxes) > 0:
        factor = -1
    else:
        factor = 1
    #ax.set_ylim(factor * 1.1 * np.min(fluxes), 1.5 * np.max(fluxes))
    if profit.options['save_plots']:
        plt.savefig(f'{profit.options["plot_dir"]}/{profit.Line.id}.window.png',
                    dpi=windowfig.dpi)
    plt.show(block=False)
    profit.options['open'] = True

def ContinuumPlot(wl_fit:list, flux_fit:list, errs_fit:list, approx_wl:float, fit_type:str, p:list) -> bool:
    
    # set up plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # params used for both plots
    wl_apg = np.linspace(wl_fit[0], wl_fit[-1], 1000)

    # plots on both halves (0 = non-normalised, 1 = normalised)
    alphas = [0.25, 1.]
    for a in range(2):

        # plot continuum line on first plot and reduced spectrum on the second plot 
        if a == 0:
            continuum_fluxes = p[0] * wl_fit + p[1]
            ax.plot(wl_fit, continuum_fluxes, ls='-', c='orange', lw=.5, alpha=alphas[a])

        else:
            flux_fit -= continuum_fluxes

        # astropy trial fits
        if fit_type == 'single':
            g_init = apm.models.Gaussian1D(np.max(flux_fit), mean=approx_wl, stddev=8.)
            fit_g = apm.fitting.LevMarLSQFitter()
            ap_fit = fit_g(g_init, wl_fit, flux_fit)
            ax.plot(wl_apg, ap_fit(wl_apg), c='mediumseagreen', lw=.5, label='Astropy Guess', alpha=alphas[a])

        elif fit_type == 'double':
            # TODO: add
            pass

        elif fit_type == 'stacked':
            gn_init = apm.models.Gaussian1D(np.max(flux_fit), mean=approx_wl, stddev=utils.Vel_To_Sigma(250))
            gb_init = apm.models.Gaussian1D(0.25 * np.max(flux_fit), mean=approx_wl, stddev=utils.Vel_To_Sigma(750))
            fit_g = apm.fitting.LevMarLSQFitter()
            ap_n_fit = fit_g(gn_init, wl_fit, flux_fit)
            ap_b_fit = fit_g(gb_init, wl_fit, flux_fit)
            ax.plot(wl_apg, ap_n_fit(wl_apg), c='mediumseagreen', lw=.5, label='Astropy Guess', alpha=alphas[a])
            ax.plot(wl_apg, ap_b_fit(wl_apg), c='mediumseagreen', lw=.5)

        elif fit_type == 'lorentzian':
            l_init = apm.models.Lorentz1D(np.max(flux_fit), x_0 = approx_wl, fwhm = 20.)
            fit_l = apm.fitting.LevMarLSQFitter()
            ap_fit = fit_l(l_init, wl_fit, flux_fit)
            ax.plot(wl_apg, ap_fit(wl_apg), c='mediumseagreen', lw=.5, label='Astropy Guess', alpha=alphas[a]) 

        # show the spectrum
        ax.axhline(0.0, ls='-', lw=.5, color='royalblue')
        ax.plot(wl_fit, flux_fit, drawstyle='steps', c='black', lw=1., label='Data', alpha=alphas[a])
        ax.plot(wl_fit, errs_fit, ls='-.', c='crimson', lw=.5, drawstyle='steps', label='Error', alpha=alphas[a])  

        ax.axvline(approx_wl, ls=':', c='k', lw=.5)
        # potential further y lim
        ax.set_xlabel(r'Wavelength $\lambda$ / Å')
        ax.set_ylabel(r'Flux F$_\lambda$ / erg s$^{-1}$ Å$^{-1}$ cm$^{-2}$')

    # handle legend
    #handles, labels = plt.gca().get_legend_handles_labels()
    #by_label = dict(zip(labels, handles))
    #ax.legend(by_label.values(), by_label.keys(), loc='upper right')

    # final plotting and show
    plt.suptitle('Comparison including Continuum Normalisation')
    ax = _format_axes(ax)
    plt.show(block=False)

    # get user feedback
    good_fit = True if input('  -> Good Fit (y/n): ').lower() == 'y' else False
    return good_fit


