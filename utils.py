# relevant imports
import numpy as np
from astropy.io import fits
from dynesty.utils import quantile as _quantile
import pickle, sys
from typing import Tuple as tpl
import profit
import matplotlib.pyplot as plt
from matplotlib import cm, colors

# -=-=-=- Runtime Methods -=-=-=-

def SetLine(line:str) -> None:
    """ Sets the line that profit should fit for according to input """
    profit.options['line_name'] = line
    profit.options['line_wl']   = profit.options[line][:-1] # allows for double peaks
    profit.options['line_path'] = profit.options[line][-1]

def SavePlots(yn:bool) -> None:
    """ Gives the option to save plots """
    profit.options['save_plots'] = yn

def GenerateColours(n:int, cmname:str) -> list:
    """ generate a set of sequential colours according to a colormap """
    cmap = cm.get_cmap(cmname, n)
    return [colors.rgb2hex(cmap(i)) for i in range(0, cmap.N)] 

def UserInput(parameter:str) -> float:
    
    # handle amplitude (as contains an e) separately:
    if 'Amplitude' in parameter:
        while True:
            try:
                if 'e' in guess:
                    return float(guess)
                guess = 0
            except:
                guess = input(f'{parameter} Guess: ')
    else:
        while True:
            try:
                return float(guess)
            except:
                guess = input(f'{parameter} Guess: ')


# -=-=-=- Data Loading / Saving Methods -=-=-=-

def FastPP(name:str, path:str) -> tpl[object, object, object]:

    # get fluxes
    fhdu = fits.open(f'{path}/{name}.fits')
    fluxes = fhdu[1].data
    errors = fhdu[2].data
    wavelengths = fhdu[3].data

    return wavelengths, fluxes, errors


def kmos1D(name:str, path:str) -> tpl[object, object, object]:

    # get name
    #name = path.split('/')[-1].split('.')[0]

    # get fluxes
    fhdu = fits.open(f'{path}/data/{name}.fits')
    fluxes = fhdu[0].data
    hdr = fhdu[0].header

    # get errors
    ehdu = fits.open(f'{path}/error/{name}.fits')
    errors = ehdu[0].data

    # generate wavelength axis
    wavelengths = None
    wl0 = hdr['CRVAL1']
    dwl = hdr['CDELT1']
    naxis = fluxes.shape[0]
    wavelengths = np.arange(wl0, wl0 + (dwl*naxis), dwl) * 1e4

    return wavelengths, fluxes, (errors)

def kmos1D_n(name:str, path:str) -> tpl[object, object, object]:

    # get fluxes
    fhdu = fits.open(f'{path}/{name}.fits')
    fluxes = fhdu[1].data
    hdr = fhdu[1].header

    # get errors
    errors = fhdu[2].data

    # generate wavelength axis
    wl0 = hdr['CRVAL1']
    dwl = hdr['CDELT1']
    naxis = fluxes.shape[0]
    wavelengths = np.arange(wl0, wl0 + (dwl*naxis), dwl) * 1e4

    return wavelengths, fluxes, errors

def smacs1d(name:str, path:str) -> tpl[object, object, object]:
    
    hdu = fits.open(path)
    data = hdu[1].data
    hdr = hdu[0].header

    Primary = hdu['PRIMARY'].data
    Flux = hdu['DATA'].data
    Flux_Error = hdu['ERR'].data
    Wavelengths = hdu['WAVELENGTH'].data  * 1e10 # s* 1e6 #  Observation Conversion Å Conversion?

    return Wavelengths, Flux, Flux_Error

def PklSave(filepath:str, results:dict) -> None:
    with open(filepath, "wb") as file:
        pickle.dump(results, file)

def PklLoad(filepath:str) -> dict:
    results = pickle.load(open(filepath, 'rb'))
    return results
    
# -=-=-=- Data Interpretation Methods -=-=-=-

def Unpack(lims:list, par:float) -> object:
    return lims[0] + (par * (lims[1] - lims[0]))

def FluxUnderGaussian(amp:float, sig:float) -> float:
    return amp * sig * np.sqrt(2 * np.pi)

def FluxUnderLorentzian(amp:float, gam:float) -> float:
    return (np.pi / 2) * amp * gam

def FWHM(sig:float) -> float:
    return 2. * np.sqrt(2 * np.log(2.)) * sig

def InvertFWHM(fwhm:float) -> float:
    return fwhm / (2. * np.sqrt(2 * np.log(2.)))

def Sigma_To_Vel(sigma:float) -> float:
    fwhm = FWHM(sigma)
    return 3e5 * fwhm / np.mean(profit.options['line_wl'])

def Sigma_From_Vel(vel:float) -> float:
    fwhm = (np.mean(profit.options['line_wl']) * (1. + profit.options['redshift'])) * vel / 3e5
    return InvertFWHM(fwhm)    

def Velocity_z(sigma:float) -> float:
    fwhm = FWHM(sigma)
    return 3e5 * fwhm / (np.mean(profit.options['line_wl']) * (1. + profit.options['redshift']))

def Velocity_z_FWHM(fwhm:float) -> float:
    return 3e5 * fwhm / (np.mean(profit.options['line_wl']) * (1. + profit.options['redshift']))

def Vel_To_Sigma(vel:float) -> float:
    fwhm = np.mean(profit.options['line_wl']) * vel / 3e5
    return InvertFWHM(fwhm)

def Stack_Sigma_Conversion(vel:float, cwl:float, z:float) -> float:
    fwhm = (cwl * (1. + z)) * vel / 3e5
    return InvertFWHM(fwhm)

def ExtractParamValues(par:object, weights:object=None) -> tuple:
    ql, qm, qh = _quantile(par, [0.16, 0.50, 0.84], weights=weights)
    return (qm, qm-ql, qh-qm)

def ExtractUnweightedParams(par:list) -> tpl[float, float, float]:
    parl, par, parh = _quantile(par, [0.16, 0.50, 0.84], weights=None)
    return (par, par - parl, parh - par)
    

# -=-=-=- Model Making Methods -=-=-=-

def Lorentzian(x:list, amp:float, gamma:float, xc:float) -> list:
    return (amp/np.pi) * (0.5 * gamma) / ( np.power(x-xc, 2) + np.power(0.5*gamma, 2)) 

def GaussianStack(x:np.ndarray, namp:float, bamp:float, nsig:float, bsig:float, xc:float) -> np.ndarray:
    narrow = Gaussian(x, namp, nsig, xc)
    broad  = Gaussian(x, bamp, bsig, xc)
    return broad + narrow

def Gaussian(x:object, amp:float, xc:float, sig:float) -> list:
    return amp * np.exp(-np.power(x-xc, 2) / (2 * np.power(sig, 2)))

def DoubleGaussian(x:object, amp1:float, xc1:float, amp2:float, xc2:float, sig:float) -> list:
    return amp1 * np.exp(-np.power(x-xc1, 2) / (2 * np.power(sig, 2))) + amp2 * np.exp(-np.power(x-xc2, 2) / (2 * np.power(sig, 2)))

# -=-=-=- Plot Handling Methods -=-=-=-

def ClosePlot():
    profit.options['open'] = False
    plt.close()

# -=-=-=- Runtime Management Methods -=-=-=-

def ValidatePlot(params):
    """ checks user validity of the plot """

    # ensures valid input
    valid = 'x'
    while valid not in ['y', 'n', 'r']:
        valid = input("Valid Result (y/r/n): ").lower()

    if valid == 'y':
        params['success'] = True
        return params, True
    elif valid == 'n':
        print('Setting Default Null Value')
        params['success'] = False
        # data
        params['z'] = [-999, -999, -999]
        params['amp'] = [-999, -999, -999]
        params['sig'] = [-999, -999, -999]
        params['flux'] = [-999, -999, -999]
        params['fwhm'] = [-999, -999, -999]
        return params, True
    elif valid == 'r':
        params['success'] = True
        return params, False
    else:
        print('Invalid Input... Exiting')
        sys.exit()

