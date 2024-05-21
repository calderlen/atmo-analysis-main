#Code from Anusha

import numpy as np
import matplotlib.pyplot as plt
import petitRADTRANS.nat_cst as nc
from petitRADTRANS import Radtrans
from specutils import Spectrum1D
from specutils.fitting import fit_continuum
from astropy import units as u
from scipy.interpolate import UnivariateSpline
import scipy.interpolate as interpolate

def wav_indices(wav, wav_low, wav_up):
    ''' Finds lower and upper index of wavelength range.
        Inputs:
        - Array of wavelengths
        - Lower bound of wavelength range
        - Upper bound of wavelength range
        Outputs:
        - Lower bound index
        - Upper bound index
    '''
    i_low = np.where(wav >= wav_low)[0][0]
    i_up = np.where(wav <= wav_up)[0][-1]
    return i_low, i_up

def interpolate_spec(wav, flx, wav_out):
    '''
    Takes in:
    -List of spectrum wavelengths
    -List of spectrum fluxes
    -List of wavelength grid

    Returns:
    -List of spectrum fluxes interpolated to wavelength grid
    '''
    quadinterp = interpolate.interp1d(wav, flx, kind='slinear', bounds_error=False, fill_value=np.nan)
    return quadinterp(wav_out)

def norm_spec4(wav, flx, wavs_norm, smooth):
    ''' Normalizes continuum using spline fitting.
        Inputs:
        - List of wavelengths
        - List of fluxes
        - Continuum cutoff fraction
        - Smoothing factor
        Outputs:
        - List of wavelengths
        - List of normalized fluxes
    '''
    wav_cont = np.array([])
    flx_cont = []
    for w in wavs_norm:
        i_l, i_u = wav_indices(wav, w[0], w[1])
        wav_sec = wav[i_l:i_u]
        flx_sec = flx[i_l:i_u]
        spl = UnivariateSpline(wav_sec, flx_sec)
        spl.set_smoothing_factor(smooth)
        wav_cont = np.hstack([wav_cont, wav_sec])
        flx_cont = np.hstack([flx_cont, spl(wav_sec)])
    flx_cont_interp = interpolate_spec(wav_cont, flx_cont, wav)
    cont_norm_spec = flx/flx_cont_interp
    return wav, cont_norm_spec, wav_cont, flx_cont

def vacuum2air(wav_vacuum):
    ''' Converts linelist wavelengths from vacuum to air.
        Inputs:
        - List of vacuum wavelengths
        Outputs:
        - List of air wavelengths
    '''
    s = (10.**4)/wav_vacuum
    n = 1 + 0.0000834254 + (0.02406147/(130 - s**2)) + (0.00015998/(38.9 - s**2))
    wav_air = wav_vacuum/n
    return wav_air

#Adapted from BANZAI-NRES

def ccf(wave, flux, flux_error, template_wave, template_flux, rvmin, rvmax, rvspacing):

    drv = np.arange(rvmin, rvmax, rvspacing)
    nrv = len(drv)
    ckms = 2.9979e5

    variance = flux_error**2

    cross_cor, sigma_cross_cor = np.array([]), np.array([])

    for shift in drv:
        doppler_shift = 1.0 / (1.0 + shift / ckms)
        kernel = np.interp(doppler_shift * wave, template_wave, template_flux)
        corr = kernel*flux
        corr /= variance
        #normalization = kernel * kernel / variance #this normalization is actually just the uncertainty on the CCF, so this should put in in SNR. Can't do this normalization here b/c need to account for his when combine CCFs
        ccf = np.sum(corr)#/np.sqrt(np.sum(normalization))
        sigma_ccf = np.sqrt(np.sum((kernel/flux_error)**2))#/np.sqrt(np.sum(normalization))
        cross_cor, sigma_cross_cor = np.append(cross_cor,ccf), np.append(sigma_cross_cor,sigma_ccf)
        #import pdb; pdb.set_trace()
    return drv, cross_cor, sigma_cross_cor

#this first function just returns a single log-likelihood value
#note: this function DOES NOT compute the terms that are constant with different shifts, need to compute those in the higher-level routine!
def one_log_likelihood(wave, flux, variance, template_wave, template_flux, shift, norm_offset):
    ckms = 2.9979e5
    doppler_shift = 1.0 / (1.0 + shift / ckms)
    model = np.interp(doppler_shift * wave, template_wave, template_flux) + norm_offset
    model_term = np.sum(model**2 / variance)
    CCF_term = np.sum(flux * model / variance)


    return model_term, CCF_term

#this function does it over a range of Doppler shifts
def log_likelihood_CCF(wave, flux, flux_error, template_wave, template_flux, rvmin, rvmax, rvspacing, alpha, beta, norm_offset):

    bigN = len(flux)

    constant_term = (-1.0) * bigN / 2. * np.log(2.*np.pi) - bigN * np.log(beta) - np.sum(np.log(flux_error))

    #import pdb; pdb.set_trace()

    drv = np.arange(rvmin, rvmax, rvspacing)
    nrv = len(drv)

    variance = flux_error**2

    flux_term = np.sum(flux**2 / variance)

    model_terms, CCF_terms = np.array([]), np.array([])

    for shift in drv:
        model_term, CCF_term = one_log_likelihood(wave, flux, variance, template_wave, template_flux, shift, norm_offset)
        model_terms, CCF_terms = np.append(model_terms, model_term), np.append(CCF_terms, CCF_term)

    chi2 = 1/beta**2 * (flux_term + alpha**2 * model_terms - 2. * alpha * CCF_terms)

    lnL = constant_term - 0.5 * chi2

    print(constant_term, np.mean(model_terms), np.mean(CCF_terms), np.mean(flux_term))

    return drv, lnL

def log_likelihood_opt_beta(wave, flux, flux_error, template_wave, template_flux, rvmin, rvmax, rvspacing, alpha, norm_offset):

    #this function is the implementation from Gibson et al. 2020, where they analytically minimize the error scaling parameter beta

    ckms = 2.9979e5
    bigN = len(flux)
    
    drv = np.arange(rvmin, rvmax, rvspacing)
    nrv = len(drv)

    chi2s = np.array([])

    for shift in drv:

        doppler_shift = 1.0 / (1.0 + shift / ckms)
        model = np.interp(doppler_shift * wave, template_wave, template_flux)

        model += norm_offset

        chi2 = np.sum((flux - alpha * model)**2/flux_error**2)

        chi2s = np.append(chi2s, chi2)

    lnL = (-1.) * bigN / 2. * np.log(chi2s / bigN)

    #import pdb; pdb.set_trace()

    return drv, lnL
