#Code from Anusha
import numpy as np
import matplotlib.pyplot as plt
from petitRADTRANS import physical_constants as cst
from petitRADTRANS.radtrans import Radtrans
from specutils.fitting import fit_continuum
# from astropy.modeling.polynomial import Chebyshev1D
# from astropy.modeling.fitting import LevMarLSQFitter
from astropy import units as u
from JW_lib import gaussian_filter1d
import time
from petitRADTRANS.physics import temperature_profile_function_guillot_global

def instantiate_radtrans(species, lambda_low, lambda_high, pressures, downsample_factor=1):

    if downsample_factor == 1:
        atmosphere = Radtrans(line_species = species, \
                          rayleigh_species = ['H2', 'He'], \
                          gas_continuum_contributors = ['H2-H2', 'H2-He', 'H-'], \
                          wavelength_boundaries = [lambda_low/10.**4,lambda_high/10.**4], \
                          line_opacity_mode = 'lbl')
    else:
        atmosphere = Radtrans(line_species = species, \
                          rayleigh_species = ['H2', 'He'], \
                          gas_continuum_contributors = ['H2-H2', 'H2-He', 'H-'], \
                          wavelength_boundaries = [lambda_low/10.**4,lambda_high/10.**4], \
                          line_opacity_mode = 'lbl', line_by_line_opacity_sampling = downsample_factor)
        # default rayleigh_species and gas_contiuum_contributors
    atmosphere.setup_opa_structure(pressures)

    return atmosphere
    
def create_model(params, spectrum_type, do_contribution, new_atmo, atmosphere='null', ptprofile = 'guillot'):

    # Select template spectrum routine
    species = params[0]
    lambda_low = params[1]
    lambda_high = params[2]
    R_pl = params[3]
    R_s = params[4]
    gravity = params[5]
    P0 = params[6]
    if ptprofile == 'guillot':
        kappa_IR = params[7]
        gamma = params[8]
    elif ptprofile == 'two-point':
        P1 = params[7]
        P2 = params[8]
    T_int = params[9]
    T_equ = params[10]
    abundances = params[11]
    pressures = params[12]
    if ptprofile == 'two-point': T_high = params[13]

    if new_atmo:
        # Set-up atmosphere as radiative transfer object with varying pressure layers, if not already done:
        atmosphere = instantiate_radtrans(species, lambda_low, lambda_high, pressures)

    # Define planetary radius, gravity, temperatures, abundances, and mean molecular weight:
    if ptprofile == 'guillot':
        temperature = temperature_profile_function_guillot_global(pressures, kappa_IR, gamma, gravity, T_int, T_equ)
    if ptprofile == 'two-point':
        temperature = np.ones_like(pressures)
        temperature = T_equ + (T_high - T_equ) / (np.log10(P2) - np.log10(P1)) * (np.log10(pressures) - np.log10(P1))
        temperature[pressures >= P1] = T_equ
        temperature[pressures <= P2] = T_high


        temperature = gaussian_filter1d(temperature,sigma=6.0,mode="nearest")


    for key in abundances:
        if isinstance(abundances[key], np.ndarray):
            abundances[key] = abundances[key][0]
        abundances[key] *= np.ones_like(temperature)
        

    MMW = 2.33 * np.ones_like(temperature)

    # Calculate and normalize transmission spectrum
    if spectrum_type == 'emission':
        atmosphere.calc_flux(temperature, abundances, gravity, MMW, contribution=do_contribution)
        out_pl = atmosphere.flux #units of 10^-6 erg/cm2/s/Hz
        contribution = atmosphere.contr_em
        for key in species:
            abundances[key] *= 0.0
        atmosphere.calc_flux(temperature, abundances, gravity, MMW, contribution=do_contribution)
        out_pl_2 = atmosphere.flux #units of 10^-6 erg/cm2/s/Hz
        out_pl = out_pl - out_pl_2
    if spectrum_type == 'transmission':
        atmosphere.calc_transm(temperature, abundances, gravity, MMW, R_pl=R_pl, P0_bar=P0,contribution=do_contribution)#, Pcloud = 1e-4)
        out_pl = atmosphere.transm_rad
        contribution = atmosphere.contr_tr
        
    wav_pl = cst.c/atmosphere.freq/1e-8 

    

    return wav_pl, out_pl, temperature, pressures, contribution
