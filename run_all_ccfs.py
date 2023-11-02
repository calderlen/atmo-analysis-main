#import packages
import numpy as np
import matplotlib.pyplot as pl
import matplotlib.gridspec as gridspec

from glob import glob
from astropy.io import fits

from scipy.stats import chisquare
from scipy.optimize import curve_fit

from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
from astropy.table import Table

from uncertainties import ufloat
from uncertainties import unumpy as unp

from atmo_utilities import ccf, one_log_likelihood, log_likelihood_CCF, vacuum2air, log_likelihood_opt_beta

import time

from dtutils import psarr

from radiant import generate_atmospheric_model, get_wavelength_range

from create_model import create_model, instantiate_radtrans
import horus


pl.rc('font', size=14) #controls default text size
pl.rc('axes', titlesize=14) #fontsize of the title
pl.rc('axes', labelsize=14) #fontsize of the x and y labels
pl.rc('xtick', labelsize=14) #fontsize of the x tick labels
pl.rc('ytick', labelsize=14) #fontsize of the y tick labels
pl.rc('legend', fontsize=14) #fontsize of the legend


 
def get_species_keys(species_label):
    """
    Given a species label, returns the corresponding species name for injection and CCF calculation.
    If the species label is not found in the list, the function will not return anything.

    Args:
    - species_label (str): The label of the species for which to retrieve the corresponding names.

    Returns:
    - species_name_inject (str): The name of the species to be used for injection.
    - species_name_ccf (str): The name of the species to be used for CCF calculation.
    """
    
    if species_label == 'TiO':
        species_name_inject = 'TiO_all_iso_Plez'
        species_name_ccf = 'TiO_all_iso_Plez'

    if species_label == 'TiO_46':
        species_name_inject = 'TiO_46_Exomol_McKemmish'
        species_name_ccf = 'TiO_46_Exomol_McKemmish'
        
    if species_label == 'TiO_47':
        species_name_inject = 'TiO_47_Exomol_McKemmish'
        species_name_ccf = 'TiO_47_Exomol_McKemmish'

    if species_label == 'TiO_48':
        species_name_inject = 'TiO_48_Exomol_McKemmish'
        species_name_ccf = 'TiO_48_Exomol_McKemmish'
        
    if species_label == 'TiO_49':
        species_name_inject = 'TiO_49_Exomol_McKemmish'
        species_name_ccf = 'TiO_49_Exomol_McKemmish'
        
    if species_label == 'TiO_50':
        species_name_inject = 'TiO_50_Exomol_McKemmish'
        species_name_ccf = 'TiO_50_Plez'

    if species_label == 'VO':
        species_name_inject = 'VO_ExoMol_McKemmish'
        species_name_ccf = 'VO_ExoMol_McKemmish'

    if species_label == 'FeH':
        species_name_inject = 'FeH_main_iso'
        species_name_ccf = 'FeH_main_iso'

    if species_label == 'CaH':
        species_name_inject = 'CaH'
        species_name_ccf = 'CaH'

    if species_label == 'Fe I':
        species_name_inject = 'Fe'
        species_name_ccf = 'Fe'

    if species_label == 'Ti I':
        species_name_inject = 'Ti'
        species_name_ccf = 'Ti'

    if species_label == 'Ti II':
        species_name_inject = 'Ti+'
        species_name_ccf = 'Ti+'

    if species_label == 'Mg I':
        species_name_inject = 'Mg'
        species_name_ccf = 'Mg'

    if species_label == 'Mg II':
        species_name_inject = 'Mg+'
        species_name_ccf = 'Mg+'

    if species_label == 'Fe II':
        species_name_inject = 'Fe+'
        species_name_ccf = 'Fe+'

    if species_label == 'Cr I':
        species_name_inject = 'Cr'
        species_name_ccf = 'Cr'

    if species_label == 'Si I':
        species_name_inject = 'Si'
        species_name_ccf = 'Si'

    if species_label == 'Ni I':
        species_name_inject = 'Ni'
        species_name_ccf = 'Ni'

    if species_label == 'Al I':
        species_name_inject = 'Al'
        species_name_ccf = 'Al'

    if species_label == 'SiO':
        species_name_inject = 'SiO_main_iso_new_incl_UV'
        species_name_ccf = 'SiO_main_iso_new_incl_UV'

    if species_label == 'H2O':
        species_name_inject = 'H2O_main_iso'
        species_name_ccf = 'H2O_main_iso'

    if species_label == 'OH':
        species_name_inject = 'OH_main_iso'
        species_name_ccf = 'OH_main_iso'

    if species_label == 'MgH':
        species_name_inject = 'MgH'
        species_name_ccf = 'MgH'

    if species_label == 'Ca I':
        species_name_inject = 'Ca'
        species_name_ccf = 'Ca'
        
    if species_label == 'NaH':
        species_name_inject = 'NaH'
        species_name_ccf = 'NaH'
        
    if species_label == 'H I':
        species_name_inject = 'H'
        species_name_ccf = 'H'
        
    if species_label == 'AlO':
        species_name_inject = 'AlO' 
        species_name_ccf = 'AlO'

    if species_label == 'Ba I':
        species_name_inject = 'Ba'
        species_name_ccf = 'Ba'

    if species_label == 'Ba II':
        species_name_inject == 'Ba+'
        species_name_ccf == 'Ba+'

    if species_label == 'CaO':
        species_name_inject = 'CaO'
        species_name_ccf = 'CaO'

    if species_label == 'Co I':
        species_name_inject = 'Co' 
        species_name_ccf = 'Co'
        
    if species_label == 'Cr II':
        species_name_inject = 'Cr+'
        species_name_ccf = 'Cr+'
        
    if species_label == 'Cs I':
        species_name_inject = 'Cs'
        species_name_ccf = 'Cs'
        
    if species_label == 'Cu I':
        species_name_inject = 'Cu' 
        species_name_ccf = 'Cu'
        
    if species_label == 'Ga I':
        species_name_inject = 'Ga' 
        species_name_ccf = 'Ga'
        
    if species_label == 'Ge I':
        species_name_inject = 'Ge' 
        species_name_ccf = 'Ge'
        
    if species_label == 'Hf I':
        species_name_inject = 'Hf'
        species_name_ccf = 'Hf'
        
    if species_label == 'In I':
        species_name_inject = 'In' 
        species_name_ccf = 'In'
        
    if species_label == 'Ir I':
        species_name_inject = 'Ir' 
        species_name_ccf = 'Ir'

    if species_label == 'Mn I':
        species_name_inject = 'Mn' 
        species_name_ccf = 'Mn'

    if species_label == 'Mo I':
        species_name_inject = 'Mo'
        species_name_ccf = 'Mo'
        

    if species_label == 'Na I':
        species_name_inject = 'Na'
        species_name_ccf = 'Na'

    if species_label == 'NaH':
        species_name_inject = 'NaH'
        species_name_ccf = 'NaH'

    if species_label == 'Nb I':
        species_name_inject = 'Nb'
        species_name_ccf = 'Nb'

    if species_label == 'Ni I':
        species_name_inject = 'Ni'
        species_name_ccf = 'Ni'

    if species_label == 'O I':
        species_name_inject = 'O'
        species_name_ccf = 'O'

    if species_label == 'Os I':
        species_name_inject = 'Os'
        species_name_ccf = 'Os'

    if species_label == 'Pb I':
        species_name_inject = 'Pb'
        species_name_ccf = 'Pb'

    if species_label == 'Pd I':
        species_name_inject = 'Pd'
        species_name_ccf = 'Pd'

    if species_label == 'Rb I':
        species_name_inject = 'Rb'
        species_name_ccf = 'Rb'

    if species_label == 'Rh I':
        species_name_inject = 'Rh'
        species_name_ccf = 'Rh'
        
    if species_label == 'Ru I':
        species_name_inject = 'Ru'
        species_name_ccf = 'Ru'

    if species_label == 'Sc I':
        species_name_inject = 'Sc'
        species_name_ccf = 'Sc'

    if species_label == 'Sc II':
        species_name_inject = 'Sc+'
        species_name_ccf = 'Sc+'

    if species_label == 'Sn I':
        species_name_inject = 'Sn'
        species_name_ccf = 'Sn'

    if species_label == 'Sr I':
        species_name_inject = 'Sr'
        species_name_ccf = 'Sr'

    if species_label == 'Sr II':
        species_name_inject = 'Sr+'
        species_name_ccf = 'Sr+'

    if species_label == 'Ti II':
        species_name_inject = 'Ti+'
        species_name_ccf = 'Ti+'

    if species_label == 'Tl':
        species_name_inject = 'Tl'
        species_name_ccf = 'Tl'
        
    if species_label == 'W I':
        species_name_inject = 'W'
        species_name_ccf = 'W'
        
    if species_label == 'Y II':
        species_name_inject = 'Y+'
        species_name_ccf = 'Y+'
        
    if species_label == 'Zn I':
        species_name_inject = 'Zn'
        species_name_ccf = 'Zn'
        
    if species_label == 'Zr I':
        species_name_inject = 'Zr'
        species_name_ccf = 'Zr'
        
    if species_label == 'Zr I':
        species_name_inject = 'Zr+'
        species_name_ccf = 'Zr+'

    if species_label == 'N I':
        species_name_inject = 'N'
        species_name_ccf = 'N'

    if species_label == 'K I':
        species_name_inject = 'K'
        species_name_ccf = 'K'

    if species_label == 'Y I':
        species_name_inject = 'Y'
        species_name_ccf = 'Y'

    if species_label == 'Li I':
        species_name_inject = 'Li'
        species_name_ccf = 'Li'

    if species_label == 'V I':
        species_name_inject = 'V'
        species_name_ccf = 'V'



    return species_name_inject, species_name_ccf

def get_sysrem_parameters(arm, observation_epoch, species_label):
    if species_label == 'TiO':
        if arm == 'red': n_systematics = [1, 1]
        if arm == 'blue': n_systematics = [2, 0]
    elif species_label == 'VO':
        if arm == 'red': n_systematics = [1, 2]
        if arm == 'blue': n_systematics = [3, 0]
    elif species_label == 'FeH':
        if arm == 'red': n_systematics = [1, 0]
    elif species_label == 'CaH':
        if arm == 'blue': n_systematics = [2, 0]
    else:
        if arm == 'blue':
            n_systematics = [0,5]
        if arm == 'red':
            n_systematics = [0,10]

    return n_systematics

def get_planet_parameters(planet_name):
    """
    Returns the orbital and physical parameters of a given exoplanet.

    Parameters:
    -----------
    planet_name : str
        The name of the exoplanet. Valid options are 'KELT-20b', 'WASP-76b', 'KELT-9b', 'WASP-12b', 'WASP-33b', and 'WASP-18b'.

    Returns:
    --------
    tuple
        A tuple containing the following elements:
        - Period: the orbital period of the planet in days (uncertainties are included as well)
        - epoch: the epoch of the first transit in BJD_TDB (uncertainties are included as well)
        - M_star: the mass of the host star in solar masses (uncertainties are included as well)
        - RV_abs: the absolute radial velocity of the star in km/s (uncertainties are included as well)
        - i: the inclination of the planet's orbit in degrees (uncertainties are included as well)
        - M_p: the mass of the planet in Jupiter masses (uncertainties are included as well)
        - R_p: the radius of the planet in Jupiter radii (uncertainties are included as well)
        - RA: the right ascension of the star in sexagesimal format
        - Dec: the declination of the star in sexagesimal format
        - Kp_expected: the expected radial velocity semi-amplitude of the star in km/s
        - half_duration_phase: the half-duration of the transit in phase units
    """
def get_planet_parameters(planet_name):

    MJoMS = 1./1047. #MJ in MSun

    if planet_name == 'KELT-20b':
        #For KELT-20 b:, from Lund et al. 2018
        Period = ufloat(3.4741085, 0.0000019) #days
        epoch = ufloat(2457503.120049, 0.000190) #BJD_TDB

        M_star = ufloat(1.76, 0.19) #MSun
        RV_abs = ufloat(0.0, 0.0) #km/s
        i = ufloat(86.12, 0.28) #degrees
        M_p = 3.382 #3-sigma limit
        R_p = 1.741

        RA = '19h38m38.74s'
        Dec = '+31d13m09.12s'

        dur = 3.755/24. #hours -> days
        
        #add vsini, lambda, spin-orbit-misalignment,any other horus params
        
    
    if planet_name == 'WASP-76b':
        #For WASP-76 b:, from West et al. 2016
        Period = ufloat(1.809886, 0.000001) #days
        epoch = ufloat(2456107.85507, 0.00034) #BJD_TDB

        M_star = ufloat(1.46, 0.07) #MSun
        RV_abs = ufloat(-1.152, 0.0033) #km/s
        i = ufloat(88.0, 1.6) #degrees
        M_p = 0.92
        R_p = 1.83

        RA = '01h46m31.90s'
        Dec = '+02d42m01.40s'

        dur = 3.694/24.
    
    if planet_name == 'KELT-9b':
        #For KELT-9 b:, from Gaudi et al. 2017 and Pai Asnodkar et al. 2021
        Period = ufloat(1.4811235, 0.0000011) #days
        epoch = ufloat(2457095.68572, 0.00014) #BJD_TDB

        M_star = ufloat(2.11, 0.78) #MSun
        RV_abs = ufloat(0.0, 1.0) #km/s
        i = ufloat(86.79, 0.25) #degrees
        M_p = ufloat(2.17, 0.56)

        RA = '20h31m26.38s'
        Dec = '+39d56m20.10s'

        dur = 3.9158/24.

    if planet_name == 'WASP-12b':
        #For WASP-12 b:, from Ivishina & Winn 2022, Bonomo+17, Charkabarty & Sengupta 2019
        Period = ufloat(1.091419108, 5.5e-08) #days
        epoch = ufloat(2457010.512173, 7e-05) #BJD_TDB

        M_star = ufloat(1.38, 0.18) #MSun
        RV_abs = ufloat(0.0, 0.0) #km/s
        i = ufloat(83.3, 1.1) #degrees
        M_p = ufloat(1.39, 0.12)

        RA = '06h30m32.79s'
        Dec = '+29d40m20.16s'

        dur = 3.0408/24.

    if planet_name == 'WASP-33b':
        #For WASP-33 b:, from Ivishina & Winn 2022, Bonomo+17, Charkabarty & Sengupta 2019
        Period = ufloat(1.219870, 0.000001) #days
        epoch = ufloat(2454163.22367, 0.00022) #BJD_TDB

        M_star = ufloat(1.495, 0.031) #MSun
        RV_abs = ufloat(0.0, 0.0) #km/s
        i = ufloat(86.63, 0.03) #degrees
        M_p = ufloat(2.093, 0.139)
        R_p = 1.593

        RA = '02h26m51.06s'
        Dec = '+37d33m01.60s '

        dur = 2.854/24.

    if planet_name == 'WASP-18b':
        #For WASP-18b: from Cortes-Zuleta+20
        Period = ufloat(0.94145223, 0.00000024) #days
        epoch = ufloat(2456740.80560, 0.00019) #BJD_TDB

        M_star = ufloat(1.294, 0.063) #MSun
        RV_abs = ufloat(0.0, 0.0) #km/s
        i = ufloat(83.5, 2.0) #degrees
        M_p = ufloat(10.20, 0.35)
        R_p = 1.240

        RA = '01h37m25.07s'
        Dec = '-45d40m40.06s'

        dur = 2.21/24.

    if planet_name == 'WASP-189b':
        
        Period = ufloat(2.7240308, 0.0000028) #days
        epoch = ufloat(2458926.5416960, 0.0000650) #BJD_TDB

        M_star = ufloat(2.030, 0.066) #MSun
        RV_abs = ufloat(-22.4, 0.0) #km/s
        i = ufloat(84.03, 0.14) #degrees
        M_p = ufloat(1.99, 0.16)
        R_p = 1.619

        RA = '15h02m44.82s'
        Dec = '-03d01m53.35s'

        dur = 4.3336/24.

    half_duration_phase = (dur/2.)/Period.n
    Kp_expected = 28.4329 * M_star/MJoMS * unp.sin(i*np.pi/180.) * (M_star + M_p * MJoMS) ** (-2./3.) * (Period/365.25) ** (-1./3.) / 1000. #to km/s
    half_duration_phase = (dur/2.)/Period.n

    return Period, epoch, M_star, RV_abs, i, M_p, R_p, RA, Dec, Kp_expected, half_duration_phase #add other outputs!!
    
def get_pepsi_data(arm, observation_epoch, planet_name, do_molecfit):

    ckms = 2.9979e5
    #get data
    if arm == 'blue':
        arm_file = 'pepsib'
    if arm == 'red':
        arm_file = 'pepsir'
    
    
    #change 'avr' to 'nor' below for more recent data
    if float(observation_epoch[0:4]) >= 2019:
        pepsi_extend = 'nor'
    else:
        pepsi_extend = 'avr'
        
    if not do_molecfit:
        data_location = '/home/calder/Documents/petitRADTRANS_data/data/' + observation_epoch + '_' + planet_name + '/' + arm_file + '*.dxt.' + pepsi_extend
    else:
        data_location = '/home/calder/Documents/petitRADTRANS_data/data/' + observation_epoch + '_' + planet_name + '/molecfit_weak/SCIENCE_TELLURIC_CORR_' + arm_file + '*.dxt.' + pepsi_extend + '.fits'
    spectra_files = glob(data_location)
    if not spectra_files:
        print('No files found at the provided location.')
        return

    n_spectra = len(spectra_files)
    i=0
    jd, snr_spectra, exptime = np.zeros(n_spectra), np.zeros(n_spectra), np.zeros(n_spectra)
    
    
    airmass = np.zeros(n_spectra)

    for spectrum in spectra_files:
        hdu = fits.open(spectrum)
        data, header = hdu[1].data, hdu[0].header
        if do_molecfit: wave_tag, flux_tag, error_tag = 'lambda', 'flux', 'error'
        if not do_molecfit: wave_tag, flux_tag, error_tag = 'Arg', 'Fun', 'Var'
        if i == 0:
            npix = len(data[wave_tag])
            wave, fluxin, errorin = np.zeros((n_spectra, npix)), np.zeros((n_spectra, npix)), np.zeros((n_spectra, npix))
        #have to do the following because for some reason some datasets do not have consistent numbers of pixels
        npixhere = len(data[wave_tag])
        if npixhere >= npix:
            wave[i,:] = data[wave_tag][0:npix]
            fluxin[i,:] = data[flux_tag][0:npix]
            errorin[i,:] = data[error_tag][0:npix]
        else:
            wave[i,0:npixhere] = data[wave_tag]
            fluxin[i,0:npixhere] = data[flux_tag]
            errorin[i,0:npixhere] = data[error_tag]

        #molecfit_utilities already handles variance->uncertainty
        if not do_molecfit: errorin[i,:]=np.sqrt(errorin[i,:]) 
        if do_molecfit:
            wave[i,:]*=10000. #microns -> Angstroms
        
            #remove shift introduced to make Molecfit work
            if observation_epoch == '20210501': introduced_shift = 6000.
            if observation_epoch == '20210518': introduced_shift = 3500.
            if observation_epoch == '20190425': introduced_shift = 464500.
            if observation_epoch == '20190504': introduced_shift = 6300.
            if observation_epoch == '20190515': introduced_shift = 506000.
            if observation_epoch == '20190623': introduced_shift = -334000.
            if observation_epoch == '20190625': introduced_shift = 97800.
            if observation_epoch == '20210303': introduced_shift = -174600.
            if observation_epoch == '20220208': introduced_shift = -141300.
            if observation_epoch == '20210628': introduced_shift = -57200.
            if observation_epoch == '20211031': introduced_shift = -94200.
            if observation_epoch == '20220929': introduced_shift = -38600.
            if observation_epoch == '20221202': introduced_shift = -96100.
            if observation_epoch == '20230327': introduced_shift = -23900.
            if observation_epoch == '20180703': introduced_shift = -61800.

            if pepsi_extend == 'nor':
                try:
                    doppler_shift = 1.0 / (1.0 - (hdu[0].header['RADVEL'] + hdu[0].header['OBSVEL'] + introduced_shift) / 1000. / ckms)
                except KeyError:
                    doppler_shift = 1.0 / (1.0 - (hdu[0].header['RADVEL'] + hdu[0].header['SSBVEL'] + introduced_shift) / 1000. / ckms)
            else:
                doppler_shift = 1.0 / (1.0 - (hdu[0].header['OBSVEL'] + introduced_shift) / 1000. / ckms) #note: old data does not correct to the stellar frame, only the barycentric

            wave[i,:] *= doppler_shift
        
    
        jd[i] = header['JD-OBS'] #mid-exposure time
        try:
            snr_spectra[i] = header['SNR']
        except KeyError:
            snr_spectra[i] = np.percentile(fluxin[i,:]/errorin[i,:], 90)
            
        exptime_strings = header['EXPTIME'].split(':')
        exptime[i] = float(exptime_strings[0]) * 3600. + float(exptime_strings[1]) * 60. + float(exptime_strings[2])            
        airmass[i] = header['AIRMASS']

        hdu.close()
        i+=1
    
    #glob gets the files out of order for some reason so we have to put them in time order

    obs_order = np.argsort(jd)


    jd, snr_spectra, exptime, airmass = jd[obs_order], snr_spectra[obs_order], exptime[obs_order], airmass[obs_order]

    wave, fluxin, errorin = wave[obs_order,:], fluxin[obs_order,:], errorin[obs_order,:]

    

    #This is approximate, to account for a small underestimate in the error by the pipeline, and at least approximately include the systematic effects due to Molecfit
    #error_estimated = np.zeros_like(fluxin)
    #for i in range (0, n_spectra):
    #    for j in range (10, npix-10):
    #        error_estimated[i,j] = np.std(fluxin[i,j-10:j+10])
    #    underestimate_factor = np.nanmedian(error_estimated[i,10:npix-10]/errorin[i,10:npix-10])
    #    errorin[i,:] *= underestimate_factor

    return wave, fluxin, errorin, jd, snr_spectra, exptime, airmass, n_spectra, npix

def get_orbital_phase(jd, epoch, Period, RA, Dec):
    """
    Calculates the orbital phase of an object given its Julian date, epoch, period, right ascension, and declination.

    Parameters:
    jd (float): Julian date of the observation.
    epoch (float): Epoch of the object.
    Period (float): Period of the object.
    RA (str): Right ascension of the object in the format 'hh:mm:ss'.
    Dec (str): Declination of the object in the format 'dd:mm:ss'.

    Returns:
    orbital_phase (float): The orbital phase of the object.
    """
    lbt_coordinates = EarthLocation.of_site('lbt')

    observed_times = Time(jd, format='jd', location=lbt_coordinates)

    coordinates = SkyCoord(RA+' '+Dec, frame='icrs', unit=(u.hourangle, u.deg))

    ltt_bary = observed_times.light_travel_time(coordinates)

    bary_times = observed_times + ltt_bary

    orbital_phase = (bary_times.value - epoch)/Period
    orbital_phase -= np.round(np.mean(unp.nominal_values(orbital_phase)))
    orbital_phase = unp.nominal_values(orbital_phase)


    return orbital_phase

def convolve_atmospheric_model(template_wave, template_flux, profile_width, profile_form, temperature_profile='emission', epsilon=0.6):
    """
    Convolve atmospheric model with a kernel.

    Args:
    template_wave (numpy.ndarray): Array of wavelengths.
    template_flux (numpy.ndarray): Array of fluxes.
    profile_width (float): Width of the kernel.
    profile_form (str): Form of the kernel. Can be 'rotational' or 'gaussian'.
    temperature_profile (str, optional): Temperature profile. Defaults to 'emission'.
    epsilon (float, optional): Epsilon value. Defaults to 0.6.

    Returns:
    numpy.ndarray: Convolved flux.
    """
    
    ckms =2.9979e5
    velocities = (template_wave - np.mean(template_wave)) / template_wave * ckms
    velocities = velocities[np.abs(velocities) <= 100.]
    if profile_form == 'rotational':
        if 'transmission' in temperature_profile:
            kernel = 1. / (np.pi * np.sqrt(1. - velocities**2 / profile_width**2))
        else:
            c1 = 2. * (1. - epsilon) / (np.pi * (1. - epsilon / 3.))
            c2 = epsilon / (2. * (1. - epsilon / 3.))
            kernel = c1 * (1. - (velocities / profile_width)**2.)**(1./2.) + c2 * (1. - (velocities / profile_width)**2.)

        kernel[~np.isfinite(kernel)] = 0.0
            
        

    if profile_form == 'gaussian':
        kernel = 1. / (profile_width * np.sqrt(2. * np.pi)) * np.exp((-1.) * velocities**2 / (2. * profile_width**2))
        
    kernel /= np.sum(kernel) #need to normalize G in order to conserve area under spectral lines
    convolved_flux = np.convolve(template_flux, kernel, mode='same')
    
    return convolved_flux

def do_convolutions(planet_name, template_wave, template_flux, do_rotate, do_instrument, temperature_profile, Resolve = 130000.):
    """
    Perform convolutions on a given atmospheric model.

    Args:
    planet_name (str): Name of the planet.
    template_wave (numpy.ndarray): Wavelength array of the atmospheric model.
    template_flux (numpy.ndarray): Flux array of the atmospheric model.
    do_rotate (bool): Whether or not to perform rotational convolution.
    do_instrument (bool): Whether or not to perform instrumental convolution.
    temperature_profile (numpy.ndarray): Temperature profile of the planet's atmosphere.
    Resolve (float): The resolving power of the instrument. Defaults to 130000.

    Returns:
    numpy.ndarray: The convolved atmospheric model.
    """
    if do_rotate:
        epsilon = 0.6
        Period, epoch, M_star, RV_abs, i, M_p, R_p, RA, Dec, Kp_expected, half_duration_phase = get_planet_parameters(planet_name)
        #the below assumes that the planet is tidally locked
        rotational_velocity = 2. * np.pi * R_p * 69911. / (Period.n * 24. *3600.)
        template_flux = convolve_atmospheric_model(template_wave, template_flux, rotational_velocity, 'rotational', temperature_profile=temperature_profile, epsilon=epsilon)

    if do_instrument:
        ckms = 2.9979e5
        sigma = ckms / Resolve / 2. #assuming that resolving power described the width of the line

        template_flux = convolve_atmospheric_model(template_wave, template_flux, sigma, 'gaussian')

    return template_flux
    

def make_spectrum_plot(template_wave, template_flux, planet_name, species_name_ccf, temperature_profile, vmr):
    """
    Plots a spectrum of a given planet and species.

    Args:
    - template_wave (array): wavelength array
    - template_flux (array): flux array
    - planet_name (str): name of the planet
    - species_name_ccf (str): name of the species
    - temperature_profile (str): temperature profile
    - vmr (float): volume mixing ratio

    Returns:
    - None
    """
    pl.fill([4800,4800,5441,5441],[np.nanmin(template_flux),np.nanmax(template_flux),np.nanmax(template_flux),np.nanmin(template_flux)],color='blue',alpha=0.25)
    pl.fill([6278,6278,7419,7419],[np.nanmin(template_flux),np.nanmax(template_flux),np.nanmax(template_flux),np.nanmin(template_flux)],color='red',alpha=0.25)

    pl.plot(template_wave, template_flux, color='black')

    pl.xlabel('wavelength (Angstroms)')

    pl.ylabel('normalized flux')

    pl.title(species_name_ccf)

    plotout = '/home/calder/Documents/atmo-analysis-main/plots/spectrum.' + planet_name + '.' + species_name_ccf + '.' + str(vmr) + '.' + temperature_profile + '.pdf'
    pl.savefig(plotout,format='pdf')
    pl.clf()

def make_new_model(instrument, species_name_new, vmr, spectrum_type, planet_name, temperature_profile, do_plot=False):
    """
    Generate a new atmospheric model for a given planet and species.

    Args:
    - instrument (str): the name of the instrument used to observe the planet
    - species_name_new (str): the name of the species to model
    - vmr (float): the volume mixing ratio of the species
    - spectrum_type (str): the type of spectrum to generate (e.g. 'transmission', 'emission')
    - planet_name (str): the name of the planet to model
    - temperature_profile (str): the type of temperature profile to use (e.g. 'inverted-emission-better', 'Borsa')
    - do_plot (bool): whether or not to generate a plot of the resulting spectrum

    Returns:
    - template_wave (numpy.ndarray): the wavelength array of the resulting spectrum
    - template_flux (numpy.ndarray): the flux array of the resulting spectrum
    """
    # function code here

    if planet_name == 'WASP-189b':
        instrument_here = 'PEPSI-25' 
    elif instrument == 'PEPSI':
        instrument_here = 'PEPSI-35'
    else:
        instrument_here = instrument
        
    lambda_low, lambda_high = get_wavelength_range(instrument_here)
    pressures = np.logspace(-8, 2, 100)
    atmosphere = instantiate_radtrans([species_name_new], lambda_low, lambda_high, pressures)
    #will need to generalize the below to any planet that is not KELT-20!
    parameters = {}
    parameters['Hm'] = 1e-9
    parameters['em'] = 0.0008355 #constant for now, link to FastChem later
    parameters['H0'] = 2.2073098e-12 #ditto
    parameters['P0'] = 1.0
    if planet_name == 'KELT-20b':
        parameters['Teq'] = 2262.
        if temperature_profile == 'inverted-emission-better' or temperature_profile == 'inverted-transmission-better':
            parameters['kappa'] = 0.04
            parameters['gamma'] = 30.
            ptprofile = 'guillot'

        elif 'Borsa' in temperature_profile:
            parameters['P1'], parameters['P2'] = 10.**(-0.11), 10.**(-4.90)
            parameters['Teq'], parameters['T_high'] = 2561.38, 5912.93
            ptprofile = 'two-point'
    
        elif 'Yan' in temperature_profile:
            parameters['P1'], parameters['P2'] = 10.**(-1.5), 10.**(-5.0)
            parameters['Teq'], parameters['T_high'] = 2550., 4900.
            ptprofile = 'two-point'
    
        elif 'Fu' in temperature_profile:
            parameters['P1'], parameters['P2'] = 7e-3, 1e-4
            parameters['Teq'], parameters['T_high'] = 2300., 3900.
            ptprofile = 'two-point'
    
        elif 'Kasper' in temperature_profile:
            parameters['gamma'] = 10.**(1.48)
            parameters['kappa'] = 10.**(-0.86)
            parameters['Teq'] = 2726.97
            ptprofile = 'guillot'
        
    elif planet_name == 'WASP-76b':

        if 'Edwards' in temperature_profile: #this one is for WASP-76 b
            parameters['P1'], parameters['P2'] = 3.2e1, 3.2e-6
            parameters['Teq'], parameters['T_high'] = 2450., 3200.
            ptprofile = 'two-point'
    elif planet_name == 'WASP-33b':
        
        if 'Herman' in temperature_profile:
            parameters['gamma'] = 2.0
            parameters['kappa'] = 0.01
            parameters['Teq'] = 3100.
            ptprofile = 'guillot'
        if 'Cont' in temperature_profile:
            parameters['P1'], parameters['P2'] = 10.**(-3.08), 10.**(-5.12)
            parameters['Teq'], parameters['T_high'] = 3424., 3981.
            ptprofile = 'two-point'

    elif planet_name == 'WASP-189b':
        if 'Yan' in temperature_profile:
            parameters['P1'], parameters['P2'] = 10.**(-1.7), 10.**(-3.1)
            parameters['Teq'], parameters['T_high'] = 2200., 4320.
            ptprofile = 'two-point'
    else:
        #fill these in later
        parameters['kappa'] = 0.01
        parameters['gamma'] = 50.


    
    
    parameters[species_name_new] = vmr
    template_wave, template_flux = generate_atmospheric_model(planet_name, spectrum_type, instrument, 'combined', [species_name_new], parameters, atmosphere, pressures, ptprofile = ptprofile)

    template_flux = do_convolutions(planet_name, template_wave, template_flux, True, True, temperature_profile)

    # if 'Plez' in species_name_new or species_name_new == 'Fe+' or species_name_new == 'Ti+' or species_name_new == 'Cr':
    # template_wave = vacuum2air(template_wave)

    if do_plot: make_spectrum_plot(template_wave, template_flux, planet_name, species_name_new, temperature_profile, vmr)   

    return template_wave, template_flux

def get_atmospheric_model(planet_name, species_name_ccf, vmr, temperature_profile, do_rotate, do_instrument):
    """
    Returns the atmospheric model for a given planet, species, VMR, temperature profile, and instrument configuration.

    Args:
        planet_name (str): Name of the planet.
        species_name_ccf (str): Name of the species.
        vmr (float): Volume mixing ratio.
        temperature_profile (str): Temperature profile.
        do_rotate (bool): Whether or not to rotate the planet.
        do_instrument (bool): Whether or not to apply instrument effects.

    Returns:
        tuple: A tuple containing the wavelength and flux arrays of the atmospheric model.
    """

    filein = '/home/calder/Documents/atmo-analysis-main/templates/' + planet_name + '.' + species_name_ccf + '.' + str(vmr) + '.' + temperature_profile + '.combined.fits'
    hdu = fits.open(filein)

    template_wave = hdu[1].data['wave']
    template_flux = hdu[1].data['flux']

    hdu.close()

    template_flux = do_convolutions(planet_name, template_wave, template_flux, True, True)

    return template_wave, template_flux




def inject_model(Kp_expected, orbital_phase, wave, fluxin, template_wave_in, template_flux_in, n_spectra):
    """
    Injects a model planet spectrum into a set of observed spectra.

    Args:
        Kp_expected (float): Expected radial velocity semi-amplitude of the planet.
        orbital_phase (float): Orbital phase of the planet.
        wave (ndarray): Array of wavelength values for each observed spectrum.
        fluxin (ndarray): Array of flux values for each observed spectrum.
        template_wave_in (ndarray): Array of wavelength values for the planet model spectrum.
        template_flux_in (ndarray): Array of flux values for the planet model spectrum.
        n_spectra (int): Number of observed spectra.

    Returns:
        tuple: A tuple containing:
            - **fluxin** (*ndarray*) - Array of flux values for each observed spectrum with the planet model injected.
            - **Kp_true** (*float*) - True radial velocity semi-amplitude of the planet.
            - **V_sys_true** (*float*) - True systemic velocity of the planet.
    """
    ckms = 2.9979e5
    scale_factor = 1.0
    Kp_true, V_sys_true = unp.nominal_values(Kp_expected), 0.0
     
    RV = Kp_true*np.sin(2.*np.pi*orbital_phase) + V_sys_true
    
    for i in range (n_spectra):
        doppler_shift = 1.0 / (1.0 - RV[i] / ckms)
        planet_flux = np.interp(wave[i,:], template_wave_in * doppler_shift, template_flux_in)
        if scale_factor != 1.0:
            planet_flux *=  scale_factor
          
        fluxin[i,:] = fluxin[i,:] + planet_flux

    return fluxin, Kp_true, V_sys_true

import numpy as np
import uncertainties.unumpy as unp

def regrid_data(wave, fluxin, errorin, n_spectra, template_wave, template_flux, snr_spectra, temperature_profile, do_make_new_model):
    """
    Regrids the input data to a common wavelength grid, and returns the regridded data along with weights for the combined CCFs.

    Args:
    wave (numpy.ndarray): 2D array of wavelength values for each spectrum.
    fluxin (numpy.ndarray): 2D array of flux values for each spectrum.
    errorin (numpy.ndarray): 2D array of error values for each spectrum.
    n_spectra (int): Number of spectra.
    template_wave (numpy.ndarray): 1D array of wavelength values for the template spectrum.
    template_flux (numpy.ndarray): 1D array of flux values for the template spectrum.
    snr_spectra (numpy.ndarray): 1D array of signal-to-noise ratios for each spectrum.
    temperature_profile (str): Temperature profile type ('transmission' or 'emission').
    do_make_new_model (bool): Whether or not to make a new model.

    Returns:
    tuple: A tuple containing:
        - wave (numpy.ndarray): 1D array of wavelength values for the regridded data.
        - flux (uncertainties.core.AffineScalarFunc): 2D array of flux values for the regridded data.
        - ccf_weights (numpy.ndarray): 1D array of weights for the combined CCFs.
    """
    for i in range (1, n_spectra):
        tempflux = np.interp(wave[0,:], wave[i,:], fluxin[i,:])
        temperror = np.interp(wave[0,:], wave[i,:], errorin[i,:])
        fluxin[i,:], errorin[i,:] = tempflux, temperror
    
    wave=wave[i,:]
    #can't do the proper error propagation until we regrid
    flux = unp.uarray(fluxin, errorin)

    #Regrid template spectrum for use in weighting combined CCFs
    if len(template_wave) > 0:
        regridded_template_flux = np.interp(wave, template_wave, template_flux)
        if 'transmission' in temperature_profile:
            ccf_weights = snr_spectra * np.sum((np.max(regridded_template_flux)-regridded_template_flux))
        else:
            ccf_weights = snr_spectra * np.sum(regridded_template_flux)
        if do_make_new_model and 'emission' in temperature_profile: ccf_weights = ccf_weights.value
    else:
        ccf_weights = []

    return wave, flux, ccf_weights

def flatten_spectra(flux, npix, n_spectra):
    median_flux1, median_error = np.zeros(npix), np.zeros(npix)
    residual_flux = unp.uarray(np.zeros((n_spectra, npix)), np.zeros((n_spectra, npix)))
    total_flux, total_error = np.zeros(npix), np.zeros(npix)

    for i in range (0, npix):
        median_flux1[i] = np.median(unp.nominal_values(flux[:,i]))
        #median doesn't really propagate errors correctly, so we need to do it by hand
        little_n = (n_spectra-1)/2
        #The latter estimated median error comes from https://mathworld.wolfram.com/StatisticalMedian.html
        median_error[i] = np.sqrt(np.sum(unp.std_devs(flux[:,i])**2))/n_spectra/np.sqrt(4*little_n/(np.pi*n_spectra))
        total_flux[i] = np.sum(unp.nominal_values(flux[:,i]))
        total_error[i] = np.sqrt(np.sum(unp.std_devs(flux[:,i])**2))
    
    median_flux = unp.uarray(median_flux1, median_error)
    
    for i in range (0, n_spectra):
        residual_flux[i,:] = flux[i, :] - median_flux

    total_snr = total_flux/total_error
    print('The maximum total SNR is ', np.max(total_snr))

    return residual_flux


def do_sysrem(wave, residual_flux, arm, airmass, n_spectra, niter, n_systematics, do_molecfit):
    """
    Perform systematics removal on a set of spectra using the SYSREM algorithm.

    Parameters:
    wave (numpy.ndarray): Array of wavelength values for each pixel in the spectra.
    residual_flux (uncertainties.core.AffineScalarFunc): Array of residual flux values for each pixel in the spectra.
    arm (str): The arm of the spectrograph used to obtain the spectra ('red' or 'blue').
    airmass (list): List of airmass values for each spectrum.
    n_spectra (int): The number of spectra being analyzed.
    niter (int): The number of iterations to perform for each systematic.
    n_systematics (list): List of the number of systematics to remove for each chunk of the spectra.
    do_molecfit (bool): Whether or not to mask out regions of the spectra affected by atmospheric absorption using molecfit.

    Returns:
    corrected_flux (numpy.ndarray): Array of corrected flux values for each pixel in the spectra.
    corrected_error (numpy.ndarray): Array of corrected error values for each pixel in the spectra.
    """
    #Running SYSREM with the uncertainties is just too slow, so unfortunately we have to go 
    #back to manual tracking
    corrected_flux = unp.nominal_values(residual_flux)
    corrected_error = unp.std_devs(residual_flux)
    if arm == 'red':
        no_tellurics = np.where((wave <= 6277.) | ((wave > 6328.) & (wave <= 6459.)) | ((wave > 6527.) & (wave <= 6867.)))
        has_tellurics = np.where(((wave > 6277.) & (wave <= 6328.)) | ((wave > 6459.) & (wave <= 6527.)) | ((wave > 6867.) & (wave < 6867.5)) | ((wave >= 6930.) & (wave < 7168.)) | (wave >= 7312.))
        has_tellurics, no_tellurics = has_tellurics[0], no_tellurics[0]
        if do_molecfit:
            do_mask = np.where(((wave >= 6867.5) & (wave < 6930.)) | ((wave >= 7168.) & (wave < 7312.)))
            do_mask = do_mask[0]
            corrected_flux[:,do_mask] = 0.0
            corrected_error[:,do_mask] = 1.0
        #else:
            
        chunks = 2
    else:
        no_tellurics = np.where((wave > 3000.) & (wave < 7000.))
        no_tellurics = no_tellurics[0]
        chunks = 1
    
    for chunk in range (chunks):
        if chunk == 0: this_one = no_tellurics
        if chunk == 1: this_one = has_tellurics
        npixhere = len(this_one)
        for system in range (n_systematics[chunk]):
    
            c, sigma_c, sigma_a = np.zeros(npixhere), np.zeros(npixhere), np.zeros(n_spectra)
            if system == 0:
                a = np.array(airmass)
            else:
                a = np.ones(n_spectra)
            
            for iter in range (niter):
            
                #minimize c for each pixel
                for s in range (npixhere):
                    err_squared = (corrected_error[:,this_one[s]])**2
                    numerator = np.sum(a*corrected_flux[:,this_one[s]]/err_squared)
                    saoa = sigma_a/a
                    bad = ~np.isfinite(saoa)
                    saoa[bad] = 0.0
                    eof = corrected_error[:,this_one[s]]/corrected_flux[:,this_one[s]]
                    bad = ~np.isfinite(eof)
                    eof[bad] = 0.0
                    sigma_1 = np.abs(a*corrected_flux[:,this_one[s]]/err_squared) * np.sqrt(saoa**2+eof**2)
                    sigma_numerator = np.sqrt(np.sum(sigma_1**2))
                    denominator = np.sum(a**2/err_squared)
                    sigma_2 = np.sqrt(2.)*np.abs(a)*sigma_a/err_squared
                    sigma_denominator = np.sqrt(np.sum(sigma_2**2))
                    c[s] = numerator/denominator
                    sigma_c[s] = np.abs(c[s]) * np.sqrt((sigma_numerator/numerator)**2+(sigma_denominator/denominator)**2)
                    
                #using c, minimize a for each epoch
                for ep in range (n_spectra):
                    err_squared = (corrected_error[ep,this_one])**2
                    numerator = np.sum(c*corrected_flux[ep,this_one]/err_squared) 
                    scoc = sigma_c/c
                    bad = ~np.isfinite(scoc)
                    scoc[bad] = 0.0
                    eof = corrected_error[ep,this_one]/corrected_flux[ep,this_one]
                    bad = ~np.isfinite(eof)
                    eof[bad] = 0.0
                    sigma_1 = np.abs(c*corrected_flux[ep,this_one]/err_squared) * np.sqrt(scoc**2+eof**2)
                    sigma_numerator = np.sqrt(np.sum(sigma_1**2))
                    denominator = np.sum(c**2/err_squared)
                    sigma_2 = np.sqrt(2.)*np.abs(c)*sigma_c/err_squared
                    sigma_denominator = np.sqrt(np.sum(sigma_2**2))
                    a[ep] = numerator/denominator
                    sigma_a[ep] = np.abs(a[ep]) * np.sqrt((sigma_numerator/numerator)**2+(sigma_denominator/denominator)**2)
                    
            #create matrix for systematic errors
            syserr, sigma_syserr = np.zeros((n_spectra, npixhere)), np.zeros((n_spectra, npixhere))
             
            for s in range (npixhere):
                for e in range (n_spectra):
                    syserr[e,s] = a[e]*c[s]
                    sigma_syserr[e,s] = np.abs(syserr[e,s]) * np.sqrt((sigma_a[e]/a[e])**2 + (sigma_c[s]/c[s])**2)
                    
            
            #remove systematic error
            corrected_flux[:,this_one] -= syserr
            corrected_error[:,this_one] = np.sqrt(corrected_error[:,this_one]**2 + sigma_syserr**2)

    return corrected_flux, corrected_error

def get_ccfs(wave, corrected_flux, corrected_error, template_wave, template_flux, n_spectra):
    """
    Computes the cross-correlation functions (CCFs) for a set of spectra.

    Args:
        wave (numpy.ndarray): Array of wavelengths for the spectra.
        corrected_flux (numpy.ndarray): Array of corrected flux values for the spectra.
        corrected_error (numpy.ndarray): Array of corrected error values for the spectra.
        template_wave (numpy.ndarray): Array of wavelengths for the template spectrum.
        template_flux (numpy.ndarray): Array of flux values for the template spectrum.
        n_spectra (int): Number of spectra to compute CCFs for.

    Returns:
        tuple: A tuple containing the following three elements:
            - drv (numpy.ndarray): Array of radial velocity values.
            - cross_cor (numpy.ndarray): Array of CCF values for each spectrum.
            - sigma_cross_cor (numpy.ndarray): Array of CCF errors for each spectrum.
    """
    rvmin, rvmax = -400., 400. #kms
    rvspacing = 1.0 #kms

    for i in range (n_spectra):
        drv, cross_cor_out, sigma_cross_cor_out = ccf(wave, corrected_flux[i,:], corrected_error[i,:], template_wave, template_flux, rvmin, rvmax, rvspacing)
        if i == 0:
            cross_cor, sigma_cross_cor = np.zeros((n_spectra, len(drv))), np.zeros((n_spectra, len(drv)))
        cross_cor[i,:], sigma_cross_cor[i,:] = cross_cor_out, sigma_cross_cor_out

    return drv, cross_cor, sigma_cross_cor

import numpy as np

def get_likelihood(wave, corrected_flux, corrected_error, template_wave, template_flux, n_spectra):
    """
    Calculates the likelihood of a given set of spectra using a template spectrum.

    Args:
    wave (numpy.ndarray): Array of wavelengths for the spectra.
    corrected_flux (numpy.ndarray): Array of corrected flux values for the spectra.
    corrected_error (numpy.ndarray): Array of corrected error values for the spectra.
    template_wave (numpy.ndarray): Array of wavelengths for the template spectrum.
    template_flux (numpy.ndarray): Array of flux values for the template spectrum.
    n_spectra (int): Number of spectra to calculate likelihood for.

    Returns:
    numpy.ndarray: Array of radial velocity values.
    numpy.ndarray: Array of likelihood values for each radial velocity value and each spectrum.
    """
    rvmin, rvmax = -400., 400. #kms
    rvspacing = 1.0 #kms

    alpha, beta, norm_offset = 1.0, 1.0, 0.0 #I think this is correct for this application--only set these scaling factors if do actual fits

    for i in range (n_spectra):
        #drv, lnL0 = log_likelihood_CCF(wave, corrected_flux[i,:], corrected_error[i,:], template_wave, template_flux, rvmin, rvmax, rvspacing, alpha, beta)
        drv, lnL0 = log_likelihood_opt_beta(wave, corrected_flux[i,:], corrected_error[i,:], template_wave, template_flux, rvmin, rvmax, rvspacing, alpha, norm_offset)
        if i == 0:
            lnL = np.zeros((n_spectra, len(drv)))
        lnL[i,:] = lnL0

    return drv, lnL

def combine_ccfs(drv, cross_cor, sigma_cross_cor, orbital_phase, n_spectra, ccf_weights, half_duration_phase, temperature_profile):
    """
    Combine cross-correlation functions (CCFs) from multiple spectra into a single CCF.

    Parameters:
    -----------
    drv : array_like
        Array of Doppler velocities (km/s) at which the CCFs are sampled.
    cross_cor : array_like
        2D array of CCFs, with shape (n_spectra, len(drv)).
    sigma_cross_cor : array_like
        2D array of uncertainties in the CCFs, with shape (n_spectra, len(drv)).
    orbital_phase : array_like
        Array of orbital phases of the planet, with values between 0 and 1.
    n_spectra : int
        Number of spectra to combine.
    ccf_weights : array_like
        Array of weights for each spectrum, with shape (n_spectra,).
    half_duration_phase : float
        Half duration of the transit in orbital phase units.
    temperature_profile : str
        Type of temperature profile to use ('isothermal' or 'transmission').

    Returns:
    --------
    snr : array_like
        Signal-to-noise ratio of the combined CCF, with shape (len(Kp), len(drv)).
    Kp : array_like
        Array of planet radial velocities (km/s) at which the CCFs are evaluated.
    drv : array_like
        Array of Doppler velocities (km/s) at which the CCFs are sampled.
    """
    
    Kp = np.arange(50, 350, 1)
    nKp, nv = len(Kp), len(drv)

    shifted_ccfs, var_shifted_ccfs = np.zeros((nKp, nv)), np.zeros((nKp, nv))

    i = 0

    for Kp_i in Kp:
        RV = Kp_i*np.sin(2.*np.pi*orbital_phase)
        
        for j in range (n_spectra):
            #restrict to only in-transit spectra if doing transmission:
            #print(orbital_phase[j])
            if not 'transmission' in temperature_profile or np.abs(orbital_phase[j]) <= half_duration_phase:
            
                temp_ccf = np.interp(drv, drv-RV[j], cross_cor[j, :], left=0., right=0.0)
                sigma_temp_ccf = np.interp(drv, drv-RV[j], sigma_cross_cor[j, :], left=0., right=0.0)
                shifted_ccfs[i,:] += temp_ccf * ccf_weights[j]
                var_shifted_ccfs[i,:] += sigma_temp_ccf**2 * ccf_weights[j]**2
        i+=1
    
    sigma_shifted_ccfs = np.sqrt(var_shifted_ccfs)

    #goods = np.abs(drv) <= 200.

    #drv = drv[goods]
    #shifted_ccfs, sigma_shifted_ccfs = shifted_ccfs[:,goods], sigma_shifted_ccfs[:,goods]

    shifted_ccfs -= np.median(shifted_ccfs) #handle any offset

    #use_for_snr = (np.abs(drv) <= 100.) & (np.abs(drv) >= 50.)#
    #use_for_snr = np.abs(drv) > 150.
    use_for_snr = np.abs(drv > 100.)
    #tempp = shifted_ccfs[:,use_for_snr]
    #use_for_snr_2 = (Kp <= 280.) & (Kp >= 180.)

    #snr = shifted_ccfs / np.std(tempp[use_for_snr_2,:])
    snr = shifted_ccfs / np.std(shifted_ccfs[:,use_for_snr])

    return snr, Kp, drv

def combine_likelihoods(drv, lnL, orbital_phase, n_spectra, half_duration_phase, temperature_profile):
    """
    Combines likelihoods for a given set of parameters.

    Parameters:
    -----------
    drv : numpy.ndarray
        Array of Doppler velocities.
    lnL : numpy.ndarray
        Array of log-likelihoods.
    orbital_phase : numpy.ndarray
        Array of orbital phases.
    n_spectra : int
        Number of spectra.
    half_duration_phase : float
        Half duration phase.
    temperature_profile : str
        Temperature profile.

    Returns:
    --------
    shifted_lnL : numpy.ndarray
        Array of shifted log-likelihoods.
    Kp : numpy.ndarray
        Array of Kp values.
    drv : numpy.ndarray
        Array of Doppler velocities.
    """
    Kp = np.arange(50, 350, 1)
    nKp, nv = len(Kp), len(drv)

    shifted_lnL = np.zeros((nKp, nv))

    i = 0

    for Kp_i in Kp:
        RV = Kp_i*np.sin(2.*np.pi*orbital_phase)
        
        for j in range (n_spectra):
            if not 'transmission' in temperature_profile or np.abs(orbital_phase[j]) <= half_duration_phase:
            
                temp_lnL = np.interp(drv, drv-RV[j], lnL[j, :])
                shifted_lnL[i,:] += temp_lnL
        i+=1

    goods = np.abs(drv) <= 200.

    drv = drv[goods]
    shifted_lnL = shifted_lnL[:,goods]

    return shifted_lnL, Kp, drv


def gaussian(x, a, mu, sigma):

    '''
    Inputs:
    x: x values
    a: amplitude
    mu: mean
    sigma: standard deviation

    Output:
    Gaussian function
    '''
    return a * np.exp(-(x - mu)**2 / (2 * sigma**2))


def multi_gaussian(x, *params):
    """Fit multiple Gaussians.
    
    Inputs:
    x: x values
    a: amplitude
    mu: mean
    sigma: standard deviation

    Output:
    Gaussian function

    
    Params should contain tuples of (a, mu, sigma) for each Gaussian.
    For example, for two Gaussians, params should be (a1, mu1, sigma1, a2, mu2, sigma2).
    """
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        a = params[i]
        mu = params[i+1]
        sigma = params[i+2]
        y = y + a * np.exp(-(x - mu)**2 / (2 * sigma**2))
    return y


def make_shifted_plot(snr, planet_name, observation_epoch, arm, species_name_ccf, model_tag, RV_abs, Kp_expected, V_sys_true, Kp_true, do_inject_model, do_combine, drv, Kp, species_label, temperature_profile, method, plotformat='pdf'):
    """
    Creates a shifted plot of the CCFs or likelihoods for a given planet, observation epoch, arm, and species.

    Args:
    snr (numpy.ndarray): Signal-to-noise ratio of the CCFs or likelihoods.
    planet_name (str): Name of the planet.
    observation_epoch (str): Observation epoch.
    arm (str): Spectrograph arm.
    species_name_ccf (str): Name of the species.
    model_tag (str): Model tag.
    RV_abs (astropy.units.Quantity): Absolute radial velocity.
    Kp_expected (astropy.units.Quantity): Expected Kp.
    V_sys_true (astropy.units.Quantity): True systemic velocity.
    Kp_true (astropy.units.Quantity): True Kp.
    do_inject_model (bool): Whether to inject the model.
    do_combine (bool): Whether to combine the plots.
    drv (astropy.units.Quantity): Radial velocity.
    Kp (astropy.units.Quantity): Kp.
    species_label (str): Species label.
    temperature_profile (str): Temperature profile.
    method (str): Method used to create the plot (either 'ccf' or 'likelihood').
    plotformat (str): Format of the plot file (default is 'pdf').

    Returns:
    None
    """
    
    if method == 'ccf':
        outtag, zlabel = 'CCFs-shifted', 'SNR'
        plotsnr = snr[:]
    if 'likelihood' in method:
        outtag, zlabel = 'likelihood-shifted', '$\Delta\ln \mathcal{L}$'
        plotsnr=snr - np.max(snr)
    plotname = '/home/calder/Documents/atmo-analysis-main/plots/'+ planet_name + '.' + observation_epoch + '.' + arm + '.' + species_name_ccf + model_tag + '.' + outtag + '.' + plotformat

    if do_combine:
        plotname = '/home/calder/Documents/atmo-analysis-main/plots/'+ planet_name + '.' + observation_epoch + '.' + arm + '.' + species_name_ccf + model_tag + '.' + outtag + '.' + plotformat

    if not do_inject_model:
        apoints = [unp.nominal_values(RV_abs), unp.nominal_values(Kp_expected)]
    else:
        apoints = [V_sys_true, Kp_true]

    if do_inject_model:
        model_label = 'injected'
    else:
        model_label = ''

    if 'transmission' in temperature_profile:
        ctable = 'bone'
    else:
        ctable = 'afmhot'

    keeprv = np.abs(drv-apoints[0]) <= 100.
    plotsnr, drv = plotsnr[:, keeprv], drv[keeprv]
    keepKp = np.abs(Kp-apoints[1]) <= 100.
    plotsnr, Kp = plotsnr[keepKp, :], Kp[keepKp]

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Fitting a Gaussian to the 1D slice during transit

    # Initializing lists to store fit parameters
    amps = []
    amps_err = []
    centers = []
    centers_err = []
    sigmas = []
    sigmas_err = []

    Kp_slices = []

    residuals = []
    chi2_red = []

    # Fitting gaussian to all 1D Kp slices
    for i in range(plotsnr.shape[0]):
        current_slice = plotsnr[i,:]
        Kp_slices.append(current_slice)
        popt, pcov = curve_fit(gaussian, drv, current_slice, p0=[5, -7, 1])

        amps.append(popt[0])
        centers.append(popt[1])
        sigmas.append(popt[2])

        # Storing errors (standard deviations)
        amps_err.append(np.sqrt(pcov[0, 0]))
        centers_err.append(np.sqrt(pcov[1, 1]))
        sigmas_err.append(np.sqrt(pcov[2, 2]))


    # Selecting a specific Kp slice
    selected_idx = np.where(Kp == int((np.floor(Kp_true))))[0][0]

    # Fitting a Gaussian to the selected slice
    popt_selected = [amps[selected_idx], centers[selected_idx], sigmas[selected_idx]]
    print('Selected SNR:' ,amps[selected_idx], '\n Selected Vsys:', centers[selected_idx], '\n Selected sigma:', sigmas[selected_idx])
 
    # Computing residuals and chi-squared for selected slice
    residual = plotsnr[selected_idx, :] - gaussian(drv, *popt_selected)
    # chi2 = np.sum((residual / np.std(residual))**2)/(len(drv)-len(popt))

    # Initialize Figure and GridSpec objects
    fig = pl.figure()
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])

    # Create Axes for the main plot and the residuals plot
    ax1 = pl.subplot(gs[0])
    ax2 = pl.subplot(gs[1], sharex=ax1)
    
    # Main Plot (ax1)
    ax1.plot(drv, plotsnr[selected_idx, :], 'k--', label='data', markersize=2)
    ax1.plot(drv, gaussian(drv, *popt_selected), 'r-', label='fit')
    pl.setp(ax1.get_xticklabels(), visible=False)
    ax1.set_ylabel('SNR')
    ax1.legend()

    # Add the horizontal line at 4 SNR
    ax1.axhline(y=4, color='g', linestyle='--', label=r'4 $\sigma$')    

    # Inset for residuals (ax2)
    ax2.plot(drv, residual, 'o-', markersize=1)
    ax2.set_xlabel('Velocity (km/s)')
    ax2.set_ylabel('Residuals')

    # Additional text information for the main plot
    params_str = f"Peak (a): {popt_selected[0]:.2f}\nMean (mu): {popt_selected[1]:.2f}\nSigma: {popt_selected[2]:.2f}\nKp: {Kp[selected_idx]:.0f}"
    ax1.text(0.05, 0.95, params_str, transform=ax1.transAxes, verticalalignment='top')

    # Show the plot
    pl.show()

    breakpoint()
   
    

    if arm == 'red':
        do_molecfit = True
    else:
        do_molecfit = False

    Period, epoch, M_star, RV_abs, i, M_p, R_p, RA, Dec, Kp_expected, half_duration_phase = get_planet_parameters(planet_name)

    wave, fluxin, errorin, jd, snr_spectra, exptime, airmass, n_spectra, npix = get_pepsi_data(arm, observation_epoch, planet_name, do_molecfit)

    orbital_phase = get_orbital_phase(jd, epoch, Period, RA, Dec)

    phase_min = np.min(orbital_phase)
    phase_max = np.max(orbital_phase)
    phase_array = np.linspace(phase_min, phase_max, np.shape(centers)[0])


    # Plotting velocity offset vs. orbital phase of the selected species  
    pl.figure()
    pl.errorbar(phase_array, centers, yerr=centers_err, fmt='o-', label='Center')
    pl.xlabel('Orbital Phase')
    pl.ylabel('Vsys')
    pl.title('Vsys vs. Orbital Phase')
    pl.legend()
    pl.show()

    # Fitting a curve to the velocity centers versus orbital phase
    # popt_centers, pcov_centers = curve_fit(linear, Kp, centers, p0=[0, 0]) Describe the 

    #Plotting sigma vs. orbital phase of the selected species
    #pl.figure()
   # pl.plot(drv, gaussian(drv, *popt_selected), 'r-', label='fit')

    #pl.errorbar(sigmas, centers, yerr=centers_err, fmt='o-', label='Center')
   # pl.xlabel('Orbital Phase')
    #pl.ylabel('Sigma')
    #pl.title('Sigma vs. Orbital Phase')
    #pl.legend()
    #pl.show()

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    psarr(plotsnr, drv, Kp, '$V_{\mathrm{sys}}$ (km/s)', '$K_p$ (km/s)', zlabel, filename=plotname, ctable=ctable, alines=True, apoints=apoints, acolor='cyan', textstr=species_label+' '+model_label, textloc = np.array([apoints[0]-75.,apoints[1]+75.]), textcolor='cyan', fileformat=plotformat)

    
def get_peak_snr(snr, drv, Kp, do_inject_model, V_sys_true, Kp_true, RV_abs, Kp_expected, arm, observation_epoch, f, method):
    """
    Calculates the peak signal-to-noise ratio (SNR) for a given set of data.

    Args:
    snr (numpy.ndarray): 2D array of SNR values.
    drv (numpy.ndarray): 1D array of Doppler radial velocity values.
    Kp (numpy.ndarray): 1D array of Kp values.
    do_inject_model (bool): Whether or not to inject a model.
    V_sys_true (float): True systemic velocity.
    Kp_true (float): True Kp value.
    RV_abs (astropy.units.Quantity): Absolute radial velocity.
    Kp_expected (astropy.units.Quantity): Expected Kp value.
    arm (str): Name of the arm.
    observation_epoch (str): Name of the observation epoch.
    f (file): File object to write output to.
    method (str): Method used to calculate peak SNR.

    Returns:
    None
    """
    if do_inject_model:
        boxy = np.where((drv > V_sys_true-20.) & (drv < V_sys_true+20.))
        boxx = np.where((Kp > Kp_true-20.) & (Kp < Kp_true+20.))
    else:
        boxy = np.where((drv > RV_abs.n-20.) & (drv < RV_abs.n+20.))
        boxx = np.where((Kp > Kp_expected.n-20.) & (Kp < Kp_expected.n+20.))
    boxx, boxy = boxx[0], boxy[0]

    tempsnr = snr[boxx,:]
    tempsnr2 = tempsnr[:,boxy]
    f.write('For the ' + arm + ' data for ' + observation_epoch + ': \n')
    if method == 'ccf':
        f.write('The maximum and minimum SNRs in the image are, ' + str(np.max(snr)) + ', '+ str(np.min(snr)) + ' \n')
        f.write('The peak SNR at the right location is, ' + str(np.max(tempsnr2)) + ' \n')
    if 'likelihood' in method:
        f.write('The maximum log-likelihood in the data is ' + str(np.max(snr)) + '\n')
        f.write('The best log-likelihood at the right location is, ' + str(np.max(tempsnr2)) + ' \n')
        f.write('The Delta-log-likelihood in the data is '+str(np.max(snr)-np.min(snr)))
        
    f.write(' \n')

def DopplerShadowModel(vsini, 
                        lambda_p, 
                        drv, 
                        planet_name, 
                        exptime,
                        orbital_phase,
                        obs,
                        inputs = {
                                    'mode':'spec',  
                                    'res':'medium', 
                                    'resnum': 0,    
                                    'onespec':'n',
                                    'convol':'n',
                                    'macroturb':'n',
                                    'diffrot':'n',
                                    'gravd':'n',
                                    'image':'n',
                                    'path':'y',
                                    'amode':'a',
                                    'emode':'simple',
                                    'lineshifts':'y',
                                    'starspot': False
                                    }
                        ):
        """
        Inputs: vsini - the projected rotational velocity (km/s)
                lambda_p - the spin-orbit misalignment (degrees)
                drv -  array containing the velocities at which the line profile will be calculated, in km/s, double check this
                planet_name - string containing the planet name
                exptime -  
                oribital_phase - 
                obs - string containing the observatory name
                inputs - dictionary containing the optional parameters

        """
        #add translation in the function from horus for obsnam
                #ex: if obsname = , then obsname =  pepsi/lbt
        if obs == 'keck' : Resolve=50000.0
        if obs == 'hjst' : Resolve=60000.0
        if obs == 'het' : Resolve=30000.0
        if obs == 'keck-lsi' : Resolve=20000.0
        if obs == 'subaru' : Resolve=80000.0
        if obs == 'aat' : Resolve=70000.0
        if obs == 'not' : Resolve=47000.0
        if obs == 'tres' : Resolve=44000.0
        if obs == 'harpsn' : Resolve=120000.0
        if obs == 'lbt' or obs == 'pepsi' : Resolve=120000.0
        if obs == 'igrins': Resolve=40000.0
        if obs == 'nres': Resolve=48000.0
        
        
        #get planetary parameters
        Period, epoch, M_star, RV_abs, i, M_p, R_p, RA, Dec, Kp_expected, half_duration_phase = get_planet_parameters(planet_name)
        # put into horus struc

        struc = {
                'vsini': vsini,
                'width':3.0,    # hardcoded  
                'gamma1':0.2, # linear limb-darkening coefficient, hardcoded
                'gamma2':0.2, # quadratic limb-darkening coefficient, hardcoded
                'vabsfine': drv,  # array containing the velocities at which the line profile will be calculated, in km/s, double check this
                'obs': obs,   # name of an observatory or spectrograph
                'sysname':'test',   # hardcoded
                'lineshifts':'y',   # hardcoded
            

                # Required for Doppler tomographic model
                'Pd':Period.n,    # Planetary orbital period (days)
                'lambda':lambda_p,  # spin-orbit misalignment (deg)
                'b':0.503,       # transit impact parameter, hardcoded
                'rplanet': 0.11440, # the Rp/Rstar value for the transit. hardcoded
                't':orbital_phase * Period.n * 24*60 
                , # minutes since center of transit
                'times': np.float64(exptime) * 1/60,    # exposure time (in minutes), array length must match 't'    
                'a': 7.466,     # scaled semimajor axis of the orbit, a/Rstar, hardcoded?
                'dur': half_duration_phase*2*Period.n, # duration of transit (days), optional?
                'e': 0.0,     # 
                'periarg': 90.0,   #  (deg)
                # Required for macroturbulence
                'zeta': 2.0,    # the macroturbulent velocity dispersion (km/s)

                # Required for differential rotation
                'inc': 25.0,    # the inclination of the stellar rotation axis wrt the line of sight (deg)
                'alpha': 0.1,   # differential rotation parameter
                
                # Required for gravity darkening
                #'inc':         # already used in differential rotation, comment out otherwise
                'beta': 0.1,    # gravity darkening parameter
                'Omega':7.27e-5,# stellar rotation rate (rad/s)
                'logg': 4.292,  # stellar logg
                'rstar': 1.561, # called Reqcm in docs, stellar radius (solar radii)
                'f': 1.0    ,
                'psi': 5.0      # doesn't say in docs to input this
                }
        

        # call horus
        model = horus.model(
        struc,
        inputs['mode'],
        inputs['convol'],
        inputs['gravd'],
        inputs['res'],
        inputs['resnum'],
        inputs['image'],
        inputs['onespec'],
        inputs['diffrot'],
        inputs['macroturb'],
        inputs['emode'],
        inputs['amode'],
        inputs['path'],
        inputs['lineshifts'],
        inputs['starspot']
        )

        
        # check if the below loops are even necessary
        if inputs['mode'] == 'spec':
            if inputs['onespec'] != 'y':
                profarr = model['profarr']
                basearr = model['basearr']
                baseline = model['baseline']
                z1 = model['z1']
                z2 = model['z2']    

        ccf_model = np.matrix(profarr-basearr)
       
        return ccf_model

def run_one_ccf(species_label, vmr, arm, observation_epoch, template_wave, template_flux, template_wave_in, template_flux_in, planet_name, temperature_profile, do_inject_model, species_name_ccf, model_tag, f, method, do_make_new_model):
    """
    Runs the cross-correlation function (CCF) for a given set of input parameters.

    Args:
        species_label (str): The label for the species being analyzed.
        vmr (float): The volume mixing ratio of the species being analyzed.
        arm (str): The arm of the spectrograph being used ('red' or 'blue').
        observation_epoch (str): The epoch of the observation.
        template_wave (numpy.ndarray): The wavelength array for the template spectrum.
        template_flux (numpy.ndarray): The flux array for the template spectrum.
        template_wave_in (numpy.ndarray): The wavelength array for the injected model spectrum.
        template_flux_in (numpy.ndarray): The flux array for the injected model spectrum.
        planet_name (str): The name of the planet being analyzed.
        temperature_profile (numpy.ndarray): The temperature profile for the planet being analyzed.
        do_inject_model (bool): Whether or not to inject a model into the data.
        species_name_ccf (str): The name of the species being analyzed for the CCF.
        model_tag (str): The tag for the model being used.
        f (float): The f parameter for the CCF.
        method (str): The method to use for the CCF ('ccf' or 'likelihood').
        do_make_new_model (bool): Whether or not to make a new model.

    Returns:
        None
    """
    
    niter = 10
    n_systematics = np.array(get_sysrem_parameters(arm, observation_epoch, species_label))
    ckms = 2.9979e5

    if arm == 'red':
        do_molecfit = True
    else:
        do_molecfit = False

    Period, epoch, M_star, RV_abs, i, M_p, R_p, RA, Dec, Kp_expected, half_duration_phase = get_planet_parameters(planet_name)

    wave, fluxin, errorin, jd, snr_spectra, exptime, airmass, n_spectra, npix = get_pepsi_data(arm, observation_epoch, planet_name, do_molecfit)

    orbital_phase = get_orbital_phase(jd, epoch, Period, RA, Dec)

    if do_inject_model:
        fluxin, Kp_true, V_sys_true = inject_model(Kp_expected, orbital_phase, wave, fluxin, template_wave_in, template_flux_in, n_spectra)
    else:
        Kp_true, V_sys_true = Kp_expected.n, RV_abs

    wave, flux, ccf_weights = regrid_data(wave, fluxin, errorin, n_spectra, template_wave, template_flux, snr_spectra, temperature_profile, do_make_new_model)

    #residual_flux = flux[:]
    residual_flux = flatten_spectra(flux, npix, n_spectra)

    #Make some diagnostic
    sysrem_file = '/home/calder/Documents/atmo-analysis-main/data_products/' + planet_name + '.' + observation_epoch + '.' + arm + '.SYSREM-' + str(n_systematics[0]) + '+' + str(n_systematics[1])+model_tag+'.npy'
 
    if do_sysrem:
        
        corrected_flux, corrected_error = do_sysrem(wave, residual_flux, arm, airmass, n_spectra, niter, n_systematics, do_molecfit)
        #corrected_flux, corrected_error = unp.nominal_values(residual_flux), unp.std_devs(residual_flux)

        np.save(sysrem_file, corrected_flux)
        np.save(sysrem_file+'.corrected-error.npy', corrected_error)

        plotname = '/home/calder/Documents/atmo-analysis-main/plots/' + planet_name + '.' + observation_epoch + '.' + arm + '.spectrum-SYSREM-'+ str(n_systematics[0]) + '+' + str(n_systematics[1])+'.pdf'

        psarr(corrected_flux, wave, orbital_phase, 'wavelength (Angstroms)', 'orbital phase', 'flux residual', filename=plotname,flat=True, ctable='gist_gray')

    else:
        corrected_flux = np.load(sysrem_file)
        corrected_error = np.load(sysrem_file+'.corrected-error.npy')

    if method == 'ccf':
        ccf_file = '/home/calder/Documents/atmo-analysis-main/data_products/'+ planet_name + '.' + observation_epoch + '.' + arm + '.' + species_name_ccf + model_tag + '.CCFs-raw.npy'

        drv, cross_cor, sigma_cross_cor = get_ccfs(wave, corrected_flux, corrected_error, template_wave, template_flux, n_spectra)

        
        
        np.save(ccf_file, cross_cor)
        np.save(ccf_file+'.sigma.npy', sigma_cross_cor)
        np.save(ccf_file+'.phase.npy', orbital_phase)
        np.save(ccf_file+'.ccf_weights', ccf_weights)

        #Normalize the CCFs
        for i in range (n_spectra): 
            cross_cor[i,:]-=np.mean(cross_cor[i,:])
            sigma_cross_cor[i,:] = np.sqrt(sigma_cross_cor[i,:]**2 + np.sum(sigma_cross_cor[i,:]**2)/len(sigma_cross_cor[i,:])**2)
            cross_cor[i,:]/=np.std(cross_cor[i,:])
            
        # Specifically for
        vsini = 110
        lambda_p = 0.5                  
        
        ccf_model = DopplerShadowModel(vsini, lambda_p, drv, planet_name, exptime, orbital_phase, 'pepsi')
        scales = np.arange(-100, 100, 0.01)
        
        # Create memory space for residuals and rms2
        residuals = np.zeros(ccf_model.shape)
        rms = np.zeros(len(scales))

        for k, scale in enumerate(scales):
                # Scale ccf_model and calculate residuals
                ccf_model_scaled = ccf_model * scale
                residuals = cross_cor - ccf_model_scaled
                
                # Compute rms
                rms[k] = np.sqrt(np.square((np.array(residuals))).mean())
                            
        # Identify scale factor
        scale_factor_index = np.argmin(rms)
        scale_factor = scales[scale_factor_index]

        # Scale ccf_model by scale factor
        ccf_model *= scale_factor
        cross_cor -= ccf_model

        #Make a plot
        plotname = '/home/calder/Documents/atmo-analysis-main/plots/' + planet_name + '.' + observation_epoch + '.' + species_name_ccf + model_tag + '.CCFs-raw.pdf'
        psarr(cross_cor, drv, orbital_phase, 'v (km/s)', 'orbital phase', 'SNR', filename=plotname, ctable='gist_gray')

        snr, Kp, drv = combine_ccfs(drv, cross_cor, sigma_cross_cor, orbital_phase, n_spectra, ccf_weights, half_duration_phase, temperature_profile)
        
        make_shifted_plot(snr, planet_name, observation_epoch, arm, species_name_ccf, model_tag, RV_abs, Kp_expected, V_sys_true, Kp_true, do_inject_model, True, drv, Kp, species_label, temperature_profile, method)

        get_peak_snr(snr, drv, Kp, do_inject_model, V_sys_true, Kp_true, RV_abs, Kp_expected, arm, observation_epoch, f, method)



    if 'likelihood' in method:
        like_file = '/home/calder/Documents/atmo-analysis-main/data_products/'+ planet_name + '.' + observation_epoch + '.' + arm + '.' + species_name_ccf + model_tag + '.likelihood-raw.npy'
        drv, lnL = get_likelihood(wave, corrected_flux, corrected_error, template_wave, template_flux, n_spectra)

        np.save(like_file, lnL)
        np.save(like_file+'.phase.npy', orbital_phase)

        #Make a plot
        plotname = '/home/calder/Documents/atmo-analysis-main/plots/' + planet_name + '.' + observation_epoch + '.' + species_name_ccf + model_tag + '.likelihoods-raw.pdf'
        psarr(lnL, drv, orbital_phase, 'v (km/s)', 'orbital phase', 'ln L', filename=plotname, ctable='gist_gray')

        #now need to combine the likelihoods along the planet orbit
        shifted_lnL, Kp, drv = combine_likelihoods(drv, lnL, orbital_phase, n_spectra, half_duration_phase, temperature_profile)

        make_shifted_plot(shifted_lnL, planet_name, observation_epoch, arm, species_name_ccf, model_tag, RV_abs, Kp_expected, V_sys_true, Kp_true, do_inject_model, True, drv, Kp, species_label, temperature_profile, method)


def combine_observations(observation_epochs, arms, planet_name, temperature_profile, species_label, species_name_ccf, model_tag, RV_abs, Kp_expected, do_inject_model, f, method):
    """
    Combine observations of a planet's cross-correlation function (CCF) from different epochs and arms.

    Parameters:
    observation_epochs (list): List of strings representing the observation epochs.
    arms (list): List of strings representing the arms of the spectrograph.
    planet_name (str): Name of the planet.
    temperature_profile (str): Temperature profile of the planet's atmosphere.
    species_label (str): Label of the species being analyzed.
    species_name_ccf (str): Name of the CCF species.
    model_tag (str): Tag for the model being used.
    RV_abs (float): Absolute radial velocity of the planet.
    Kp_expected (float): Expected semi-amplitude of the planet's radial velocity curve.
    do_inject_model (bool): Whether or not to inject a model into the observations.
    f (float): Scaling factor for the injected model.
    method (str): Method for combining the observations. Can be 'ccf' or 'likelihood'.

    Returns:
    None
    """
    
    Period, epoch, M_star, RV_abs, i, M_p, R_p, RA, Dec, Kp_expected, half_duration_phase = get_planet_parameters(planet_name)

    j=0
    for observation_epoch in observation_epochs:
        for arm in arms:
            if 'likelihood' in method:
                ccf_file_2 = '/home/calder/Documents/atmo-analysis-main/data_products/'+ planet_name + '.' + observation_epoch + '.' + arm + '.' + species_name_ccf + model_tag + '.likelihood-raw.npy'
            if method == 'ccf':
                ccf_file_2 = '/home/calder/Documents/atmo-analysis-main/data_products/'+ planet_name + '.' + observation_epoch + '.' + arm + '.' + species_name_ccf + model_tag + '.CCFs-raw.npy'
                sigma_cross_cor_2 = np.load(ccf_file_2+'.sigma.npy')
                ccf_weights_2 = np.load(ccf_file_2+'.ccf_weights.npy')

            cross_cor_2 = np.load(ccf_file_2)
            orbital_phase_2 = np.load(ccf_file_2+'.phase.npy')
            

            if method == 'ccf':
                #I don't think the below is right, this is just for display purposes for now
                for i in range (cross_cor_2.shape[0]): 
                    cross_cor_2[i,:]-=np.mean(cross_cor_2[i,:])
                    cross_cor_2[i,:]/=np.std(cross_cor_2[i,:])

            
    
            if j == 0:
                cross_cor, orbital_phase = cross_cor_2, orbital_phase_2
                if method == 'ccf':
                    ccf_weights, sigma_cross_cor = ccf_weights_2, sigma_cross_cor_2
            else:
                cross_cor = np.append(cross_cor, cross_cor_2, axis=0)
                orbital_phase = np.append(orbital_phase, orbital_phase_2)
                if method == 'ccf':
                    sigma_cross_cor = np.append(sigma_cross_cor, sigma_cross_cor_2, axis=0)
                    ccf_weights = np.append(ccf_weights, ccf_weights_2)

            j+=1

    rvmin, rvmax = -400., 400. #kms
    rvspacing = 1.0 #kms
    drv = np.arange(rvmin, rvmax, rvspacing)

    if do_inject_model:
        Kp_true, V_sys_true = unp.nominal_values(Kp_expected), 0.0
    else:
        Kp_true, V_sys_true = Kp_expected.n, RV_abs

    if method == 'ccf':
        snr, Kp, drv = combine_ccfs(drv, cross_cor, sigma_cross_cor, orbital_phase, len(orbital_phase), ccf_weights, half_duration_phase, temperature_profile)
    if 'likelihood' in method:
        snr, Kp, drv = combine_likelihoods(drv, cross_cor, orbital_phase, len(orbital_phase), half_duration_phase, temperature_profile)

    if any('red' in s for s in arms) and ('red' in s for s in arms):
        all_arms = 'combined'
    else:
        all_arms = arms[0]

    all_epochs = observation_epochs[0]
    if len(observation_epochs) > 1:
        for i in range (1, len(observation_epochs)):
            all_epochs += '+'+observation_epochs[i]

    make_shifted_plot(snr, planet_name, all_epochs, all_arms, species_name_ccf, model_tag, RV_abs, Kp_expected, V_sys_true, Kp_true, do_inject_model, True, drv, Kp, species_label, temperature_profile, method)

    #import pdb; pdb.set_trace()

    get_peak_snr(snr, drv, Kp, do_inject_model, V_sys_true, Kp_true, RV_abs, Kp_expected, all_arms, all_epochs, f, method)
    
                
            

def run_all_ccfs(planet_name, temperature_profile, species_label, vmr, do_inject_model, do_run_all, do_make_new_model, method):
    """
    Runs all cross-correlation functions (CCFs) for a given planet, temperature profile, and species label.

    Args:
    planet_name (str): Name of the planet.
    temperature_profile (str): Type of temperature profile (e.g. 'hot', 'cold', 'transmission', 'emission').
    species_label (str): Label of the species to analyze (e.g. 'FeH', 'CaH').
    vmr (float): Volume mixing ratio of the species.
    do_inject_model (bool): Whether or not to inject the model.
    do_run_all (bool): Whether or not to run all observations.
    do_make_new_model (bool): Whether or not to make a new model.
    method (str): Method to use for the CCF (e.g. 'Gaussian')

    Returns:
    None
    """
    initial_time=time.time()
    ckms = 2.9979e5

    instrument = 'PEPSI'

    if do_inject_model:
        model_tag = '.injected-'+str(vmr)
    else:
        model_tag = ''

    if 'transmission' in temperature_profile:
        spectrum_type = 'transmission'
        if planet_name == 'KELT-20b': observation_epochs = ['20190504']
    else:
        spectrum_type = 'emission'
        if planet_name == 'KELT-20b': observation_epochs = ['20210501', '20210518']
        if planet_name == 'WASP-12b': observation_epochs = ['20210303', '20220208']
        if planet_name == 'KELT-9b': observation_epochs = ['20210628']
        if planet_name == 'WASP-76b': observation_epochs = ['20211031']
        if planet_name == 'WASP-33b': observation_epochs = ['20220929', '20221202']
        if planet_name == 'WASP-189b': observation_epochs = ['20230327']


    if species_label == 'FeH':
        arms = ['red']
    elif species_label == 'CaH':
        arms = ['blue']
    else:
        arms = ['blue','red']
            
    species_name_inject, species_name_ccf = get_species_keys(species_label)
    
    file_out = '/home/calder/Documents/atmo-analysis-main/logs/'+ planet_name + '.' + species_name_ccf + model_tag + '.log' #edited for my own machine
    f = open(file_out,'w')

    f.write('Log file for ' + planet_name + ' for ' + species_name_ccf + ' \n')

    if do_make_new_model:
        template_wave, template_flux = make_new_model(instrument, species_name_ccf, vmr, spectrum_type, planet_name, temperature_profile, do_plot=True)
    else:
        template_wave, template_flux = get_atmospheric_model(planet_name, species_name_ccf, vmr, temperature_profile, True, True)

    if species_name_ccf != species_name_inject:
        if do_make_new_model:
            template_wave_in, template_flux_in = make_new_model(instrument, species_name_inject, vmr, spectrum_type, planet_name, temperature_profile)
        else:
            template_wave_in, template_flux_in = get_atmospheric_model(planet_name, species_name_inject, vmr, temperature_profile, True, True)
    else:
        template_wave_in, template_flux_in = template_wave, template_flux

    if do_run_all:
        for observation_epoch in observation_epochs:
            for arm in arms:
                print('Now running the ',arm,' data for ',observation_epoch)
                run_one_ccf(species_label, vmr, arm, observation_epoch, template_wave, template_flux, template_wave_in, template_flux_in, planet_name, temperature_profile, do_inject_model, species_name_ccf, model_tag, f, method, do_make_new_model)

    print('Now combining all of the data')

    Period, epoch, M_star, RV_abs, i, M_p, R_p, RA, Dec, Kp_expected, half_duration_phase = get_planet_parameters(planet_name)

    
    
    combine_observations(observation_epochs, arms, planet_name, temperature_profile, species_label, species_name_ccf, model_tag, RV_abs, Kp_expected, do_inject_model, f, method)

    
    f.close()
    