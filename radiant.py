import numpy as np
import petitRADTRANS.nat_cst as nc
from petitRADTRANS import Radtrans
from astropy import units as u
from astropy.io import ascii, fits

import matplotlib.pyplot as pl
import matplotlib.gridspec as gridspec
from glob import glob

from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
from astropy.table import Table

from uncertainties import ufloat
from uncertainties import unumpy as unp

import time

from dtutils import psarr

from create_model import create_model, instantiate_radtrans
from atmo_utilities import *
from run_all_ccfs import *

import emcee
import argparse
import os
import horus

from matplotlib.backends.backend_pdf import PdfPages    
import radvel
from scipy.optimize import curve_fit


pl.rc('font', size=14) #controls default text size
pl.rc('axes', titlesize=14) #fontsize of the title
pl.rc('axes', labelsize=14) #fontsize of the x and y labels
pl.rc('xtick', labelsize=14) #fontsize of the x tick labels
pl.rc('ytick', labelsize=14) #fontsize of the y tick labels
pl.rc('legend', fontsize=14) #fontsize of the legend

# global varaibles defined for harcoded path to data on my computer
path_modifier_plots = '/home/calder/Documents/atmo-analysis-main/'  #linux
path_modifier_data = '/home/calder/Documents/petitRADTRANS_data/'   #linux
path_modifier_plots = '/Users/calder/Documents/atmo-analysis-main/' #mac
path_modifier_data = '/Volumes/sabrent/petitRADTRANS_data/'  #mac
path_modifier_data = '/Users/calder/Documents/petitRADTRANS_data/' #mac

def get_species_keys(species_label):
    species_names = set()  # Create a set to store unique species names

    if species_label == 'TiO':
        species_names.add('TiO_all_iso_Plez')
    if species_label == 'TiO_46':
        species_names.add('TiO_46_Exomol_McKemmish')
    if species_label == 'TiO_47':
        species_names.add('TiO_47_Exomol_McKemmish')
    if species_label == 'TiO_48':
        species_names.add('TiO_48_Exomol_McKemmish')  
    if species_label == 'TiO_49':
        species_names.add('TiO_49_Exomol_McKemmish')
    if species_label == 'TiO_50':
        species_names.add('TiO_50_Exomol_McKemmish')
    if species_label == 'VO':
        species_names.add('VO_ExoMol_McKemmish')
    if species_label == 'FeH':
        species_names.add('FeH_main_iso')
    if species_label == 'CaH':
        species_names.add('CaH')
    if species_label == 'Fe I':
        species_names.add('Fe')
    if species_label == 'Ti I':
        species_names.add('Ti')
    if species_label == 'Ti II':
        species_names.add('Ti+')
    if species_label == 'Mg I':
        species_names.add('Mg')
    if species_label == 'Mg II':
        species_names.add('Mg+')
    if species_label == 'Fe II':
        species_names.add('Fe+')
    if species_label == 'Cr I':
        species_names.add('Cr')
    if species_label == 'Si I':
        species_names.add('Si')
    if species_label == 'Ni I':
        species_names.add('Ni')
    if species_label == 'Al I':
        species_names.add('Al')
    if species_label == 'SiO':
        species_names.add('SiO_main_iso_new_incl_UV')
    if species_label == 'H2O':
        species_names.add('H2O_main_iso')
    if species_label == 'OH':
        species_names.add('OH_main_iso')
    if species_label == 'MgH':
        species_names.add('MgH')
    if species_label == 'Ca I':
        species_names.add('Ca')
    if species_label == 'CO_all':
        species_names.add('CO_all_iso')
    if species_label == 'CO_main':
        species_names.add('CO_main_iso')    
    if species_label == 'NaH':
        species_names.add('NaH')
    if species_label == 'H I':
        species_names.add('H')
    if species_label == 'AlO':
        species_names.add('AlO')
    if species_label == 'Ba I':
        species_names.add('Ba')
    if species_label == 'Ba II':
        species_names.add('Ba+')
    if species_label == 'CaO':
        species_names.add('CaO')
    if species_label == 'Co I':
        species_names.add('Co')
    if species_label == 'Cr II':
        species_names.add('Cr+')
    if species_label == 'Cs I':
        species_names.add('Cs')
    if species_label == 'Cu I':
        species_names.add('Cu') 
    if species_label == 'Ga I':
        species_names.add('Ga') 
    if species_label == 'Ge I': 
        species_names.add('Ge')
    if species_label == 'Hf I':
        species_names.add('Hf')
    if species_label == 'In I':
        species_names.add('In') 
    if species_label == 'Ir I':
        species_names.add('Ir') 
    if species_label == 'Mn I':
        species_names.add('Mn') 
    if species_label == 'Mo I':
        species_names.add('Mo')
    if species_label == 'Na I':
        species_names.add('Na')
    if species_label == 'NaH':
        species_names.add('NaH')
    if species_label == 'Nb I':
        species_names.add('Nb')
    if species_label == 'Ni I':
        species_names.add('Ni')
    if species_label == 'O I':
        species_names.add('O')
    if species_label == 'Os I':
        species_names.add('Os')
    if species_label == 'Pb I':
        species_names.add('Pb')
    if species_label == 'Pd I':
        species_names.add('Pd')
    if species_label == 'Rb I':
        species_names.add('Rb')
    if species_label == 'Rh I':
        species_names.add('Rh')
    if species_label == 'Ru I':
        species_names.add('Ru')
    if species_label == 'Sc I':
        species_names.add('Sc')
    if species_label == 'Sc II':
        species_names.add('Sc+')
    if species_label == 'Sn I':
        species_names.add('Sn')
    if species_label == 'Sr I':
        species_names.add('Sr')
    if species_label == 'Sr II':
        species_names.add('Sr+')
    if species_label == 'Ti II':
        species_names.add('Ti+')
    if species_label == 'Tl':
        species_names.add('Tl')
    if species_label == 'W I':
        species_names.add('W')
    if species_label == 'Y II':
        species_names.add('Y+')
    if species_label == 'Zn I':
        species_names.add('Zn')
    if species_label == 'Zr I':
        species_names.add('Zr')
    if species_label == 'Zr I':
        species_names.add('Zr+')
    if species_label == 'N I':
        species_names.add('N')
    if species_label == 'K I':
        species_names.add('K')
    if species_label == 'Y I':
        species_names.add('Y')
    if species_label == 'Li I':
        species_names.add('Li')
    if species_label == 'V I':
        species_names.add('V')
    if species_label == 'V II':
        species_names.add('V+')
    if species_label == 'Ca II':
        species_names.add('Ca+')

    species_names = list(set(species_names))  # Convert the set back to a list
    species_names.append(species_names[0])
    species_name_inject, species_name_ccf = species_names
        
    return species_name_inject, species_name_ccf

def get_sysrem_parameters(arm, observation_epoch, species_label, planet_name):
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
    elif species_label == 'Fe I' and planet_name == 'KELT-9 b':
        if arm == 'red': n_systematics = [5, 5]
        if arm == 'blue': n_systematics = [6, 0]
    if species_label == 'TiO':
        if arm == 'red': n_systematics = [1, 1]
        if arm == 'blue': n_systematics = [2, 0]
    elif species_label == 'Ni I':
        if arm == 'red': n_systematics = [2,3]
        if arm == 'blue': n_systematics = [5,0]
    elif species_label == 'Cr I':
        if arm == 'red': n_systematics = [0,10]
        if arm == 'blue': n_systematics = [10,0]
    elif species_label == 'V I':
        if arm == 'red': n_systematics = [2,5]
        if arm == 'blue': n_systematics = [8,0]
    elif species_label == 'VO':
        if arm == 'red': n_systematics = [1, 2]
        if arm == 'blue': n_systematics = [3, 0]
    elif species_label == 'FeH':
        if arm == 'red': n_systematics = [1, 0]
        if arm == 'blue': n_systematics = [5, 0]
    elif species_label == 'CaH':
        if arm == 'red': n_systematics = [0, 10]
        if arm == 'blue': n_systematics = [2, 0]
    elif species_label == 'Na I':
        if arm == 'red': n_systematics = [0, 10]
        if arm == 'blue': n_systematics = [1,0]
    elif species_label == 'Mn I':
        if arm == 'red': n_systematics = [2,2]
        if arm == 'blue': n_systematics = [1,0]
     
    else:
        if arm == 'blue':
            n_systematics = [3, 0]
        if arm == 'red':
            n_systematics = [1, 1]

    if planet_name == 'KELT-9b':
        if arm == 'blue':
            n_systematics = [20, 0]
        if arm == 'red':
            n_systematics = [20, 20]

    if planet_name == 'TOI-1518b':
        if arm == 'blue':
            n_systematics = [4, 0]
        if arm == 'red':
            n_systematics = [3, 1]

    if planet_name == 'TOI-1431b':
        if arm == 'blue':
            n_systematics = [3, 0]
        if arm == 'red':
            n_systematics = [1, 1]

    return n_systematics

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

        Ks_expected = 0.0
        
    
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

        Ks_expected = 0.0
    
    if planet_name == 'KELT-9b':
        #For KELT-9 b:, from Gaudi et al. 2017 and Pai Asnodkar et al. 2021
        Period = ufloat(1.4811235, 0.0000011) #days
        epoch = ufloat(2457095.68572, 0.00014) #BJD_TDB

        M_star = ufloat(2.11, 0.78) #MSun
        RV_abs = ufloat(-37.11, 1.0) #km/s
        i = ufloat(86.79, 0.25) #degrees
        M_p = ufloat(2.17, 0.56)
        R_p = 1.891

        RA = '20h31m26.38s'
        Dec = '+39d56m20.10s'

        dur = 3.9158/24.

        Ks_expected = 0.0

    if planet_name == 'WASP-12b':
        #For WASP-12 b:, from Ivishina & Winn 2022, Bonomo+17, Charkabarty & Sengupta 2019
        Period = ufloat(1.091419108, 5.5e-08) #days
        epoch = ufloat(2457010.512173, 7e-05) #BJD_TDB

        M_star = ufloat(1.38, 0.18) #MSun
        RV_abs = ufloat(0.0, 0.0) #km/s
        i = ufloat(83.3, 1.1) #degrees
        M_p = ufloat(1.39, 0.12)
        R_p = 1.937

        RA = '06h30m32.79s'
        Dec = '+29d40m20.16s'

        dur = 3.0408/24.

        Ks_expected = 0.0

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

        Ks_expected = 0.0

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

        Ks_expected = 0.0

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

        Ks_expected = 0.0

    if planet_name == 'MASCARA-1b':
        
        Period = ufloat(2.14877381, 0.00000088) #days
        epoch = ufloat(2458833.488151, 0.000092) #BJD_TDB

        M_star = ufloat(1.900, 0.068) #MSun
        RV_abs = ufloat(0.0, 0.0) #km/s
        i = ufloat(88.45, 0.17) #degrees
        M_p = ufloat(3.7, 0.9)
        R_p = 1.597

        RA = '21h10m12.37s'
        Dec = '+10d44m20.03s'

        dur = 4.226/24.

        Ks_expected = 0.0

    if planet_name == 'TOI-1431b':
        
        Period = ufloat(2.650237, 0.000003) #days
        epoch = ufloat(2458739.17737, 0.00007) #BJD_TDB

        M_star = ufloat(1.90, 0.10) #MSun
        RV_abs = ufloat(0.0, 0.0) #km/s
        i = ufloat(80.13, 0.13) #degrees
        M_p = ufloat(3.12, 0.18)
        R_p = 1.49

        RA = '21h04m48.89s'
        Dec = '+55d35m16.88s'

        dur = 2.489/24.
        Ks_expected = 294.1 #m/s


    if planet_name == 'TOI-1518b':
        
        Period = ufloat(1.902603, 0.000011) #days
        epoch = ufloat(2458787.049255, 0.000094) #BJD_TDB

        #this is the updated ephemeris from Alison
        Period = ufloat(1.9026055, 0.0000066) #days
        epoch = ufloat(2459854.41433, 0.00012) #BJD_TDB

        M_star = ufloat(1.79, 0.26) #MSun
        RV_abs = ufloat(0.0, 0.0) #km/s
        i = ufloat(77.84, 0.26) #degrees
        M_p = ufloat(2.3, 2.3)
        R_p = 1.875

        RA = '23h29m04.20s'
        Dec = '+67d02m05.30s'

        dur = 2.365/24.
        Ks_expected = 0.0 

    half_duration_phase = (dur/2.)/Period.n
    Kp_expected = 28.4329 * M_star/MJoMS * unp.sin(i*np.pi/180.) * (M_star + M_p * MJoMS) ** (-2./3.) * (Period/365.25) ** (-1./3.) / 1000. #to km/s
    half_duration_phase = (dur/2.)/Period.n

    return Period, epoch, M_star, RV_abs, i, M_p, R_p, RA, Dec, Kp_expected, half_duration_phase, Ks_expected

def get_wavelength_range(instrument_here):
    if instrument_here == 'PEPSI-35': lambda_low, lambda_high = 4750., 7500.
    if instrument_here == 'PEPSI-25': lambda_low, lambda_high = 4250., 7500.
    if instrument_here == 'MaroonX': lambda_low, lambda_high = 5000., 9200.
    if instrument_here == 'full-range': lambda_low, lambda_high = 3800., 37000.
    if instrument_here == 'GMT-all-optical': lambda_low, lambda_high = 3500., 10000.
    if instrument_here == 'GMT-all-infrared': lambda_low, lambda_high = 10000., 53000.
    if instrument_here == 'G-CLEF': lambda_low, lambda_high = 3500., 9500.
    if instrument_here == 'GMTNIRS': lambda_low, lambda_high = 10700., 54000.
    return lambda_low, lambda_high
    
def get_pepsi_data(arm, observation_epoch, planet_name, do_molecfit):

    ckms = 2.9979e5
    #get data
    if arm == 'blue':
        arm_file = 'pepsib'
    if arm == 'red':
        arm_file = 'pepsir'
    
    
    #change 'avr' to 'nor' below for more recent data
    if observation_epoch != 'mock-obs':
        if float(observation_epoch[0:4]) >= 2019 and float(observation_epoch[0:4]) <= 2023:
            pepsi_extend = 'nor'
        elif float(observation_epoch[0:4]) >= 2024:
            pepsi_extend = 'bwl'
    else:
        pepsi_extend = 'avr'
        
    if not do_molecfit:
        data_location = path_modifier_data + 'data/' + observation_epoch + '_' + planet_name + '/' + arm_file + '*.dxt.' + pepsi_extend
    else:
        data_location = path_modifier_data + 'data/' + observation_epoch + '_' + planet_name + '/molecfit_weak/SCIENCE_TELLURIC_CORR_' + arm_file + '*.dxt.' + pepsi_extend + '.fits'
    spectra_files = glob(data_location)

    n_spectra = len(spectra_files)
    i=0
    jd, snr_spectra, exptime = np.zeros(n_spectra), np.zeros(n_spectra), np.zeros(n_spectra)
    airmass = np.zeros(n_spectra)
    for spectrum in spectra_files:

        hdu = fits.open(spectrum)
        data, header = hdu[1].data, hdu[0].header
        if do_molecfit: wave_tag, flux_tag, error_tag = 'lambda', 'flux', 'error'
        if not do_molecfit: wave_tag, flux_tag, error_tag = 'Arg', 'Fun', 'Var'
        if i ==0:
            npix, npixmin = len(data[wave_tag]), len(data[wave_tag])
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
            npixmin = npixhere * 1.0

        #molecfit_utilities already handles variance->uncertainty
        if not do_molecfit:
            errorin[i,:]=np.sqrt(errorin[i,:])
            total_velocity = 0.0
            #if planet_name == 'KELT-20b':
            #    total_velocity = 3.2234 * 1000.

            #    doppler_shift = 1.0 / (1.0 - total_velocity / 1000. / ckms)
                #wave[i,:] *= doppler_shift
            
        if do_molecfit:
            wave[i,:]*=10000. #microns -> Angstroms
        
            #remove shift introduced to make Molecfit work
            if observation_epoch == '20210501': introduced_shift = 6000.
            if observation_epoch == '20210518': introduced_shift = 3500.
            if observation_epoch == '20190425': introduced_shift = 464500.
            if observation_epoch == '20190504': introduced_shift = 6300.
            if observation_epoch == '20190515': introduced_shift = 506000.
            if observation_epoch == '20190622': introduced_shift = -54300.
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
            if observation_epoch == '20230430': introduced_shift = -19900.
            if observation_epoch == '20220925': introduced_shift = -117200.
            if observation_epoch == '20230615': introduced_shift = -32400.
            if observation_epoch == '20231023': introduced_shift = -97000.
            if observation_epoch == '20231106': introduced_shift = -84700.
            if observation_epoch == '20240114': introduced_shift = -112100.
            if observation_epoch == '20220926': introduced_shift = -65000.
            if observation_epoch == '20240312': introduced_shift = -75200.

            #if pepsi_extend == 'nor':

            total_velocity = introduced_shift + 0.0
            if any('RADVEL' in s for s in hdu[0].header['HISTORY']):
                try:
                    total_velocity += hdu[0].header['RADVEL']
                except KeyError:
                    total_velocity += 0.0
            if any('OBSVEL' in s for s in hdu[0].header):
                total_velocity += hdu[0].header['OBSVEL']
            if any('SSBVEL' in s for s in hdu[0].header):
                total_velocity += hdu[0].header['SSBVEL']
                
        if planet_name == 'KELT-20b': total_velocity += 3.2234 * 1000.
        if planet_name == 'TOI-1431b': total_velocity += 24.903 * 1000.
        if planet_name == 'TOI-1518b': total_velocity += 11.170 * 1000.
        
            #if i != 0:
            #    drv, cross_cor_out, sigma_cross_cor_out = ccf(wave[0,10000:20000], fluxin[0,10000:20000], errorin[0,10000:20000], wave[i,10000:20000], fluxin[i,10000:20000], -15., 15, 0.01)
            #    total_velocity -= drv[np.argmax(cross_cor_out)] * 1000.

        doppler_shift = 1.0 / (1.0 - total_velocity / 1000. / ckms)
        
            #    try:
            #        doppler_shift = 1.0 / (1.0 - (hdu[0].header['RADVEL'] + hdu[0].header['OBSVEL'] + introduced_shift) / 1000. / ckms)
            #    except KeyError:
            #        doppler_shift = 1.0 / (1.0 - (hdu[0].header['RADVEL'] + hdu[0].header['SSBVEL'] + introduced_shift) / 1000. / ckms)
            #else:
            #    doppler_shift = 1.0 / (1.0 - (hdu[0].header['OBSVEL'] + introduced_shift) / 1000. / ckms) #note: old data does not correct to the stellar frame, only the barycentric

        wave[i,:] *= doppler_shift

        jd[i] = header['JD-OBS']
            
        try:
            snr_spectra[i] = header['SNR']
        except KeyError:
            snr_spectra[i] = np.percentile(fluxin[i,:]/errorin[i,:], 90)
            
        if isinstance(hdu[0].header['EXPTIME'], str):
            exptime_strings = header['EXPTIME'].split(':') #in decimal h/m/s. WHYYYYYY
            exptime[i] = float(exptime_strings[0]) * 3600. + float(exptime_strings[1]) * 60. + float(exptime_strings[2])
        else:
            exptime[i] = header['EXPTIME']
        airmass[i] = header['AIRMASS']

        hdu.close()
        i+=1

    #glob gets the files out of order for some reason so we have to put them in time order
    obs_order = np.argsort(jd)

    jd, snr_spectra, exptime, airmass = jd[obs_order], snr_spectra[obs_order], exptime[obs_order], airmass[obs_order]

    

    wave, fluxin, errorin = wave[obs_order,:], fluxin[obs_order,:], errorin[obs_order,:]

    #protect against single missing pixels at the end:
    npixmin = int(npixmin)
    if npixmin < npix:
        wave, fluxin, errorin = wave[:,:npixmin], fluxin[:,:npixmin], errorin[:,:npixmin]
        npix = npixmin

    

    #This is approximate, to account for a small underestimate in the error by the pipeline, and at least approximately include the systematic effects due to Molecfit
    #error_estimated = np.zeros_like(fluxin)
    #for i in range (0, n_spectra):
    #    for j in range (10, npix-10):
    #        error_estimated[i,j] = np.std(fluxin[i,j-10:j+10])
    #    underestimate_factor = np.nanmedian(error_estimated[i,10:npix-10]/errorin[i,10:npix-10])
    #    errorin[i,:] *= underestimate_factor

    
    return wave, fluxin, errorin, jd, snr_spectra, exptime, airmass, n_spectra, npix

def get_orbital_phase(jd, epoch, Period, RA, Dec):

    lbt_coordinates = EarthLocation.of_site('lbt')

    observed_times = Time(jd, format='jd', location=lbt_coordinates)

    coordinates = SkyCoord(RA+' '+Dec, frame='icrs', unit=(u.hourangle, u.deg))

    ltt_bary = observed_times.light_travel_time(coordinates)

    bary_times = observed_times + ltt_bary

    orbital_phase = (bary_times.value - epoch)/Period
    orbital_phase -= np.round(np.mean(unp.nominal_values(orbital_phase)))
    orbital_phase = unp.nominal_values(orbital_phase)

    negatives = orbital_phase < -0.15
    orbital_phase[negatives] += 1.0

    return orbital_phase

def convolve_atmospheric_model(template_wave, template_flux, profile_width, profile_form, temperature_profile='emission', epsilon=0.6):
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

def do_convolutions(planet_name, template_wave, template_flux, do_rotate, do_instrument, temperature_profile, Resolve = 130000.): #Resolve defaults to PEPSI

    if do_rotate:
        epsilon = 0.6
        Period, epoch, M_star, RV_abs, i, M_p, R_p, RA, Dec, Kp_expected, half_duration_phase, Ks_expected = get_planet_parameters(planet_name)
        #the below assumes that the planet is tidally locked
        rotational_velocity = 2. * np.pi * R_p * 69911. / (Period.n * 24. *3600.)
        template_flux = convolve_atmospheric_model(template_wave, template_flux, rotational_velocity, 'rotational', temperature_profile=temperature_profile, epsilon=epsilon)

    if do_instrument:
        ckms = 2.9979e5
        sigma = ckms / Resolve / 2. #assuming that resolving power described the width of the line

        template_flux = convolve_atmospheric_model(template_wave, template_flux, sigma, 'gaussian')

    return template_flux
    

def make_spectrum_plot(template_wave, template_flux, planet_name, species_name_ccf, temperature_profile, vmr):

    if planet_name == 'WASP-189b' or planet_name == 'KELT-20b':
        pl.fill([4265,4265,4800,4800],[np.nanmin(template_flux),np.nanmax(template_flux),np.nanmax(template_flux),np.nanmin(template_flux)],color='cyan',alpha=0.25)
    if planet_name != 'WASP-189b':
        pl.fill([4800,4800,5441,5441],[np.nanmin(template_flux),np.nanmax(template_flux),np.nanmax(template_flux),np.nanmin(template_flux)],color='blue',alpha=0.25)
    pl.fill([6278,6278,7419,7419],[np.nanmin(template_flux),np.nanmax(template_flux),np.nanmax(template_flux),np.nanmin(template_flux)],color='red',alpha=0.25)

    pl.plot(template_wave, template_flux, color='black')

    pl.xlabel('wavelength (Angstroms)')

    pl.ylabel('normalized flux')

    pl.title(species_name_ccf)

    plotout = path_modifier_plots+'plots/spectrum.' + planet_name + '.' + species_name_ccf + '.' + str(vmr) + '.' + temperature_profile + '.pdf'
    pl.savefig(plotout,format='pdf')
    pl.clf()

def make_new_model(instrument, species_name_new, vmr, spectrum_type, planet_name, temperature_profile, do_plot=False):

    if instrument == 'PEPSI':
        if (planet_name == 'WASP-189b' or planet_name == 'KELT-20b'):
            instrument_here = 'PEPSI-25'
        else:
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

    elif planet_name == 'KELT-9b':
        parameters['kappa'] = 0.01
        parameters['gamma'] = 50.
        parameters['Teq'] = 3500.
        ptprofile = 'guillot'

    elif planet_name == 'MASCARA-1b':
        parameters['kappa'] = 0.01
        parameters['gamma'] = 50.
        parameters['Teq'] = 2594.
        ptprofile = 'guillot'

    elif planet_name == 'TOI-1431b':
        parameters['kappa'] = 0.005
        parameters['gamma'] = 30.
        parameters['Teq'] = 2370.
        ptprofile = 'guillot'

    elif planet_name == 'TOI-1518b':
        parameters['kappa'] = 0.04 #0.02 #0.04 #
        parameters['gamma'] = 45. #30 #15. #
        parameters['Teq'] = 2492.-200. #getting best results with this Teq
        ptprofile = 'guillot'
   
    else:
        #fill these in later
        parameters['kappa'] = 0.01
        parameters['gamma'] = 50.
        ptprofile = 'guillot'

    
    
    parameters[species_name_new] = vmr
    template_wave, template_flux = generate_atmospheric_model(planet_name, spectrum_type, instrument, 'combined', [species_name_new], parameters, atmosphere, pressures, ptprofile = ptprofile)

    

    template_flux = do_convolutions(planet_name, template_wave, template_flux, True, True, temperature_profile)

    #if 'Plez' in species_name_new or species_name_new == 'Fe+' or species_name_new == 'Ti+': # or species_name_new == 'Cr':
        #template_wave = vacuum2air(template_wave)
        #template_wave = air2vacuum(template_wave)

    if do_plot: make_spectrum_plot(template_wave, template_flux, planet_name, species_name_new, temperature_profile, vmr)

    

    return template_wave, template_flux

def get_atmospheric_model(planet_name, species_name_ccf, vmr, temperature_profile, do_rotate, do_instrument):

    filein = 'templates/' + planet_name + '.' + species_name_ccf + '.' + str(vmr) + '.' + temperature_profile + '.combined.fits'
    hdu = fits.open(filein)

    template_wave = hdu[1].data['wave']
    template_flux = hdu[1].data['flux']

    hdu.close()

    template_flux = do_convolutions(planet_name, template_wave, template_flux, do_rotate, do_instrument, temperature_profile)

    return template_wave, template_flux


def correct_for_reflex_motion(Ks_expected, orbital_phase, wave, n_spectra):

    ckms = 2.9979e5

    RV = Ks_expected*np.sin(2.*np.pi*(orbital_phase-0.5)) 

    doppler_shift = 1.0 / (1.0 - RV / 1000. / ckms)
    for i in range (0, n_spectra): wave[i,:] *= doppler_shift[i]

    return wave

def inject_model(Kp_expected, orbital_phase, wave, fluxin, template_wave_in, template_flux_in, n_spectra):

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

def regrid_data(wave, fluxin, errorin, n_spectra, template_wave, template_flux, snr_spectra, temperature_profile, do_make_new_model):
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

    U_sysrem = np.ones((n_spectra, np.max(n_systematics), chunks))
    
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

            U_sysrem[:,system,chunk] = a
                    
            
            #remove systematic error
            corrected_flux[:,this_one] -= syserr
            corrected_error[:,this_one] = np.sqrt(corrected_error[:,this_one]**2 + sigma_syserr**2)

            return corrected_flux, corrected_error, U_sysrem, no_tellurics

def get_sysrem_invariant_matrix(wave, corrected_flux, corrected_error, template_wave, template_flux, U_sysrem, telluric_free, n_spectra):

    Lambda = np.zeros((n_spectra, n_spectra))

    M = np.zeros((n_spectra, len(template_wave)))

    for i in range (0, n_spectra):
        Lambda[i,i] = 1. / np.mean(corrected_error[i,:])
        M[i,:] = template_flux

    LambdaU = np.linalg.matmul(Lambda, U_sysrem)
    LambdaUt = np.linalg.pinv(LambdaU)
    ULambdaU = np.linalg.matmul(U_sysrem, LambdaUt)

    ULambdaULambda = np.linalg.matmul(ULambdaU, Lambda)

    return ULambdaULambda

def sysrem_correct_model(wave, corrected_flux, corrected_error, template_wave, template_flux, U_sysrem, telluric_free, n_spectra, ULambdaULambda):


    M = np.zeros((n_spectra, len(template_wave)))

    for i in range (0, n_spectra):
        M[i,:] = template_flux

    Mprime = np.linalg.matmul(ULambdaULambda, M)

    return Mprime

    
        

def get_ccfs(wave, corrected_flux, corrected_error, template_wave, template_flux, n_spectra, U_sysrem, telluric_free):

    #rvmin, rvmax = -100., 100. #kms
    rvmin, rvmax = -400., 400. #kms
    rvspacing = 1.0 #kms
    for i in range (n_spectra):
        drv, cross_cor_out, sigma_cross_cor_out = ccf(wave, corrected_flux[i,:], corrected_error[i,:], template_wave, template_flux, rvmin, rvmax, rvspacing)
        

        if i == 0:
            cross_cor, sigma_cross_cor = np.zeros((n_spectra, len(drv))), np.zeros((n_spectra, len(drv)))
        cross_cor[i,:], sigma_cross_cor[i,:] = cross_cor_out, sigma_cross_cor_out
    return drv, cross_cor, sigma_cross_cor

def get_likelihood(wave, corrected_flux, corrected_error, template_wave, template_flux, n_spectra, U_sysrem, telluric_free):

    rvmin, rvmax = -400., 400. #kms
    rvspacing = 0.01 #kms

    alpha, beta, norm_offset = 1.0, 1.0, 0.0 #I think this is correct for this application--only set these scaling factors if do actual fits

    for i in range (n_spectra):
        #drv, lnL0 = log_likelihood_CCF(wave, corrected_flux[i,:], corrected_error[i,:], template_wave, template_flux, rvmin, rvmax, rvspacing, alpha, beta)
        drv, lnL0 = log_likelihood_opt_beta(wave, corrected_flux[i,:], corrected_error[i,:], template_wave, template_flux, rvmin, rvmax, rvspacing, alpha, norm_offset)
        if i == 0:
            lnL = np.zeros((n_spectra, len(drv)))
        lnL[i,:] = lnL0

    return drv, lnL

def combine_ccfs(drv, cross_cor, sigma_cross_cor, orbital_phase, n_spectra, ccf_weights, half_duration_phase, temperature_profile):

    Kp = np.arange(50, 350, 1)
    nKp, nv = len(Kp), len(drv)

    shifted_ccfs, var_shifted_ccfs = np.zeros((nKp, nv)), np.zeros((nKp, nv))

    i = 0
    for Kp_i in Kp:
        RV = Kp_i*np.sin(2.*np.pi*orbital_phase)
        
        for j in range (n_spectra):
            #restrict to only in-transit spectra if doing transmission:
            #also want to leave out observations in 2ndary eclipse!

            if not 'transmission' in temperature_profile or np.abs(orbital_phase[j]) <= half_duration_phase or np.abs(orbital_phase[j]-0.5) >= half_duration_phase:
                temp_ccf = np.interp(drv, drv-RV[j], cross_cor[j, :], left=0., right=0.0)
                sigma_temp_ccf = np.interp(drv, drv-RV[j], sigma_cross_cor[j, :], left=0., right=0.0)
                shifted_ccfs[i,:] += temp_ccf * ccf_weights[j]
                var_shifted_ccfs[i,:] += sigma_temp_ccf**2 * ccf_weights[j]**2
        i+=1
    
    sigma_shifted_ccfs = np.sqrt(var_shifted_ccfs)
    shifted_ccfs -= np.median(shifted_ccfs) #handle any offset

    #use_for_snr = (np.abs(drv) <= 100.) & (np.abs(drv) >= 50.)#
    #use_for_snr = np.abs(drv) > 150.
    use_for_snr = np.abs(drv > 100.)
    #tempp = shifted_ccfs[:,use_for_snr]
    #use_for_snr_2 = (Kp <= 280.) & (Kp >= 180.)

    #snr = shifted_ccfs / np.std(tempp[use_for_snr_2,:])
    snr = shifted_ccfs / np.std(shifted_ccfs[:,use_for_snr])
    return snr, Kp, drv, cross_cor, sigma_shifted_ccfs, ccf_weights

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

def phase2angle(phase):
    return phase * 360.

def angle2phase(phase):
    return phase / 360.

def combine_ccfs_binned(drv, cross_cor, sigma_cross_cor, orbital_phase, n_spectra, ccf_weights, half_duration_phase, temperature_profile, Kp_here, species_name_ccf, planet_name):

    if 'Fe' in species_name_ccf:
        binsize = 0.015
    else:
        binsize = 0.05

    phase_bin = np.arange(np.min(orbital_phase), np.max(orbital_phase), binsize)
    
    nphase, nv = len(phase_bin), len(drv)

    #shifted_ccfs, var_shifted_ccfs = np.zeros((nKp, nv)), np.zeros((nKp, nv))

    binned_ccfs, var_shifted_ccfs = np.zeros((nphase, nv)), np.zeros((nphase, nv))

    i = 0


    Kp_here = unp.nominal_values(Kp_here)
    RV = Kp_here*np.sin(2.*np.pi*orbital_phase)
    
        
    for j in range (n_spectra):
        #restrict to only in-transit spectra if doing transmission:
        #print(orbital_phase[j])
        if not 'transmission' in temperature_profile or np.abs(orbital_phase[j]) <= half_duration_phase:
            phase_here = np.argmin(np.abs(phase_bin - orbital_phase[j]))
            
            temp_ccf = np.interp(drv, drv-RV[j], cross_cor[j, :], left=0., right=0.0)
            sigma_temp_ccf = np.interp(drv, drv-RV[j], sigma_cross_cor[j, :], left=0., right=0.0)
            binned_ccfs[phase_here,:] += temp_ccf * ccf_weights[j]
            #use_for_sigma = (np.abs(drv) <= 100.) & (temp_ccf != 0.)
            use_for_sigma = (np.abs(drv) > 100.) & (temp_ccf != 0.)
            #this next is b/c the uncertainties produced through the ccf routine are just wrong
            var_shifted_ccfs[phase_here,:] += np.std(temp_ccf[use_for_sigma])**2 * ccf_weights[j]**2

    sigma_shifted_ccfs = np.sqrt(var_shifted_ccfs)

    if planet_name == 'KELT-20b' and (species_name_ccf == 'Fe' or species_name_ccf == 'Fe+' or species_name_ccf == 'Ni'):
        ecc = 0.019999438851877625#0.0037 + 0.010 * 3.0 #rough 3-sigma limit
        omega = 309.2455607770675#151.

        ftransit=np.pi/2.-omega*np.pi/180.#-np.pi #true anomaly at transit
        Etransit=2.*np.arctan(np.sqrt((1.-ecc)/(1.+ecc))*np.tan(ftransit/2.)) #eccentric anomaly at transit
        timesince=1.0/(2.*np.pi)*(Etransit-ecc*np.sin(Etransit)) #time since periastron to transit
        RVe = radvel.kepler.rv_drive(orbital_phase, np.array([1.0, 0.0-timesince, ecc, omega*np.pi/180.-np.pi, Kp_here]))

        RVdiff = RVe - RV
        order = np.argsort(orbital_phase)

    good = np.abs(drv) < 30.
    pl.pcolor(drv[good], phase_bin, binned_ccfs[:,good], edgecolors='none',rasterized=True)
    pl.plot([0.,0.],[np.min(phase_bin), np.max(phase_bin)],':',color='white')
    #pl.plot(RVdiff[order], orbital_phase[order], '--', color='white')

    pl.xlabel('v (km/s)')
    pl.ylabel('orbital phase')
    pl.savefig(path_modifier_plots+'plots/'+planet_name+'.'+species_name_ccf+'.phase-binned.pdf', format='pdf')
    pl.clf()

    rvs, widths, rverrors, widtherrors = np.zeros(nphase), np.zeros(nphase), np.zeros(nphase), np.zeros(nphase)



    drvfit = drv[good]
    ccffit = binned_ccfs[:,good]
    sigmafit = sigma_shifted_ccfs[:,good]

    pp = PdfPages(path_modifier_plots+'plots/'+planet_name+'.'+species_name_ccf+'.phase-binned-RV-fits.pdf')

    

    for i in range (0, nphase):
        pl.subplot(3, 3, np.mod(i, 9)+1)
        peak = np.argmax(ccffit[i,:])
        popt, pcov = curve_fit(gaussian, drvfit, ccffit[i,:], p0=[ccffit[i,peak], drvfit[peak], 2.5], sigma = sigmafit[i,:], maxfev=10000)

        pl.plot(drvfit, ccffit[i,:], color='blue')
        pl.plot(drvfit, gaussian(drvfit, popt[0], popt[1], popt[2]), color='red')
        pl.xlabel('$v$ (km/s)')
        pl.ylabel('amplitude')
        pl.title(str(i) + ' ' + str(phase_bin[i]))

        if np.mod(i, 9)+1 == 9:
            pl.savefig(pp, format='pdf')
            pl.clf()

        rvs[i] = popt[1]
        widths[i] = popt[2]
        rverrors[i] = np.sqrt(pcov[1,1])
        widtherrors[i] = np.sqrt(pcov[2,2])

    pp.close()
    pl.clf()

    
    fig, ax = pl.subplots(layout='constrained')
    ax.pcolor(drv[good], phase_bin, binned_ccfs[:,good], edgecolors='none',rasterized=True)
    ax.plot([0.,0.],[np.min(phase_bin), np.max(phase_bin)],':',color='grey')

    goodrv = (rvs > 0.) & (rvs < 10.)
    ax.plot(rvs[goodrv], phase_bin[goodrv], 'o', color='white')
    ax.errorbar(rvs[goodrv], phase_bin[goodrv], xerr = rverrors[goodrv], color='white', fmt='none')

    ax.set_xlabel('v (km/s)')
    ax.set_ylabel('orbital phase (fraction)')

    ax.set_xlim([-10.,10.])

    secax = ax.secondary_yaxis('right', functions = (phase2angle, angle2phase))
    secax.set_ylabel('orbital phase (degrees)')
    pl.savefig(path_modifier_plots+'plots/'+planet_name+'.'+species_name_ccf+'.phase-binned+RVs.pdf', format='pdf')
    pl.clf()
    
    Kp = np.arange(50, 350, 1)
    # Line Profile plot
    idx = np.where(Kp == int(np.floor(Kp_here)))[0][0] #Kp slice corresponding to expected Kp

    rv_chars, rv_chars_error = np.zeros((len(Kp), n_spectra)), np.zeros((len(Kp), n_spectra))
    phase_array = np.linspace(np.min(orbital_phase), np.max(orbital_phase), num=n_spectra)   
    slice_peak_chars = np.zeros(len(Kp))

    i = 0
    for Kp_i in Kp:
        RV = Kp_i*np.sin(2.*np.pi*orbital_phase)
        
        for j in range(n_spectra):
            #restrict to only in-transit spectra if doing transmission:
            #also want to leave out observations in 2ndary eclipse!
            phase_here = np.argmin(np.abs(phase_array - orbital_phase[j]))
            temp_ccf = np.interp(drv, drv-RV[j], cross_cor[j, :], left=0., right=0.0)
            temp_ccf *= ccf_weights[j]
            peak = np.argmax(temp_ccf[390:411]) + 390
            sigma_temp_ccf = np.interp(drv, drv-RV[j], sigma_cross_cor[j, :], left=0., right=0.0)
            sigma_temp_ccf = sigma_temp_ccf**2 * ccf_weights[j]**2
            popt, pcov = curve_fit(gaussian, drv, temp_ccf, p0=[temp_ccf[peak], drv[peak], 2.5], sigma = np.sqrt(sigma_temp_ccf), maxfev=1000)
            rv_chars[i,phase_here] = popt[1]
            rv_chars_error[i,phase_here] = np.sqrt(pcov[1,1])
            slice_peak_chars[i] = temp_ccf[peak]
        i+=1

    fig, ax1 = pl.subplots(figsize=(8,8))

    ax1.text(0.05, 0.99, species_name_ccf, transform=ax1.transAxes, verticalalignment='top', horizontalalignment='left', fontsize=12)
    ax1.plot(rv_chars[idx,:], phase_array, '-', label='Center', color='white')
    #breakpoint()
    ax1.fill_betweenx(phase_array, rv_chars[idx,:] - rv_chars_error[idx, :], rv_chars[idx,:] + rv_chars_error[idx,:], color='blue', alpha=0.2, zorder=2)
    ax1.set_ylabel('Orbital Phase')
    ax1.set_xlabel('$v_{sys}$ (km/s)', color='b')
    ax1.tick_params(axis='x', labelcolor='b')
    
    # add a vertical line at 0km/s
    #ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)

    line_profile = path_modifier_plots + 'plots/' + planet_name + '.' + 'combined' + '.' + 'combined' + '.' + species_name_ccf + '.line-profile-binned.pdf'
    fig.savefig(line_profile, dpi=300, bbox_inches='tight')

    return binned_ccfs, rvs, widths, rverrors, widtherrors
        

def combine_likelihoods(drv, lnL, orbital_phase, n_spectra, half_duration_phase, temperature_profile):

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

def gaussian_fit(Kp, Kp_true, drv, species_label, planet_name, observation_epoch, arm, species_name_ccf, model_tag, plotsnr, sigma_shifted_ccfs, temperature_profile, cross_cor, sigma_cross_cor, ccf_weights):
 
    if arm == 'red':
        do_molecfit = True
    else:
        do_molecfit = False

    Period, epoch, M_star, RV_abs, i, M_p, R_p, RA, Dec, Kp_expected, half_duration_phase, Ks_expected = get_planet_parameters(planet_name)
    
    if arm == 'red' or arm == 'blue':
        wave, fluxin, errorin, jd, snr_spectra, exptime, airmass, n_spectra, npix = get_pepsi_data(arm, observation_epoch, planet_name, do_molecfit)
    else:
        # If arm is neither 'red' nor 'blue', use 'blue' as the default as do_molecfit will throw false when arm is 'combined'
        wave, fluxin, errorin, jd, snr_spectra, exptime, airmass, n_spectra, npix = get_pepsi_data('blue', observation_epoch, planet_name, do_molecfit)

    orbital_phase = get_orbital_phase(jd, epoch, Period, RA, Dec)

    # Gaussian Fit plot
    
    # Initializing lists to store fit parameters
    amps, amps_error = np.zeros(plotsnr.shape[0]), np.zeros(plotsnr.shape[0])
    rv, rv_error = np.zeros(plotsnr.shape[0]), np.zeros(plotsnr.shape[0])
    width, width_error = np.zeros(plotsnr.shape[0]), np.zeros(plotsnr.shape[0])

    slice_peak = np.zeros(plotsnr.shape[0])
    # Fitting gaussian to all 1D Kp slices

    for i in range(plotsnr.shape[0]):
        
        # Sort the peaks in descending order
        peaks = np.argsort(plotsnr[i,:])[::-1]
        
        # Iterate over the peaks
        for peak in peaks:
            # If the peak is within the desired range, fit the Gaussian
            if np.abs(drv[peak]) <= 15:
                slice_peak[i] = plotsnr[i, peak]
                
                popt, pcov = curve_fit(gaussian, drv, plotsnr[i,:], p0=[plotsnr[i, peak], drv[peak], 2.55035], sigma = sigma_shifted_ccfs[i,:], maxfev=10000)

                amps[i] = popt[0]
                rv[i] = popt[1]
                width[i] = popt[2]
                amps_error[i] = np.sqrt(pcov[0,0])
                rv_error[i] = np.sqrt(pcov[1,1])
                width_error[i] = np.sqrt(pcov[2,2])
                
                idx = np.where(Kp == int(np.floor(Kp_true)))[0][0] #Kp slice corresponding to expected Kp
                idx = np.argmax(slice_peak)                       #Kp slice corresponding to max SNR 
                # Break the loop as we have found a suitable peak
                break
            

    popt_selected = [amps[idx], rv[idx], width[idx]]
    print('Selected SNR:', amps[idx], '\n Selected Vsys:', rv[idx], '\n Selected sigma:', width[idx], '\n Selected Kp:', Kp[idx])

    # Computing residuals and chi-squared for selected slice
    residual = plotsnr[idx, :] - gaussian(drv, *popt_selected)
    # chi2 = np.sum((residual / np.std(residual))**2)/(len(drv)-len(popt))

    # Initialize Figure and GridSpec objects
    fig = pl.figure(figsize=(8,8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])

    # Create Axes for the main plot and the residuals plot
    ax1 = pl.subplot(gs[0])
    ax2 = pl.subplot(gs[1], sharex=ax1)

    plot_mask = np.abs(drv) <= 25.
    # Restrict arrays to the region of interest for plotting
    drv_restricted = drv[plot_mask]
    plotsnr_restricted = plotsnr[idx, plot_mask]
    residual_restricted = residual[plot_mask]

    # Main Plot (ax1)
    ax1.plot(drv_restricted, plotsnr_restricted, 'k--', label='data', markersize=2)
    ax1.plot(drv_restricted, gaussian(drv_restricted, *popt_selected), 'r-', label='fit')

    # Species Label
    ax1.text(0.05, 0.99, species_label, transform=ax1.transAxes, verticalalignment='top', horizontalalignment='left', fontsize=12)

    pl.setp(ax1.get_xticklabels(), visible=False)
    ax1.set_ylabel('SNR')
    # Annotating the arm and species on the plot

    # Additional text information for the main plot
    #params_str = f"Peak (a): {popt_selected[0]:.2f}\nMean (mu): {popt_selected[1]:.2f}\nSigma: {popt_selected[2]:.2f}\nKp: {Kp[idx]:.0f}"
    #ax1.text(0.01, 0.95, params_str, transform=ax1.transAxes, verticalalignment='top', fontsize=10)

    arm_species_text = f'Arm: {arm}'
    ax1.text(0.15, 0.95, arm_species_text, transform=ax1.transAxes, verticalalignment='top', fontsize=10)

    # Vertical line for the Gaussian peak center
    ax1.axvline(x=rv[idx], color='b', linestyle='-', label='Center')
    #ax1.set_title('1D CCF Slice + Gaussian Fit')

    # Vertical lines for sigma width (center ± sigma)
    #sigma_left = rv[idx] - width[idx]
    #sigma_right = rv[idx] + width[idx]
    #ax1.axvline(x=sigma_left, color='purple', linestyle='--', label='- Sigma')
    #ax1.axvline(x=sigma_right, color='purple', linestyle='--', label='+ Sigma')

    #ax1.legend()

    # Add the horizontal line at 4 SNR
    ax1.axhline(y=4, color='g', linestyle='--', label=r'4 $\sigma$')    

    # Inset for residuals (ax2)
    ax2.plot(drv_restricted, residual_restricted, 'o-', markersize=1)
    ax2.set_xlabel('$v_{sys}$ (km/s)')
    ax2.set_ylabel('Residuals')

    # Consider a clearer naming scheme
    snr_fit = path_modifier_plots + 'plots/'+ planet_name + '.' + observation_epoch + '.' + arm + '.' + species_name_ccf + model_tag + '.SNR-Gaussian.pdf'
    # Save the plot
    fig.savefig(snr_fit, dpi=300, bbox_inches='tight')


    # Line Profile plot
    idx = np.where(Kp == int(np.floor(Kp_true)))[0][0] #Kp slice corresponding to expected Kp

    Kp = np.arange(50, 350, 1)
    rv_chars, rv_chars_error = np.zeros((len(Kp), n_spectra)), np.zeros((len(Kp), n_spectra))
    phase_array = np.linspace(np.min(orbital_phase), np.max(orbital_phase), num=n_spectra)   
    slice_peak_chars = np.zeros(len(Kp))

    i = 0
    for Kp_i in Kp:
        RV = Kp_i*np.sin(2.*np.pi*orbital_phase)
        
        for j in range(n_spectra):
            #restrict to only in-transit spectra if doing transmission:
            #also want to leave out observations in 2ndary eclipse!
            phase_here = np.argmin(np.abs(phase_array - orbital_phase[j]))
            temp_ccf = np.interp(drv, drv-RV[j], cross_cor[j, :], left=0., right=0.0)
            temp_ccf *= ccf_weights[j]
            peak = np.argmax(temp_ccf[390:411]) + 390
            sigma_temp_ccf = np.interp(drv, drv-RV[j], sigma_cross_cor[j, :], left=0., right=0.0)
            sigma_temp_ccf = sigma_temp_ccf**2 * ccf_weights[j]**2
            popt, pcov = curve_fit(gaussian, drv, temp_ccf, p0=[temp_ccf[peak], drv[peak], 2.5], sigma = np.sqrt(sigma_temp_ccf), maxfev=1000)
            rv_chars[i,phase_here] = popt[1]
            rv_chars_error[i,phase_here] = np.sqrt(pcov[1,1])
            slice_peak_chars[i] = temp_ccf[peak]
        i+=1

    fig, ax1 = pl.subplots(figsize=(8,8))

    ax1.text(0.05, 0.99, species_label, transform=ax1.transAxes, verticalalignment='top', horizontalalignment='left', fontsize=12)
    ax1.plot(rv_chars[idx,:], phase_array, '-', label='Center')
    ax1.fill_betweenx(phase_array, rv_chars[idx,:] - rv_chars_error[idx, :], rv_chars[idx,:] + rv_chars_error[idx,:], color='blue', alpha=0.2, zorder=2)
    ax1.set_ylabel('Orbital Phase')
    ax1.set_xlabel('$v_{sys}$ (km/s)', color='b')
    ax1.tick_params(axis='x', labelcolor='b')
    
    line_profile = path_modifier_plots + 'plots/' + planet_name + '.' + observation_epoch + '.' + arm + '.' + species_name_ccf + model_tag + '.line-profile.pdf'
    fig.savefig(line_profile, dpi=300, bbox_inches='tight')

    return amps, amps_error, rv, rv_error, width, width_error, residual, do_molecfit, idx, line_profile, drv_restricted, plotsnr_restricted, residual_restricted

def make_shifted_plot(snr, planet_name, observation_epoch, arm, species_name_ccf, model_tag, RV_abs, Kp_expected, V_sys_true, Kp_true, do_inject_model, drv, Kp, species_label, temperature_profile, sigma_shifted_ccfs, method, cross_cor_display, sigma_cross_cor, ccf_weights, plotformat = 'pdf'):
    
    if method == 'ccf':
        outtag, zlabel = 'CCFs-shifted', 'SNR'
        plotsnr = snr[:]
    if 'likelihood' in method:
        outtag, zlabel = 'likelihood-shifted', '$\Delta\ln \mathcal{L}$'
        plotsnr=snr - np.max(snr)
    plotname = path_modifier_plots + 'plots/'+ planet_name + '.' + observation_epoch + '.' + arm + '.' + species_name_ccf + model_tag + '.' + outtag + '.' + plotformat

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

    #keeprv = np.abs(drv-apoints[0]) <= 100.
    keeprv = np.abs(drv-apoints[0]) <= 401.

    plotsnr, drv = plotsnr[:, keeprv], drv[keeprv]
    #keepKp = np.abs(Kp-apoints[1]) <= 100.
    keepKp = np.abs(Kp-apoints[1]) <= 401.

    plotsnr, Kp = plotsnr[keepKp, :], Kp[keepKp]
    
    # Fit a Gaussian to the line profile
    amps, amps_error, rv, rv_error, width, width_error, residual, do_molecfit, idx, line_profile, drv_restricted, plotsnr_restricted, residual_restricted = gaussian_fit(Kp, Kp_true, drv, species_label, planet_name, observation_epoch, arm, species_name_ccf, model_tag, plotsnr, sigma_shifted_ccfs, temperature_profile, cross_cor_display, sigma_cross_cor, ccf_weights)

    psarr(plotsnr, drv, Kp, '$V_{\mathrm{sys}}$ (km/s)', '$K_p$ (km/s)', zlabel, filename=plotname, ctable=ctable, alines=True, apoints=apoints, acolor='cyan', textstr=species_label+' '+model_label, textloc = np.array([apoints[0]-75.,apoints[1]+75.]), textcolor='cyan', fileformat=plotformat)
    
    return plotsnr, amps, amps_error, rv, rv_error, width, width_error, idx, drv_restricted, plotsnr_restricted, residual_restricted

def dopplerShadowRemove(drv, planet_name, exptime, orbital_phase, obs, inputs = {
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

        resolve_mapping = {
            'keck': 50000.0,
            'hjst': 60000.0,
            'het': 30000.0,
            'keck-lsi': 20000.0,
            'subaru': 80000.0,
            'aat': 70000.0,
            'not': 47000.0,
            'tres': 44000.0,
            'harpsn': 120000.0,
            'lbt': 120000.0,
            'pepsi': 120000.0,
            'igrins': 40000.0,
            'nres': 48000.0
        }
        if planet_name == 'KELT-20b':
            vsini = 110
            lambda_p = 0.5  

        Resolve = resolve_mapping.get(obs, 0.0)  # Default value of 0.0 if obs is not found in the mapping
        
        #get planetary parameters
        Period, epoch, M_star, RV_abs, i, M_p, R_p, RA, Dec, Kp_expected, half_duration_phase, Ks_expected = get_planet_parameters(planet_name)
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

def get_peak_snr(snr, drv, Kp, do_inject_model, V_sys_true, Kp_true, RV_abs, Kp_expected, arm, observation_epoch, f, method):

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

def inreader(infile):
    temp=ascii.read(infile)
    names, values = temp[0][:],temp[1][:]
    outstruc = dict(list(zip(names, values)))
    outstruc['index']=names
    outstruc['invals']=values
    return outstruc

def str2bool(inval):
    if inval == 'True':
        return True
    else:
        return False

def get_species_mass(species_name):

    if 'TiO' in species_name: return 47.867+15.999
    if 'Ti' in species_name: return 47.867
    if 'VO' in species_name: return 50.9415+15.999
    if 'V' in species_name: return 50.9415
    if 'FeH' in species_name: return 55.845+1.00784
    if 'Fe' or 'Fe+' in species_name: return 55.845
    if 'Ni' in species_name: return 58.6934
    if 'SiO' in species_name: return 28.085+15.999
    if 'Si' in species_name: return 28.085
    if 'OH' in species_name: return 15.999+1.00784
    if 'H2O' in species_name: return 15.999+2.*1.00784
    if 'H2S' in species_name: return 32.06+2.*1.00784
    if 'CrH' in species_name: return 51.9961+1.00784
    if 'Cr' in species_name: return 51.9961
    if 'CaH' in species_name: return 40.078+1.00784
    if 'Ca' in species_name: return 40.078
    if 'CO' in species_name: return 12.011+15.999
    if 'MgH' in species_name: return 24.305+1.00784
    if 'Mg' in species_name: return 24.305
    if 'HCN' in species_name: return 1.00784 + 12.011 + 14.007
    if 'PH3' in species_name: return 30.974 + 3*1.00784
    if 'Ca' in species_name: return 40.078
    if 'NaH' in species_name: return 22.990+1.00784
    if 'H' in species_name: return 1.0078
    if 'N' in species_name: return 14.007
    if 'Li' in species_name: return 6.9410
    if 'Na' in species_name: return 22.990
    if 'Al' in species_name: return 26.982
    
    if 'Si' in species_name: return 28.086
    if 'K' in species_name: return 39.098
    if 'Sc' in species_name: return 44.956
    if 'Sr' in species_name: return 87.62
    if 'Mn' in species_name: return 54.938
    if 'Co' in species_name: return 58.933
    if 'Cu' in species_name: return 63.546
    if 'Zn' in species_name: return 65.380
    if 'Ga' in species_name: return 69.723
    if 'Ge' in species_name: return 72.640

    if 'Rb' in species_name: return 85.468
    if 'Sr' in species_name: return 87.620
    if 'Y' in species_name: return 88.906
    if 'Zr' in species_name: return 91.224
    if 'Nb' in species_name: return 92.906
    if 'Ru' in species_name: return 101.07
    if 'Rh' in species_name: return 102.91
    if 'Pd' in species_name: return 106.42
    if 'In' in species_name: return 114.82
    
    if 'Sn' in species_name: return 118.71
    if 'Cs' in species_name: return 132.91
    if 'Ba' in species_name: return 137.33
    if 'Hf' in species_name: return 178.49
    if 'Os' in species_name: return 190.23
    if 'Ir' in species_name: return 192.22
    if 'Tl' in species_name: return 204.38
    if 'Pb' in species_name: return 207.20 
    
def get_planetary_parameters(planet_name):
    #will need to change this if want to actually marginalize over the planetary parameters, but just fix them for now.

    if planet_name == 'KELT-20b':
        R_host = 1.565*nc.r_sun # NEA: Lund et al. 2017, (+0.057, -0.064)
        R_pl = 19.51*nc.r_earth # NEA (+0.77, -0.83)
        M_pl = 1072*nc.m_earth # NEA
        Teff = 8720.

    if planet_name == 'KELT-9b':
        R_host = 2.52 * nc.r_sun 
        R_pl = 1.891 * nc.r_jup
        M_pl = 2.88 * nc.m_jup    
        Teff = 10170.

    if planet_name == 'WASP-76b': #all from West et al. 2016
        R_host = 1.73 * nc.r_sun 
        R_pl = 1.83 * nc.r_jup
        M_pl = 0.92 * nc.m_jup    
        Teff = 6250.

    if planet_name == 'WASP-33b': #from Stassun+17, Lehmann+15, Chakrabarty & Sengupta 19
        R_host = 1.55 * nc.r_sun 
        R_pl = 1.60 * nc.r_jup
        M_pl = 2.1 * nc.m_jup    
        Teff = 7308.
    
    if planet_name == 'WASP-189b':#from Anderson+18, Deline+22
        R_host = 2.363 * nc.r_sun 
        R_pl = 1.600 * nc.r_jup
        M_pl = 2.13 * nc.m_jup    
        Teff = 8000.
    
    if planet_name == 'TOI-1431b':#from Addison+
        R_host = 1.92 * nc.r_sun 
        R_pl = 1.49 * nc.r_jup
        M_pl = 3.12 * nc.m_jup    
        T_equ = 2370.
        Teff = 7690.

    if planet_name == 'TOI-1518b':#from Cabot+
        R_host = 1.950 * nc.r_sun 
        R_pl = 1.875 * nc.r_jup
        M_pl = 2.3 * nc.m_jup    #UPPER LIMIT
        T_equ = 2492.
        Teff = 7300.
    
    if planet_name == 'WASP-12b':#from Bonomo+17, Charkabarty & Sengupta 2019
        R_host = 1.619 * nc.r_sun 
        R_pl = 1.825 * nc.r_jup
        M_pl = 1.39 * nc.m_jup    
        Teff = 6250.

    if planet_name == 'MASCARA-1b':
        R_host = 2.082 * nc.r_sun 
        R_pl = 1.597 * nc.r_jup
        M_pl = 3.7 * nc.m_jup    
        Teff = 7490.

    gravity = nc.G * (M_pl)/(R_pl**2) 
    

    return R_host, R_pl, M_pl, Teff, gravity

def process_data(observation_epochs, arms, planet_name):
    niter = 10
    ckms = 2.9979e5

    datastruc = {}
    count = 0

    Period, epoch, M_star, RV_abs, i, M_p, R_p, RA, Dec, Kp_expected, half_duration_phase, Ks_expected = get_planet_parameters(planet_name)

    for observation_epoch in observation_epochs:
        for arm in arms:
            n_systematics = np.array(get_sysrem_parameters(arm, observation_epoch, 'null', planet_name))
            countstr = str(count)
            if arm == 'red':
                do_molecfit = True
            else:
                do_molecfit = False

        

            wave0, fluxin0, errorin0, jd0, snr_spectra0, exptime0, airmass0, n_spectra0, npix0 = get_pepsi_data(arm, observation_epoch, planet_name, do_molecfit)

            orbital_phase0 = get_orbital_phase(jd0, epoch, Period, RA, Dec)


            wave0, flux0, ccf_weights = regrid_data(wave0, fluxin0, errorin0, n_spectra0, [], [], snr_spectra0, 'null', True)

            residual_flux0 = flatten_spectra(flux0, npix0, n_spectra0)

            corrected_flux0, corrected_error0, U_sysrem, telluric_free = do_sysrem(wave0, residual_flux0, arm, airmass0, n_spectra0, niter, n_systematics, do_molecfit)

            datastruc['observation_epoch'+countstr], datastruc['arm'+countstr] = observation_epoch, arm
            datastruc['wave'+countstr], datastruc['flux'+countstr], datastruc['error'+countstr] = wave0, corrected_flux0, corrected_error0
            datastruc['orbital_phase'+countstr] = orbital_phase0
            datastruc['n_spectra'+countstr] = n_spectra0
            datastruc['U_sysrem'+countstr] = U_sysrem
            datastruc['telluric_free'+countstr] = telluric_free

            count+=1

    datastruc['n_datasets'] = count

    return datastruc
     




def generate_atmospheric_model(planet_name, spectrum_type, instrument, arm, all_species, parameters, atmosphere, pressures, ptprofile='guillot'):
    #handle the W-189 observations being with CD II
    if (planet_name == 'WASP-189b' or planet_name == 'KELT-20b') and 'GMT-all' not in instrument:
        instrument_here = 'PEPSI-25' 
    elif instrument == 'PEPSI':
        instrument_here = 'PEPSI-35'
    else:
        instrument_here = instrument

    lambda_low, lambda_high = get_wavelength_range(instrument_here)

    mass = []
    vmrs = []
    

    for species in all_species:
        mass.append(get_species_mass(species))
        vmrs.append(parameters[species])

    mass = np.array(mass)
    mmw = 2.33
    species_abundance = mass/mmw * vmrs

    H2_abundance = 1.008
    He_abundance = 4.00126 * (10.**(10.925-12))

    Hm_abundance = (1.00784/mmw) * parameters['Hm']
    e_abundance = ((1./1822.8884845)/mmw) * parameters['em']
    H_abundance = (1.00784/mmw) * parameters['H0']

    total_abundance = np.sum(species_abundance) + H2_abundance + He_abundance + e_abundance+ H_abundance + Hm_abundance
    abundances = {}
    i=0
    for species in all_species:
        if len(all_species) > 1:
            abundances[species] = species_abundance[i][0]/total_abundance #[0]
        else:
            abundances[species] = species_abundance[i]/total_abundance
        i+=1
    abundances['H2'] = H2_abundance/total_abundance
    abundances['He'] = He_abundance/total_abundance
    abundances['H'] = H_abundance/total_abundance
    abundances['H-'] = Hm_abundance/total_abundance
    abundances['e-'] = e_abundance/total_abundance
    R_host, R_pl, M_pl, Teff, gravity = get_planetary_parameters(planet_name)
    T_equ = parameters['Teq']

    P0 = parameters['P0']
    if ptprofile == 'guillot':
        kappa_IR, gamma = parameters['kappa'], parameters['gamma']
    if ptprofile == 'two-point':
        P1, P2, T_high = parameters['P1'], parameters['P2'], parameters['T_high']
    T_int = 100.

    #parameters to pass into petitRADTRANS
    params = []
    params.append(all_species)
    params.append(lambda_low)
    params.append(lambda_high)
    params.append(R_pl)
    params.append(R_host)
    params.append(gravity)
    params.append(P0)
    if ptprofile == 'guillot':
        params.append(kappa_IR)
        params.append(gamma)
    else:
        params.append(P1)
        params.append(P2)
    params.append(T_int)
    params.append(T_equ)
    params.append(abundances)
    params.append(pressures)
    if ptprofile == 'two-point': 
        params.append(T_high)



    wav_pl, flux_pl, temperature, pressures, contribution = create_model(params, spectrum_type, False, False, atmosphere=atmosphere, ptprofile = ptprofile)

    flux_st = nc.b(Teff, nc.c/(wav_pl/1e8)) * u.erg / (u.cm * u.cm) / u.s / u.Hz

    if spectrum_type == 'emission':
        flux_pl = flux_pl * u.erg / (u.cm * u.cm) / u.s / u.Hz

        flux_ratio = flux_pl / flux_st * (R_pl / R_host) **2
        flux_ratio += 1.0

    if spectrum_type == 'transmission':
        flux_ratio = 1.0 - (flux_pl**2 - R_pl**2) / R_host**2

    #convert to air wavelengths
    #want to do this for everything except Plez line lists
    if 'Plez' not in species and 'Fe+' not in species and 'Ti+' not in species:
        wav_pl = vacuum2air(wav_pl)

    return wav_pl, flux_ratio - 1.0

def make_mocked_data(struc1, index, invals, instruc, atmosphere, pressures, lambda_low, lambda_high):

    Period, epoch, M_star, RV_abs, i, M_p, R_p, RA, Dec, Kp_expected, half_duration_phase, Ks_expected = get_planet_parameters(struc1['planet_name'])

    parameter_names = ['par-Hm', 'par-H0', 'par-em', 'par-P0', 'par-kappa', 'par-gamma', 'par-Teq', 'par-alpha', 'par-beta', 'par-norm_offset', 'par-Kp', 'par-DeltaRV']
    parameters = {}
    
    for parameter in parameter_names:
        parameters[parameter.split('-')[1]] = np.float(get_parameters(invals, index, instruc, parameter))

    for species in instruc['all_species']:
        parameters[species] = 10.**np.float(get_parameters(invals, index, instruc, 'par-vmr' + species)) #fit the log of the VMR

    template_wave, template_flux = generate_atmospheric_model(struc1['planet_name'], struc1['spectrum_type'], struc1['instrument'], 'combined', instruc['all_species'], parameters, atmosphere, pressures, ptprofile='guillot')

    datastruc = {}
    count = 0

    if 'PEPSI' in struc1['instrument']:
        arms = ['blue', 'red']

        for arm in arms:

            countstr = str(count)
    
            if arm == 'red': wave = np.linspace(6231, 7427, 43416)
            if arm == 'blue': wave = np.linspace(4752, 5425, 33669)
        
            n_pix = len(wave)

            fluxin = np.zeros((int(struc1['n_spectra']), n_pix)) + np.random.randn(int(struc1['n_spectra']), n_pix)/np.float(struc1['snr'])
            errorin = np.zeros((int(struc1['n_spectra']), n_pix)) + 1./np.float(struc1['snr'])
            wavein = np.zeros_like(fluxin)
            for i in range (0, int(struc1['n_spectra'])):
                wavein[i,:] = wave

            orbital_phase = np.linspace(np.float(struc1['phasemin']), np.float(struc1['phasemax']), int(struc1['n_spectra']))

            fluxin, Kp_true, V_sys_true = inject_model(Kp_expected, orbital_phase, wavein, fluxin, template_wave, template_flux, int(struc1['n_spectra']))

            datastruc['observation_epoch'+countstr], datastruc['arm'+countstr] = struc1['dataset0'], arm
            datastruc['wave'+countstr], datastruc['flux'+countstr], datastruc['error'+countstr] = wave, fluxin, errorin
            datastruc['orbital_phase'+countstr] = orbital_phase
            datastruc['n_spectra'+countstr] = int(struc1['n_spectra'])

            count+=1

        datastruc['n_datasets'] = count
    
    else:
        print('Other instruments have not yet been implemented for mock data')
        sys.exit()

    return datastruc

def parameters():

    parameters = {}

    #the below are default parameters for KELT-20b for testing
    parameters['vmr'] = [1e-5]
    parameters['Hm'] = 1e-9
    parameters['em'] = 0.0008355 #constant for now, link to FastChem later
    parameters['H0'] = 2.2073098e-12 #ditto
    parameters['P0'] = 1.0
    parameters['kappa'] = 0.01
    parameters['gamma'] = 50.
    parameters['Teq'] = 2262.

def get_parameters(theta, index, instruc, name):
    if any(name in s for s in index):
        value = theta[[i for i, s in enumerate(index) if name in s]]
    else:
        value = instruc[name]
    if 'Hm' in name or 'H0' in name or 'em' in name: #handle strictly positive parameters for which we fit the log:
        value = np.float(value)
        return 10.**value
    else:
        return value

def get_likelihood(theta, instruc, datastruc, inposindex, atmosphere):

    #prepare the parameters
    parameter_names = ['Hm', 'H0', 'em', 'P0', 'kappa', 'gamma', 'Teq', 'alpha', 'beta', 'norm_offset', 'Kp', 'DeltaRV']
    parameters = {}
    
    for parameter in parameter_names:
        parameters[parameter] = get_parameters(theta, inposindex, instruc, parameter)

    for species in instruc['all_species']:
        parameters[species] = 10.**get_parameters(theta, inposindex, instruc, species) #fit the log of the VMR


    template_wave, template_flux = generate_atmospheric_model(instruc['planet_name'], instruc['spectrum_type'], instruc['instrument'], 'combined', instruc['all_species'], parameters, atmosphere, instruc['pressures'])

    #pl.plot(template_wave, template_flux)
    #pl.show()

    lnL = 0.0

    #need to iterate over the different datasets
    for i in range (datastruc['n_datasets']):
        ii = str(i)

        wave, flux, flux_error = datastruc['wave'+ii], datastruc['flux'+ii], datastruc['error'+ii]
        orbital_phase = datastruc['orbital_phase'+ii]
    

        variance = flux_error**2

    
        RV = parameters['Kp'] * np.sin(2. * np.pi * orbital_phase)
        for j in range (datastruc['n_spectra'+ii]):
            bigN = len(flux[j, :])
            constant_term = (-1.0) * bigN / 2. * np.log(2.*np.pi) - bigN * np.log(parameters['beta']) - np.sum(np.log(flux_error[j, :]))
            flux_term = np.sum(flux[j, :]**2 / variance[j, :])
            model_term, CCF_term = one_log_likelihood(wave, flux[j, :], variance[j, :], template_wave, template_flux, RV[j] + parameters['DeltaRV'], parameters['norm_offset']) #not sure if I have the right sign on the shift!
            chi2 = 1/parameters['beta']**2 * (flux_term + parameters['alpha']**2 * model_term - 2. * parameters['alpha'] * CCF_term)
            lnL += constant_term - 0.5 * chi2

    #print(lnL)
    return lnL

def lnprior(theta, instruc, inpos, inposindex, priorstruc):

    lp = 0.0
    #reject unphysical values
    #also need to make sure no VMR is >1, need to figure out how to do that...
    if any(t < 0. for t in theta[[i for i, s in enumerate(inposindex) if 'alpha' in s]]) or any(t < 0. for t in theta[[i for i, s in enumerate(inposindex) if 'P0' in s]]) or any(t < 0. for t in theta[[i for i, s in enumerate(inposindex) if 'kappa' in s]]) or any(t < 0. for t in theta[[i for i, s in enumerate(inposindex) if 'gamma' in s]]) or any(np.abs(t) > 300. for t in theta[[i for i, s in enumerate(inposindex) if 'DeltaRV' in s]]):
        return -np.inf

    #keep VMRs less than 1
    for species in instruc['all_species']:
        if any(t >= 1.0 for t in theta[[i for i, s in enumerate(inposindex) if species in s]]):
            return -np.inf

    for par, sigma in priorstruc.items():
        lp += (theta[[i for i, s in enumerate(inposindex) if par in s]] - instruc['val-'+par])**2 / sigma**2
    

    lp*=(-0.5)
    if not np.isfinite(lp):
        return 0.0
    return lp

def lnprob(theta, instruc, datastruc, inpos, inposindex, priorstruc, atmosphere):

    datastruc['count'] += 1

    lnL = 0.0

    if bool(priorstruc):
        lnL += lnprior(theta, instruc, inpos, inposindex, priorstruc)
    #skip doing the expensive modeling if we have unphysical parameters
    if not np.isfinite(lnL):
        return lnL

    lnl = get_likelihood(theta, instruc, datastruc, inposindex, atmosphere)
    lnL += float(lnl) #get rid of Quantity that somehow got attached

    if np.isnan(lnL):
        print('The likelihood returned a NaN',lnL)
        return -np.inf #take care of nan crashes, hopefully...
    #print('Finished with another iteration: ',str(datastruc['count']),' lnL = ',lnL)
    return lnL

def nsf(num, n=1):
    #from StackOverflow: https://stackoverflow.com/questions/9415939/how-can-i-print-many-significant-figures-in-python
    """n-Significant Figures"""
    while n-1 < 0: n+=1
    numstr = ("{0:.%ie}" % (n-1)).format(num)
    return float(numstr)

def parstring(temp):
    if temp[1] == 0.: return str(temp[0])+' (fixed)'
    if np.isnan(temp[1]) or np.isnan(temp[2]): return str(temp[0])+' bad values!'
    numnum1=np.array([int(np.abs(np.floor(np.log10(temp[1]))-1)),int(np.abs(np.floor(np.log10(temp[2]))-1))])
    numnum=np.max(numnum1)
    valstr, errpstr, errmstr = str(round(temp[0],numnum)), str(nsf(temp[1],2)), str(nsf(temp[2],2))
    if 'e' in valstr:
        valstr=valstr.replace('e-0','\\times10^{-')
        valstr=valstr.replace('e+0','\\times10^{+')
        valstr=valstr.replace('e-','\\times10^{-')
        valstr=valstr.replace('e+','\\times10^{+')
        valstr+='}'
    if 'e' in errpstr:
        errpstr=errpstr.replace('e-0','\\times10^{-')
        errpstr=errpstr.replace('e+0','\\times10^{+')
        errpstr=errpstr.replace('e-','\\times10^{-')
        errpstr=errpstr.replace('e+','\\times10^{+')
        errpstr+='}'
    if 'e' in errmstr:
        errmstr=errmstr.replace('e-0','\\times10^{-')
        errmstr=errmstr.replace('e+0','\\times10^{+')
        errmstr=errmstr.replace('e-','\\times10^{-')
        errmstr=errmstr.replace('e+','\\times10^{+')
        errmstr+='}'    
    if nsf(temp[1],2) == nsf(temp[2],2):
        return valstr+' \pm '+ errpstr
    else:
        if numnum1[0] == numnum1[1]:
            return valstr+'^{+'+errpstr+'}_{-'+errmstr+'}'
        else:
            diff=numnum1[1]-numnum1[0]
            if diff < 0:
                errmstr=str(nsf(temp[2],2+diff))
                if 'e' in errmstr:
                    errmstr=errmstr.replace('e-0','\\times10^{-')
                    errmstr=errmstr.replace('e+0','\\times10^{+')
                    errmstr=errmstr.replace('e-','\\times10^{-')
                    errmstr=errmstr.replace('e+','\\times10^{+')
                    errmstr+='}'    
                return valstr+'^{+'+errpstr+'}_{-'+errmstr+'}'
            else:
                errpstr=str(nsf(temp[1],2+diff))
                if 'e' in errpstr:
                    errpstr=errpstr.replace('e-0','\\times10^{-')
                    errpstr=errpstr.replace('e+0','\\times10^{+')
                    errpstr=errpstr.replace('e-','\\times10^{-')
                    errpstr=errpstr.replace('e+','\\times10^{+')
                    errpstr+='}'
                return valstr+'^{+'+errpstr+'}_{-'+errmstr+'}'

def symdex(par):
    if 'beta' in par: return '$\\beta$'
    if 'alpha' in par: return '$\\alpha$'
    if 'P0' in par: return '$P_0$ (bar)'
    if 'kappa' in par: return '$\kappa_{\mathrm{IR}}$'
    if 'gamma' in par: return '$\gamma$'
    if 'Teq' in par: return '$T_{\mathrm{eq}}$ (K)'
    if 'DeltaRV' in par: return '$\Delta$v (km s$^{-1}$)'
    if 'Kp' in par: return '$K_P$ (km s$^{-1}$)'
    if 'Hm' in par: return '$\log$ VMR (H$^-$)'
    if 'em' in par: return '$\log$ VMR (e$^-$)'
    if 'H0' in par: return '$\log$ VMR (H)'
    else: return '$\log$ VMR ('+par+')'
        
def print_results(samples, inposindex, ndim, struc1, allpardex, index):

    f = open('logs/radiant_results.tex','w')
    f.write('\\begin{table} \n')
    f.write('\\caption{Retrieval results for '+struc1['planet_name']+'} \n')
    f.write('\\begin{tabular}{lc} \n')
    f.write('\\hline \n')
    f.write('\\hline \n')
    f.write('Parameter & Value & Prior \\\ \n')
    f.write('\\hline \n')
    f.write('Measured Parameters \\\ \n')
    f.write('\\hline \n')
    
    for i in range (0,ndim):
        v=np.nanpercentile(samples[:,i], [16, 50, 84], axis=0)
        temp=[v[1], v[2]-v[1], v[1]-v[0]]
        if any('pri-'+inposindex[i] in s for s in index):
            priorstring = '$\mathcal{G}$('+struc1['par-'+inposindex[i]]+', '+struc1['pri-'+inposindex[i]]+')'
        else:
            priorstring = '$\mathcal{U}(-\infty, \infty)$'
        f.write(symdex(inposindex[i])+' & $'+parstring(temp)+'$ & ' + priorstring + '\\\ \n')

    f.write('\\hline \n')
    f.write('Fixed Parameters \\\ \n')
    for par0 in allpardex:
        temp = par0.split('-')
        par = temp[1]
        if not str2bool(struc1['fit-'+par]):
            f.write(symdex(par) + ' & $'+struc1[par0]+'$ & fixed \\\ \n')
    f.write('\\hline \n')
    f.write('\\end{tabular} \n')
    f.write('\\end{table} \n')
    f.close()

def make_corner(samples, inposindex, ndim):
    
    import corner

    labels=np.zeros(ndim,dtype='object')
    for i in range (0,ndim):
        labels[i]=symdex(inposindex[i])
    corner.corner(samples,labels=labels,quantiles=[0.16,0.5,0.84])
    pl.savefig(path_modifier_plots+'plots/radiant-corner.pdf',format='pdf')
    pl.clf()

def plot_pt_profile(samples, inposindex, struc1, pressures, planet_name):

    import petitRADTRANS.nat_cst as nc

    R_host, R_pl, M_pl, Teff, gravity = get_planetary_parameters(planet_name)

    free = False

    shape = samples.shape

    random_samps = np.random.randint(0, shape[0], size=100)
    
    if any('gamma' in s for s in inposindex):
        gamma = samples[random_samps,[i for i, s in enumerate(inposindex) if 'gamma' in s]]
        gamma_med = np.median(samples[:,[i for i, s in enumerate(inposindex) if 'gamma' in s]])
        free = True
    else:
        gamma = float(struc1['par-gamma']) * np.ones(50)
        gamma_med = float(struc1['par-gamma'])

    if any('kappa' in s for s in inposindex):
        kappa = samples[random_samps,[i for i, s in enumerate(inposindex) if 'kappa' in s]]
        kappa_med = np.median(samples[:,[i for i, s in enumerate(inposindex) if 'kappa' in s]])
        free = True
    else:
        kappa = float(struc1['par-kappa']) * np.ones(50)
        kappa_med = float(struc1['par-kappa'])

    if any('Teq' in s for s in inposindex):
        Teq = samples[random_samps,[i for i, s in enumerate(inposindex) if 'Teq' in s]]
        Teq_med = np.median(samples[:,[i for i, s in enumerate(inposindex) if 'Teq' in s]])
        free = True
    else:
        Teq = float(struc1['par-Teq']) * np.ones(50)
        Teq_med = float(struc1['par-Teq'])


    temperature = nc.guillot_global(pressures, kappa_med, gamma_med, gravity, 100., Teq_med)

    pl.plot(temperature, pressures, color='red', linewidth=3)

    pl.xlabel('$T$ (K)')
    pl.ylabel('$P$ (bar)')

    pl.yscale('log')
    pl.ylim([1e0,1e-5])

    pl.tight_layout()

    pl.savefig(path_modifier_plots+'plots/radiant_pt_profile.pdf', format='pdf')
    pl.clf()