import numpy as np
import matplotlib.pyplot as pl
import matplotlib.gridspec as gridspec
from glob import glob
from astropy.io import fits, ascii
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
from astropy.table import Table
from uncertainties import ufloat, unumpy as unp
import time
from dtutils import psarr
from create_model import create_model, instantiate_radtrans
from atmo_utilities import *
import emcee
import argparse
import os
import petitRADTRANS.nat_cst as nc
from petitRADTRANS import Radtrans
import csv
from csv import reader
import pyfastchem
from astropy import constants as const
from scipy.optimize import curve_fit
from scipy.integrate import simps
from petitRADTRANS.physics import guillot_global
from matplotlib.backends.backend_pdf import PdfPages
import radvel
from matplotlib.colors import Normalize, LogNorm, to_hex
from matplotlib.cm import plasma, inferno, magma, viridis, cividis, turbo, ScalarMappable
from pandas import options
from typing import List
import warnings
from matplotlib import cm, colors
from matplotlib import patheffects
import os.path
from bokeh.models import ColumnDataSource, LinearColorMapper, LogColorMapper, ColorBar, BasicTicker
from bokeh.plotting import figure, output_file
from bokeh.io import show as show_, export_png
from bokeh.sampledata.periodic_table import elements
from bokeh.transform import dodge
from selenium import webdriver
import horus

pl.rc('font', size=14) #controls default text size
pl.rc('axes', titlesize=14) #fontsize of the title
pl.rc('axes', labelsize=14) #fontsize of the x and y labels
pl.rc('xtick', labelsize=14) #fontsize of the x tick labels
pl.rc('ytick', labelsize=14) #fontsize of the y tick labels
pl.rc('legend', fontsize=14) #fontsize of the legend


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
    
    if planet_name == 'TOI-1431b':#from Anderson+18, Deline+22
        R_host = 1.92 * nc.r_sun 
        R_pl = 1.49 * nc.r_jup
        M_pl = 3.12 * nc.m_jup    
        T_equ = 2370.
        Teff = 7690.
    
    if planet_name == 'WASP-12b':#from Bonomo+17, Charkabarty & Sengupta 2019
        R_host = 1.619 * nc.r_sun 
        R_pl = 1.825 * nc.r_jup
        M_pl = 1.39 * nc.m_jup    
        Teff = 6250.

    gravity = nc.G * (M_pl)/(R_pl**2) 
    

    return R_host, R_pl, M_pl, Teff, gravity

def process_data(observation_epochs, arms, planet_name):
    niter = 10
    ckms = 2.9979e5

    datastruc = {}
    count = 0

    Period, epoch, M_star, RV_abs, i, M_p, R_p, RA, Dec, Kp_expected, half_duration_phase = get_planet_parameters(planet_name)

    for observation_epoch in observation_epochs:
        for arm in arms:
            n_systematics = np.array(get_sysrem_parameters(arm, observation_epoch, 'null'))
            countstr = str(count)
            if arm == 'red':
                do_molecfit = True
            else:
                do_molecfit = False

        

            wave0, fluxin0, errorin0, jd0, snr_spectra0, exptime0, airmass0, n_spectra0, npix0 = get_pepsi_data(arm, observation_epoch, planet_name, do_molecfit)

            orbital_phase0 = get_orbital_phase(jd0, epoch, Period, RA, Dec)


            wave0, flux0, ccf_weights = regrid_data(wave0, fluxin0, errorin0, n_spectra0, [], [], snr_spectra0, 'null', True)

            residual_flux0 = flatten_spectra(flux0, npix0, n_spectra0)

            corrected_flux0, corrected_error0 = do_sysrem(wave0, residual_flux0, arm, airmass0, n_spectra0, niter, n_systematics, do_molecfit)

            datastruc['observation_epoch'+countstr], datastruc['arm'+countstr] = observation_epoch, arm
            datastruc['wave'+countstr], datastruc['flux'+countstr], datastruc['error'+countstr] = wave0, corrected_flux0, corrected_error0
            datastruc['orbital_phase'+countstr] = orbital_phase0
            datastruc['n_spectra'+countstr] = n_spectra0

            count+=1

    datastruc['n_datasets'] = count

    return datastruc
     
def get_wavelength_range(instrument_here):
    if instrument_here == 'PEPSI-35': lambda_low, lambda_high = 4750., 7500.
    if instrument_here == 'PEPSI-25': lambda_low, lambda_high = 4250., 7500.
    if instrument_here == 'MaroonX': lambda_low, lambda_high = 5000., 9200.
    if instrument_here == 'full-range': lambda_low, lambda_high = 3800., 37000.
    return lambda_low, lambda_high


def generate_atmospheric_model(planet_name, spectrum_type, instrument, arm, all_species, parameters, atmosphere, pressures, ptprofile='guillot'):

    #handle the W-189 observations being with CD II
    if planet_name == 'WASP-189b':
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
    #wav_pl = vacuum2air(wav_pl)

    return wav_pl, flux_ratio - 1.0

def make_mocked_data(struc1, index, invals, instruc, atmosphere, pressures, lambda_low, lambda_high):

    Period, epoch, M_star, RV_abs, i, M_p, R_p, RA, Dec, Kp_expected, half_duration_phase = get_planet_parameters(struc1['planet_name'])

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

            fluxin = np.zeros((np.int(struc1['n_spectra']), n_pix)) + np.random.randn(np.int(struc1['n_spectra']), n_pix)/np.float(struc1['snr'])
            errorin = np.zeros((np.int(struc1['n_spectra']), n_pix)) + 1./np.float(struc1['snr'])
            wavein = np.zeros_like(fluxin)
            for i in range (0, np.int(struc1['n_spectra'])):
                wavein[i,:] = wave

            orbital_phase = np.linspace(np.float(struc1['phasemin']), np.float(struc1['phasemax']), np.int(struc1['n_spectra']))

            fluxin, Kp_true, V_sys_true = inject_model(Kp_expected, orbital_phase, wavein, fluxin, template_wave, template_flux, np.int(struc1['n_spectra']))

            datastruc['observation_epoch'+countstr], datastruc['arm'+countstr] = struc1['dataset0'], arm
            datastruc['wave'+countstr], datastruc['flux'+countstr], datastruc['error'+countstr] = wave, fluxin, errorin
            datastruc['orbital_phase'+countstr] = orbital_phase
            datastruc['n_spectra'+countstr] = np.int(struc1['n_spectra'])

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
    pl.savefig('plots/radiant-corner.pdf',format='pdf')
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

    pl.savefig('plots/radiant_pt_profile.pdf', format='pdf')
    pl.clf()
    
def get_wavelength_range(instrument_here):
    if instrument_here == 'PEPSI-35': lambda_low, lambda_high = 4750., 7500.
    if instrument_here == 'PEPSI-25': lambda_low, lambda_high = 4250., 7500.
    if instrument_here == 'MaroonX': lambda_low, lambda_high = 5000., 9200.
    if instrument_here == 'full-range': lambda_low, lambda_high = 3800., 37000.
    return lambda_low, lambda_high

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
    if species_label == 'Zr II':
        species_names.add('Zr+')
    if species_label == 'Tl I':
        species_names.add('Tl')

    if species_label == 'Na_Allard':
        species_names.add(('Na_allard_new', 'Na'))
    if species_label == 'Na_Burrows':
        species_names.add(('Na_burrows', 'Na'))
    if species_label == 'Na_lor_cut':
        species_names.add(('', 'Na'))
    if species_label == 'Co_0_Kurucz':
        species_names.add(('Co_0_Kurucz', 'Co'))
    if species_label == 'Cr_1_VALD':
        species_names.add(('Cr_1_VALD', 'Cr'))
    if species_names == 'Fe_0_Kurucz':
        species_names.add(('Fe-Kurucz', 'Fe'))
    if species_label == 'Fe_1_Kurucz':
        species_names.add(('Fe_1_Kurucz', 'Fe+'))
    if species_label == 'Fe_0_Vald':
        species_names.add(('Fe_0_Vald', 'Fe'))
    if species_label == 'Mg_0_Kurucz':
        species_names.add(('Mg_0_Kurucz', 'Mg'))
    if species_label == 'Ni_0_Kurucz':
        species_names.add(('Ni_0_Kurucz', 'Ni'))
    if species_label == 'Ti_0_Kurucz':
        species_names.add(('Ti_0_Kurucz', 'Ti'))
    if species_label == 'Ti_1_Kurucz':
        species_names.add(('Ti_1_Kurucz', 'Ti+'))
    if species_label == 'Ti_0_VALD':
        species_names.add(('Ti_0_VALD', 'Ti'))
    if species_label == 'Ti1':
        species_names.add(('Ti_1_VALD', 'Ti+'))
    if species_label == 'Ti_1_Kurucz':
        species_names.add(('Ti_1_Kurucz', 'Ti+'))

    if species_label == 'Al I':
        species_names.add('Al')
    if species_label == 'B I':
        species_names.add('B')
    if species_label == 'Be I':
        species_names.add('Be')
        
    if not species_names:
        raise ValueError(f"Invalid species_label: {species_label}")
    species_names = list(set(species_names))  # Convert the set back to a list
    
    if type(species_names[0])  == str:
        species_name_inject = str(species_names[0])
        species_name_ccf = str(species_names[0])
    elif type(species_names[0]) == tuple:
        species_name_inject = str(species_names[0][1])
        species_name_ccf = str(species_names[0][0])
    
    return species_name_inject, species_name_ccf



def get_species_label(species_name, charge_state=None):
  """
  This function maps a species name and optionally its charge state to the corresponding species_label.

  Args:
      species_name: The name of the species (e.g., "Fe").
      charge_state: The charge state of the species (e.g., "+" for singly ionized). Defaults to None.

  Returns:
      The corresponding species_label or None if not found.
  """
  if charge_state:
    species_name += "_" + charge_state

  lookup = {
      "TiO_all_iso_Plez": "TiO",
      "TiO_46_Exomol_McKemmish": "TiO_46",
      "TiO_47_Exomol_McKemmish": "TiO_47",
      "TiO_48_Exomol_McKemmish": "TiO_48",
      "TiO_49_Exomol_McKemmish": "TiO_49",
      "TiO_50_Exomol_McKemmish": "TiO_50",
      "VO_ExoMol_McKemmish": "VO",
      "FeH_main_iso": "FeH",
      "CaH": "CaH",
      "Fe": "Fe I",
      "Ti": "Ti I",
      "Ti+": "Ti II",
      "Mg": "Mg I",
      "Mg+": "Mg II",
      "Fe+": "Fe II",
      "Cr": "Cr I",
      "Si": "Si I",
      "Ni": "Ni I",
      "Al": "Al I",
      "SiO_main_iso_new_incl_UV": "SiO",
      "H2O_main_iso": "H2O",
      "OH_main_iso": "OH",
      "MgH": "MgH",
      "Ca": "Ca I",
      "CO_all_iso": "CO_all",
      "CO_main_iso": "CO_main",
      "NaH": "NaH",
      "H": "H I",
      "AlO": "AlO",
      "Ba": "Ba I",
      "Ba+": "Ba II",
      "CaO": "CaO",
      "Co": "Co I",
      "Cr+": "Cr II",
      "Cs": "Cs I",
      "Cu": "Cu I",
      "Ga": "Ga I",
      "Ge": "Ge I",
      "Hf": "Hf I",
      "In": "In I",
      "Ir": "Ir I",
      "Mn": "Mn I",
      "Mo": "Mo I",
      "Na": "Na I",
      "Nb": "Nb I",
      "O": "O I",
      "Os": "Os I",
      "Pb": "Pb I",
      "Pd": "Pd I",
      "Rb": "Rb I",
      "Rh": "Rh I",
      "Ru": "Ru I",
      "Sc": "Sc I",
      "Sc+": "Sc II",
      "Sn": "Sn I",
      "Sr": "Sr I",
      "Sr+": "Sr II",
      "Tl": "Tl I",
      "W": "W I",
      "Y+": "Y II",
      "Zn": "Zn I",
      "Zr": "Zr I",
      "Zr+": "Zr II",
      "N": "N I",
      "K": "K I",
      "Y": "Y I",
      "Li": "Li I",
      "V": "V I",
      "V+": "V II",
      "Ca+": "Ca II",
      "Tl+": "Tl1",
      ('Na_allard_new', 'Na'): "Na_Allard",
      ('Na_burrows', 'Na'): "Na_Burrows",
      ('', 'Na'): "Na_lor_cut",
      ('Co_0_Kurucz', 'Co'): "Co_0_Kurucz",
      ('Cr_1_VALD', 'Cr'): "Cr_1_VALD",
      ('Fe-Kurucz', 'Fe'): "Fe_0_Kurucz",  # Handle the case where "Fe" maps to "Fe_0_Kurucz"
      ('Fe_1_Kurucz', 'Fe+'): "Fe_1_Kurucz",
      ('Fe_0_Vald', 'Fe'): "Fe_0_Vald",
      ('Mg_0_Kurucz', 'Mg'): "Mg_0_Kurucz",
      ('Ni_0_Kurucz', 'Ni'): "Ni_0_Kurucz",
      ('Ti_0_Kurucz', 'Ti'): "Ti_0_Kurucz",
      ('Ti_1_Kurucz', 'Ti+'): "Ti_1_Kurucz",
      ('Ti_0_VALD', 'Ti'): "Ti_0_VALD",
  }
  return lookup.get(species_name, None)


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
            n_systematics = [3, 0]
        if arm == 'red':
            n_systematics = [1, 1]

    return n_systematics

def get_planet_parameters(planet_name):

    MJoMS = 1./1047. #MJ in MSun
    
    if planet_name == 'KELT-20b':
        #For KELT-20 b:, from Lund et al. 2018
        Period = ufloat(3.4741070, 0.0000019)
        epoch = ufloat(2457503.120049, 0.000190)

        M_star = ufloat(1.76, 0.19) #MSun
        RV_abs = ufloat(0.0, 0.0) #km/s
        i = ufloat(86.12, 0.28) #degrees
        M_p = 3.382 #3-sigma limit
        R_p = 1.741

        RA = '19h38m38.74s'
        Dec = '+31d13m09.12s'

        dur = 0.14898 #hours -> days

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

    half_duration_phase = (dur/2.)/Period.n
    Kp_expected = 28.4329 * M_star/MJoMS * unp.sin(i*np.pi/180.) * (M_star + M_p * MJoMS) ** (-2./3.) * (Period/365.25) ** (-1./3.) / 1000. #to km/s
    half_duration_phase = (dur/2.)/Period.n

    return Period, epoch, M_star, RV_abs, i, M_p, R_p, RA, Dec, Kp_expected, half_duration_phase
    
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
        data_location = 'data/' + observation_epoch + '_' + planet_name + '/' + arm_file + '*.dxt.' + pepsi_extend
    else:
        data_location = 'data/' + observation_epoch + '_' + planet_name + '/molecfit_weak/SCIENCE_TELLURIC_CORR_' + arm_file + '*.dxt.' + pepsi_extend + '.fits'
    spectra_files = glob(data_location)

    n_spectra = len(spectra_files)
    i=0
    jd, snr_spectra, exptime = np.zeros(n_spectra), np.zeros(n_spectra), np.zeros(n_spectra,dtype=str)
    airmass = np.zeros(n_spectra)

    for spectrum in spectra_files:
        hdu = fits.open(spectrum)
        data, header = hdu[1].data, hdu[0].header
        if do_molecfit: wave_tag, flux_tag, error_tag = 'lambda', 'flux', 'error'
        if not do_molecfit: wave_tag, flux_tag, error_tag = 'Arg', 'Fun', 'Var'
        if i ==0:
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
        exptime[i] = header['EXPTIME'] #in decimal h/m/s. WHYYYYYY
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

    pl.fill([4800,4800,5441,5441],[np.nanmin(template_flux),np.nanmax(template_flux),np.nanmax(template_flux),np.nanmin(template_flux)],color='blue',alpha=0.25)
    pl.fill([6278,6278,7419,7419],[np.nanmin(template_flux),np.nanmax(template_flux),np.nanmax(template_flux),np.nanmin(template_flux)],color='red',alpha=0.25)

    pl.plot(template_wave, template_flux, color='black')

    pl.xlabel('wavelength (Angstroms)')

    pl.ylabel('normalized flux')

    pl.title(species_name_ccf)

    plotout = 'plots/spectrum.' + planet_name + '.' + species_name_ccf + '.' + str(vmr) + '.' + temperature_profile + '.pdf'
    pl.savefig(plotout,format='pdf')
    pl.clf()

def make_new_model(instrument, species_name_new, vmr, spectrum_type, planet_name, temperature_profile, do_plot=False):


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

    if 'Plez' in species_name_new or species_name_new == 'Fe+' or species_name_new == 'Ti+' or species_name_new == 'Cr':
        template_wave = vacuum2air(template_wave)

    if do_plot: make_spectrum_plot(template_wave, template_flux, planet_name, species_name_new, temperature_profile, vmr)   

    return template_wave, template_flux

def get_atmospheric_model(planet_name, species_name_ccf, vmr, temperature_profile, do_rotate, do_instrument):

    filein = 'templates/' + planet_name + '.' + species_name_ccf + '.' + str(vmr) + '.' + temperature_profile + '.combined.fits'
    hdu = fits.open(filein)

    template_wave = hdu[1].data['wave']
    template_flux = hdu[1].data['flux']

    hdu.close()

    template_flux = do_convolutions(planet_name, template_wave, template_flux, True, True)

    return template_wave, template_flux

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

    rvmin, rvmax = -400., 400. #kms
    rvspacing = 1.0 #kms

    for i in range (n_spectra):
        drv, cross_cor_out, sigma_cross_cor_out = ccf(wave, corrected_flux[i,:], corrected_error[i,:], template_wave, template_flux, rvmin, rvmax, rvspacing)
        if i == 0:
            cross_cor, sigma_cross_cor = np.zeros((n_spectra, len(drv))), np.zeros((n_spectra, len(drv)))
        cross_cor[i,:], sigma_cross_cor[i,:] = cross_cor_out, sigma_cross_cor_out

    return drv, cross_cor, sigma_cross_cor

def get_likelihood(wave, corrected_flux, corrected_error, template_wave, template_flux, n_spectra):

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

def combine_ccfs_binned(drv, cross_cor, sigma_cross_cor, orbital_phase, n_spectra, ccf_weights, half_duration_phase, temperature_profile, Kp_here, species_name_ccf, planet_name, arm):

    #if 'Fe' in species_name_ccf or 'Fe+' in species_name_ccf:
    #    binsize = 0.015
    #else:
    #    binsize = 0.05

    binsize = (np.max(orbital_phase) - np.min(orbital_phase))/len(orbital_phase)*3
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
        #if not 'transmission' in temperature_profile or np.abs(orbital_phase[j]) <= half_duration_phase:
        if np.abs(orbital_phase[j]) <= half_duration_phase:
            phase_here = np.argmin(np.abs(phase_bin - orbital_phase[j]))
            
            temp_ccf = np.interp(drv, drv-RV[j], cross_cor[j, :], left=0., right=0.0)
            sigma_temp_ccf = np.interp(drv, drv-RV[j], sigma_cross_cor[j, :], left=0., right=0.0)
            binned_ccfs[phase_here,:] += temp_ccf * ccf_weights[j]
            #use_for_sigma = (np.abs(drv) <= 100.) & (temp_ccf != 0.)
            use_for_sigma = (np.abs(drv) > 100.) & (temp_ccf != 0.)
            #this next is b/c the uncertainties produced through the ccf routine are just wrong
            var_shifted_ccfs[phase_here,:] += np.std(temp_ccf[use_for_sigma])**2 * ccf_weights[j]**2

    sigma_shifted_ccfs = np.sqrt(var_shifted_ccfs)

    #if planet_name == 'KELT-20b' and (species_name_ccf == 'Fe' or species_name_ccf == 'Fe+' or species_name_ccf == 'Ni'):
    if planet_name == 'KELT-20b':
        ecc = 0.019999438851877625#0.0037 + 0.010 * 3.0 #rough 3-sigma limit
        omega = 309.2455607770675#151.

        ftransit=np.pi/2.-omega*np.pi/180.#-np.pi #true anomaly at transit
        Etransit=2.*np.arctan(np.sqrt((1.-ecc)/(1.+ecc))*np.tan(ftransit/2.)) #eccentric anomaly at transit
        timesince=1.0/(2.*np.pi)*(Etransit-ecc*np.sin(Etransit)) #time since periastron to transit
        RVe = radvel.kepler.rv_drive(orbital_phase, np.array([1.0, 0.0-timesince, ecc, omega*np.pi/180.-np.pi, Kp_here]))

        RVdiff = RVe - RV
        order = np.argsort(orbital_phase)

    good = np.abs(drv) < 25.
    c = pl.pcolor(drv[good], phase_bin, binned_ccfs[:,good], edgecolors='none',rasterized=True, cmap='magma_r')
    pl.plot([0.,0.],[np.min(phase_bin), np.max(phase_bin)],':',color='white')
    #pl.plot(RVdiff[order], orbital_phase[order], '--', color='white')
    pl.colorbar(c)

    pl.xlabel('$\Delta V$ (km/s)')
    pl.ylabel('orbital phase')
    pl.savefig('plots/'+planet_name+'.'+species_name_ccf + '.' + arm + '.phase-binned.pdf', format='pdf')
    pl.clf()

    rvs, widths, rverrors, widtherrors = np.zeros(nphase), np.zeros(nphase), np.zeros(nphase), np.zeros(nphase)
    drvfit = drv[good]
    ccffit = binned_ccfs[:,good]
    sigmafit = sigma_shifted_ccfs[:,good]

    pp = PdfPages('plots/'+planet_name+'.'+species_name_ccf+ '.' + arm +'.phase-binned-RV-fits.pdf')

    for i in range (0, nphase):
        pl.subplot(3, 3, np.mod(i, 9)+1)
        peak = np.argmax(ccffit[i,:])
        popt, pcov = curve_fit(gaussian, drvfit, ccffit[i,:], p0=[ccffit[i,peak], drvfit[peak], 2.5], sigma = sigmafit[i,:], maxfev=1000000)

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

    use_for_snr = np.abs(drv > 100.)
    snr = binned_ccfs / np.std(binned_ccfs[:,use_for_snr])
    masked_snr = np.ma.masked_where(snr <= 3, snr)

    fig, ax = pl.subplots(layout='constrained', figsize=(10,8))
    #c = ax.pcolor(drv[good], phase_bin, masked_snr[:,good], edgecolors='none',rasterized=True, cmap='viridis_r')
    c = ax.pcolor(drv[good], phase_bin, snr[:,good], edgecolors='none',rasterized=True, cmap='viridis_r')
    ax.plot([0.,0.],[np.min(phase_bin), np.max(phase_bin)],':',color='grey')

    goodrv = (rvs > 0.) & (rvs < 25.)
    ax.plot(rvs[goodrv], phase_bin[goodrv], 'o', color='white')
    ax.errorbar(rvs[goodrv], phase_bin[goodrv], xerr = rverrors[goodrv], color='white', fmt='none')

    ax.set_xlabel('$\Delta V$ (km/s)')
    ax.set_ylabel('Orbital Phase (fraction)')

    # add title with species name above the plot
    ax.text(0.5, 1.03, species_name_ccf, transform=ax.transAxes, verticalalignment='top', horizontalalignment='left', fontsize=12)
    

    ax.set_xlim([-25.,25.])

    secax = ax.secondary_yaxis('right', functions = (phase2angle, angle2phase))
    secax.set_ylabel('Orbital Phase (degrees)')

    # add a color bar
    fig.colorbar(c, ax=ax, label='SNR ($\sigma$)')

    
    pl.savefig('plots/'+planet_name+'.'+species_name_ccf+ '.' + arm +'.phase-binned+RVs.pdf', format='pdf')
    pl.clf()
    
    Kp = np.arange(50, 350, 1)
    # Line Profile plot
    idx = np.where(Kp == int(np.floor(Kp_here)))[0][0] #Kp slice corresponding to expected Kp
    idx = np.argmax(slice_peak) #Kp slice corresponding to max SNR 

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
            popt, pcov = curve_fit(gaussian, drv, temp_ccf, p0=[temp_ccf[peak], drv[peak], 2.5], sigma = np.sqrt(sigma_temp_ccf), maxfev=1000000)
            rv_chars[i,phase_here] = popt[1]
            rv_chars_error[i,phase_here] = np.sqrt(pcov[1,1])
            slice_peak_chars[i] = temp_ccf[peak]
        i+=1

    fig, ax1 = pl.subplots(figsize=(8,8))

    ax1.text(0.05, 0.99, species_name_ccf, transform=ax1.transAxes, verticalalignment='top', horizontalalignment='left', fontsize=12)
    ax1.plot(rv_chars[idx,:], phase_array, '-', label='Center', color='b')
    ax1.fill_betweenx(phase_array, rv_chars[idx,:] - rv_chars_error[idx, :], rv_chars[idx,:] + rv_chars_error[idx,:], color='blue', alpha=0.2, zorder=2)
    ax1.set_ylabel('Orbital Phase (fraction)')
    ax1.set_xlabel('$\Delta V$ (km/s)', color='b')
    secax = ax1.secondary_yaxis('right', functions = (phase2angle, angle2phase))
    secax.set_ylabel('Orbital Phase (degrees)')
    
    # add a vertical line at 0km/s
    #ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)

    line_profile = 'plots/' + planet_name + '.' + 'combined' + '.' + 'combined' + '.' + species_name_ccf + '.line-profile-binned.pdf'
    fig.savefig(line_profile, dpi=300, bbox_inches='tight')
    pl.close(fig)

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

    Period, epoch, M_star, RV_abs, i, M_p, R_p, RA, Dec, Kp_expected, half_duration_phase = get_planet_parameters(planet_name)
    
    if observation_epoch != 'mock-obs':
        if arm == 'red' or arm == 'blue':
            wave, fluxin, errorin, jd, snr_spectra, exptime, airmass, n_spectra, npix = get_pepsi_data(arm, observation_epoch, planet_name, do_molecfit)

        else:
            # If arm is neither 'red' nor 'blue', use 'blue' as the default as do_molecfit will throw false when arm is 'combined'
            wave, fluxin, errorin, jd, snr_spectra, exptime, airmass, n_spectra, npix = get_pepsi_data('blue', observation_epoch, planet_name, do_molecfit)

    # Temporary hack for aliasing KELT-20b transmission spectra -- REMOVE THIS LATER
    else:
        wave, fluxin, errorin, jd, snr_spectra, exptime, airmass, n_spectra, npix = get_pepsi_data(arm, '20190504', planet_name, do_molecfit)
        

    orbital_phase = get_orbital_phase(jd, epoch, Period, RA, Dec)
    # Gaussian Fit plot
    # Initializing lists to store fit parameters
    amps, amps_error = np.zeros(plotsnr.shape[0]), np.zeros(plotsnr.shape[0])
    rv, rv_error = np.zeros(plotsnr.shape[0]), np.zeros(plotsnr.shape[0])
    width, width_error = np.zeros(plotsnr.shape[0]), np.zeros(plotsnr.shape[0])
    fwhm, fwhm_error = np.zeros(plotsnr.shape[0]), np.zeros(plotsnr.shape[0])

    slice_peak = np.zeros(plotsnr.shape[0])
    # Fitting gaussian to all 1D Kp slices
    Kp_valid = np.arange(int(np.floor(Kp_true)) - 50, int(np.floor(Kp_true)) + 50, 1)
    for i in Kp_valid:
        # Get the index of the peak with the maximum value within the desired range
        valid_peaks = np.abs(drv) <= 15
        peak = np.argmax(plotsnr[i, :] * valid_peaks)

        # If a valid peak was found, fit the Gaussian
        if valid_peaks[peak]:
            slice_peak[i] = plotsnr[i, peak]
            
            popt, pcov = curve_fit(gaussian, drv, plotsnr[i,:], p0=[plotsnr[i, peak], drv[peak], 2.55035], sigma = sigma_shifted_ccfs[i,:], maxfev=1000000)

            amps[i] = popt[0]
            rv[i] = popt[1]
            width[i] = popt[2]
            amps_error[i] = np.sqrt(pcov[0,0])
            rv_error[i] = np.sqrt(pcov[1,1])
            width_error[i] = np.sqrt(pcov[2,2])
            fwhm[i] = 2*np.sqrt(2*np.log(2))*width[i]
            fwhm_error[i] = 2*np.sqrt(2*np.log(2))*width_error[i]

            

    idx = np.flatnonzero(Kp == int(np.floor(Kp_true)))[0] #Kp slice corresponding to expected Kp
    idx = np.argmax(slice_peak) #Kp slice corresponding to max SNR 
                
    popt_selected = [amps[idx], rv[idx], width[idx]]
    print('Selected SNR:', amps[idx], '$/pm$', amps_error[idx],
          '\n Selected Vsys:', rv[idx], '$/pm$', rv_error[idx],
          '\n Selected sigma:', width[idx], '$/pm$', width_error[idx],
          '\n Selected Kp:', Kp[idx],
          '\n Selected FWHM:', fwhm[idx], '$/pm$', fwhm_error[idx]
    )

    # Computing residuals and chi-squared for selected slice
    residual = plotsnr[idx, :] - gaussian(drv, *popt_selected)
    # chi2 = np.sum((residual / np.std(residual))**2)/(len(drv)-len(popt))

    # Initialize Figure and GridSpec objects
    fig1 = pl.figure(figsize=(8,8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])

    # Create Axes for the main plot and the residuals plot
    ax1 = pl.subplot(gs[0])
    ax2 = pl.subplot(gs[1], sharex=ax1)

    plot_mask = np.abs(drv) <= 50.
    # Restrict arrays to the region of interest for plotting
    drv_restricted = drv[plot_mask]
    plotsnr_restricted = plotsnr[idx, plot_mask]
    residual_restricted = residual[plot_mask]
   # Main Plot (ax1)
    #ax1.plot(drv_restricted, plotsnr_restricted, 'k--', label='data', markersize=2)
    #ax1.plot(drv_restricted, gaussian(drv_restricted, *popt_selected), 'r-', label='fit')

    ax1.plot(drv, plotsnr[idx,:], 'k-', label='data', markersize=1)
    ax1.plot(drv, gaussian(drv, *popt_selected), 'r--', label='fit', linewidth=0.66)

    ax1.set_xlim([-100, 100])
    # Species Label
    ax1.text(0.05, 0.99, species_label, transform=ax1.transAxes, verticalalignment='top', horizontalalignment='left', fontsize=12)

    pl.setp(ax1.get_xticklabels(), visible=False)
    ax1.set_ylabel('SNR ($\sigma$)')
    # Annotating the arm and species on the plot

    # Additional text information for the main plot
    #params_str = f"Peak (a): {popt_selected[0]:.2f}\nMean (mu): {popt_selected[1]:.2f}\nSigma: {popt_selected[2]:.2f}\nKp: {Kp[idx]:.0f}"
    #ax1.text(0.01, 0.95, params_str, transform=ax1.transAxes, verticalalignment='top', fontsize=10)

    arm_species_text = f'Arm: {arm}'
    ax1.text(0.15, 0.95, arm_species_text, transform=ax1.transAxes, verticalalignment='top', fontsize=10)

    # Vertical line for the Gaussian peak center
    ax1.axvline(x=rv[idx], color='b', linestyle='-', label='Center')
    #ax1.set_title('1D CCF Slice + Gaussian Fit')

    # Vertical lines for sigma width (center Â± sigma)
    #sigma_left = rv[idx] - width[idx]
    #sigma_right = rv[idx] + width[idx]
    #ax1.axvline(x=sigma_left, color='purple', linestyle='--', label='- Sigma')
    #ax1.axvline(x=sigma_right, color='purple', linestyle='--', label='+ Sigma')

    #ax1.legend()

    # Add the horizontal line at 4 SNR
    ax1.axhline(y=4, color='g', linestyle='--', label=r'4 $\sigma$')    

    # Inset for residuals (ax2)
    ax2.plot(drv, residual, 'o-', markersize=1)
    ax2.set_xlim([-100, 100])

    ax2.set_xlabel('$\Delta V$ (km/s)')
    ax2.set_ylabel('Residuals')

    # Consider a clearer naming scheme
    snr_fit = 'plots/'+ planet_name + '.' + observation_epoch + '.' + arm + '.' + species_name_ccf + model_tag + '.SNR-Gaussian.pdf'
    # Save the plot
    fig1.savefig(snr_fit, dpi=300, bbox_inches='tight')
    pl.close(fig1)

    # Line Profile plot
    idx = np.where(Kp == int(np.floor(Kp_true)))[0][0] #Kp slice corresponding to expected Kp
    idx = np.argmax(slice_peak) #Kp slice corresponding to max SNR 

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
            peak = np.argmax(temp_ccf[385:416]) + 385
            sigma_temp_ccf = np.interp(drv, drv-RV[j], sigma_cross_cor[j, :], left=0., right=0.0)
            sigma_temp_ccf = sigma_temp_ccf**2 * ccf_weights[j]**2
            popt, pcov = curve_fit(gaussian, drv, temp_ccf, p0=[temp_ccf[peak], drv[peak], 2.5], sigma = np.sqrt(sigma_temp_ccf), maxfev=1000000)
            rv_chars[i,phase_here] = popt[1]
            rv_chars_error[i,phase_here] = np.sqrt(pcov[1,1])
            slice_peak_chars[i] = temp_ccf[peak]
        i+=1

    fig2, ax3 = pl.subplots(figsize=(8,8))

    ax3.text(0.05, 0.99, species_label, transform=ax3.transAxes, verticalalignment='top', horizontalalignment='left', fontsize=12)
    ax3.plot(rv_chars[idx,:], phase_array, '-', label='Center')
    ax3.fill_betweenx(phase_array, rv_chars[idx,:] - rv_chars_error[idx, :], rv_chars[idx,:] + rv_chars_error[idx,:], color='blue', alpha=0.2, zorder=2)
    ax3.set_ylabel('Orbital Phase (fraction)')
    ax3.set_xlabel('$\Delta V$ (km/s)', color='b')
    ax3.tick_params(axis='x', labelcolor='b')
    
    secax = ax3.secondary_yaxis('right', functions = (phase2angle, angle2phase))
    secax.set_ylabel('Orbital phase (degrees)')
    
    # add a vertical line at 0km/s
    ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    line_profile = 'plots/' + planet_name + '.' + observation_epoch + '.' + arm + '.' + species_name_ccf + model_tag + '.line-profile.pdf'
    fig2.savefig(line_profile, dpi=300, bbox_inches='tight')

    pl.close(fig2)

    return amps, amps_error, rv, rv_error, width, width_error, residual, do_molecfit, idx, line_profile, drv_restricted, plotsnr_restricted, residual_restricted, fig1, ax1, ax2, fig2, ax3


def gaussian_fit(Kp, Kp_true, drv, species_label, planet_name, observation_epoch, arm, species_name_ccf, model_tag, plotsnr, sigma_shifted_ccfs, temperature_profile, cross_cor, sigma_cross_cor, ccf_weights):
 
    if arm == 'red':
        do_molecfit = True
    else:
        do_molecfit = False

    Period, epoch, M_star, RV_abs, i, M_p, R_p, RA, Dec, Kp_expected, half_duration_phase = get_planet_parameters(planet_name)
    
    if observation_epoch != 'mock-obs':
        if arm == 'red' or arm == 'blue':
            wave, fluxin, errorin, jd, snr_spectra, exptime, airmass, n_spectra, npix = get_pepsi_data(arm, observation_epoch, planet_name, do_molecfit)

        else:
            # If arm is neither 'red' nor 'blue', use 'blue' as the default as do_molecfit will throw false when arm is 'combined'
            wave, fluxin, errorin, jd, snr_spectra, exptime, airmass, n_spectra, npix = get_pepsi_data('blue', observation_epoch, planet_name, do_molecfit)

    # Temporary hack for aliasing KELT-20b transmission spectra -- REMOVE THIS LATER
    else:
        wave, fluxin, errorin, jd, snr_spectra, exptime, airmass, n_spectra, npix = get_pepsi_data(arm, '20190504', planet_name, do_molecfit)
        

    orbital_phase = get_orbital_phase(jd, epoch, Period, RA, Dec)
    # Gaussian Fit plot
    # Initializing lists to store fit parameters
    amps, amps_error = np.zeros(plotsnr.shape[0]), np.zeros(plotsnr.shape[0])
    rv, rv_error = np.zeros(plotsnr.shape[0]), np.zeros(plotsnr.shape[0])
    width, width_error = np.zeros(plotsnr.shape[0]), np.zeros(plotsnr.shape[0])
    fwhm, fwhm_error = np.zeros(plotsnr.shape[0]), np.zeros(plotsnr.shape[0])

    slice_peak = np.zeros(plotsnr.shape[0])
    # Fitting gaussian to all 1D Kp slices
    Kp_valid = np.arange(int(np.floor(Kp_true)) - 50, int(np.floor(Kp_true)) + 50, 1)
    for i in Kp_valid:
        # Get the index of the peak with the maximum value within the desired range
        valid_peaks = np.abs(drv) <= 15
        peak = np.argmax(plotsnr[i, :] * valid_peaks)

        # If a valid peak was found, fit the Gaussian
        if valid_peaks[peak]:
            slice_peak[i] = plotsnr[i, peak]
            
            popt, pcov = curve_fit(gaussian, drv, plotsnr[i,:], p0=[plotsnr[i, peak], drv[peak], 2.55035], sigma = sigma_shifted_ccfs[i,:], maxfev=1000000)

            amps[i] = popt[0]
            rv[i] = popt[1]
            width[i] = popt[2]
            amps_error[i] = np.sqrt(pcov[0,0])
            rv_error[i] = np.sqrt(pcov[1,1])
            width_error[i] = np.sqrt(pcov[2,2])
            fwhm[i] = 2*np.sqrt(2*np.log(2))*width[i]
            fwhm_error[i] = 2*np.sqrt(2*np.log(2))*width_error[i]

            

    idx = np.flatnonzero(Kp == int(np.floor(Kp_true)))[0] #Kp slice corresponding to expected Kp
    idx = np.argmax(slice_peak) #Kp slice corresponding to max SNR 
                
    popt_selected = [amps[idx], rv[idx], width[idx]]
    print('Selected SNR:', amps[idx], '$/pm$', amps_error[idx],
          '\n Selected Vsys:', rv[idx], '$/pm$', rv_error[idx],
          '\n Selected sigma:', width[idx], '$/pm$', width_error[idx],
          '\n Selected Kp:', Kp[idx],
          '\n Selected FWHM:', fwhm[idx], '$/pm$', fwhm_error[idx]
    )

    # Computing residuals and chi-squared for selected slice
    residual = plotsnr[idx, :] - gaussian(drv, *popt_selected)
    # chi2 = np.sum((residual / np.std(residual))**2)/(len(drv)-len(popt))

    # Initialize Figure and GridSpec objects
    fig1 = pl.figure(figsize=(8,8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])

    # Create Axes for the main plot and the residuals plot
    ax1 = pl.subplot(gs[0])
    ax2 = pl.subplot(gs[1], sharex=ax1)

    plot_mask = np.abs(drv) <= 50.
    # Restrict arrays to the region of interest for plotting
    drv_restricted = drv[plot_mask]
    plotsnr_restricted = plotsnr[idx, plot_mask]
    residual_restricted = residual[plot_mask]
   # Main Plot (ax1)
    #ax1.plot(drv_restricted, plotsnr_restricted, 'k--', label='data', markersize=2)
    #ax1.plot(drv_restricted, gaussian(drv_restricted, *popt_selected), 'r-', label='fit')

    ax1.plot(drv, plotsnr[idx,:], 'k-', label='data', markersize=1)
    ax1.plot(drv, gaussian(drv, *popt_selected), 'r--', label='fit', linewidth=0.66)

    ax1.set_xlim([-100, 100])
    # Species Label
    ax1.text(0.05, 0.99, species_label, transform=ax1.transAxes, verticalalignment='top', horizontalalignment='left', fontsize=12)

    pl.setp(ax1.get_xticklabels(), visible=False)
    ax1.set_ylabel('SNR ($\sigma$)')
    # Annotating the arm and species on the plot

    # Additional text information for the main plot
    #params_str = f"Peak (a): {popt_selected[0]:.2f}\nMean (mu): {popt_selected[1]:.2f}\nSigma: {popt_selected[2]:.2f}\nKp: {Kp[idx]:.0f}"
    #ax1.text(0.01, 0.95, params_str, transform=ax1.transAxes, verticalalignment='top', fontsize=10)

    arm_species_text = f'Arm: {arm}'
    ax1.text(0.15, 0.95, arm_species_text, transform=ax1.transAxes, verticalalignment='top', fontsize=10)

    # Vertical line for the Gaussian peak center
    ax1.axvline(x=rv[idx], color='b', linestyle='-', label='Center')
    #ax1.set_title('1D CCF Slice + Gaussian Fit')

    # Vertical lines for sigma width (center Â± sigma)
    #sigma_left = rv[idx] - width[idx]
    #sigma_right = rv[idx] + width[idx]
    #ax1.axvline(x=sigma_left, color='purple', linestyle='--', label='- Sigma')
    #ax1.axvline(x=sigma_right, color='purple', linestyle='--', label='+ Sigma')

    #ax1.legend()

    # Add the horizontal line at 4 SNR
    ax1.axhline(y=4, color='g', linestyle='--', label=r'4 $\sigma$')    

    # Inset for residuals (ax2)
    ax2.plot(drv, residual, 'o-', markersize=1)
    ax2.set_xlim([-100, 100])

    ax2.set_xlabel('$\Delta V$ (km/s)')
    ax2.set_ylabel('Residuals')

    # Consider a clearer naming scheme
    snr_fit = 'plots/'+ planet_name + '.' + observation_epoch + '.' + arm + '.' + species_name_ccf + model_tag + '.SNR-Gaussian.pdf'
    # Save the plot
    fig1.savefig(snr_fit, dpi=300, bbox_inches='tight')
    pl.close(fig1)

    # Line Profile plot
    idx = np.where(Kp == int(np.floor(Kp_true)))[0][0] #Kp slice corresponding to expected Kp
    idx = np.argmax(slice_peak) #Kp slice corresponding to max SNR 

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
            peak = np.argmax(temp_ccf[385:416]) + 385
            sigma_temp_ccf = np.interp(drv, drv-RV[j], sigma_cross_cor[j, :], left=0., right=0.0)
            sigma_temp_ccf = sigma_temp_ccf**2 * ccf_weights[j]**2
            popt, pcov = curve_fit(gaussian, drv, temp_ccf, p0=[temp_ccf[peak], drv[peak], 2.5], sigma = np.sqrt(sigma_temp_ccf), maxfev=1000000)
            rv_chars[i,phase_here] = popt[1]
            rv_chars_error[i,phase_here] = np.sqrt(pcov[1,1])
            slice_peak_chars[i] = temp_ccf[peak]
        i+=1

    fig2, ax3 = pl.subplots(figsize=(8,8))

    ax3.text(0.05, 0.99, species_label, transform=ax3.transAxes, verticalalignment='top', horizontalalignment='left', fontsize=12)
    ax3.plot(rv_chars[idx,:], phase_array, '-', label='Center')
    ax3.fill_betweenx(phase_array, rv_chars[idx,:] - rv_chars_error[idx, :], rv_chars[idx,:] + rv_chars_error[idx,:], color='blue', alpha=0.2, zorder=2)
    ax3.set_ylabel('Orbital Phase (fraction)')
    ax3.set_xlabel('$\Delta V$ (km/s)', color='b')
    ax3.tick_params(axis='x', labelcolor='b')
    
    secax = ax3.secondary_yaxis('right', functions = (phase2angle, angle2phase))
    secax.set_ylabel('Orbital phase (degrees)')
    
    # add a vertical line at 0km/s
    ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    line_profile = 'plots/' + planet_name + '.' + observation_epoch + '.' + arm + '.' + species_name_ccf + model_tag + '.line-profile.pdf'
    fig2.savefig(line_profile, dpi=300, bbox_inches='tight')

    pl.close(fig2)

    return amps, amps_error, rv, rv_error, width, width_error, residual, do_molecfit, idx, line_profile, drv_restricted, plotsnr_restricted, residual_restricted, fig1, ax1, ax2, fig2, ax3

def make_shifted_plot(snr, planet_name, observation_epoch, arm, species_name_ccf, model_tag, RV_abs, Kp_expected, V_sys_true, Kp_true, do_inject_model, drv, Kp, species_label, temperature_profile, sigma_shifted_ccfs, method, cross_cor_display, sigma_cross_cor, ccf_weights, plotformat = 'pdf'):
    
    if method == 'ccf':
        outtag, zlabel = 'CCFs-shifted', 'SNR'
        plotsnr = snr[:]
    if 'likelihood' in method:
        outtag, zlabel = 'likelihood-shifted', '$\Delta\ln \mathcal{L}$'
        plotsnr=snr - np.max(snr)
    plotname = 'plots/'+ planet_name + '.' + observation_epoch + '.' + arm + '.' + species_name_ccf + model_tag + '.' + outtag + '.' + plotformat

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
    
    mask = np.abs(drv) <= 100.
    
    drv_masked, plotsnr_masked = drv[mask], plotsnr[:, mask]

    plotsnr, Kp = plotsnr[keepKp, :], Kp[keepKp]
    
    amps, amps_error, rv, rv_error, width, width_error, residual, do_molecfit, idx, line_profile, drv_restricted, plotsnr_restricted, residual_restricted, fig1, ax1, ax2, fig2, ax3 = gaussian_fit(Kp, Kp_true, drv, species_label, planet_name, observation_epoch, arm, species_name_ccf, model_tag, plotsnr, sigma_shifted_ccfs, temperature_profile, cross_cor_display, sigma_cross_cor, ccf_weights)
    
    psarr(plotsnr_masked, drv_masked, Kp, '$RV - V_{sys}$ (km/s)', '$K_p$ (km/s)', zlabel, filename=plotname, ctable=ctable, alines=True, apoints=apoints, acolor='white', textstr=species_label+' '+model_label, textloc = np.array([apoints[0]-75.,apoints[1]+75.]), textcolor='black', fileformat=plotformat)
    fig, axs = pl.subplots(2, sharex=True)
    
    return plotsnr, amps, amps_error, rv, rv_error, width, width_error, idx, drv_restricted, plotsnr_restricted, residual_restricted, pl
    
    
def dopplerShadowRemove(drv, planet_name, exptime, orbital_phase, obs, inputs = {
                                    'mode':'spec',  
                                    'res':'high', 
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
            vsini = 117.4
            lambda_p = 3.4  

        Resolve = resolve_mapping.get(obs, 0.0)  # Default value of 0.0 if obs is not found in the mapping
        
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
                'lambda':lambda_p,  # projected_obliquity (deg)
                'b':0.503,       # transit impact parameter, hardcoded
                'rplanet': 0.11440, # the Rp/Rstar value for the transit. hardcoded
                't':orbital_phase * Period.n * 24*60 
                , # minutes since center of transit
                'times': np.float64(exptime) * 1/60,    # exposure time (in minutes), array length must match 't'    
                'a': 7.42,     # scaled semimajor axis of the orbit, a/Rstar, hardcoded?
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
                'logg': 4.290,  # stellar logg
                'rstar': 1.565, # called Reqcm in docs, stellar radius (solar radii)
                'f': 1.0    ,
                'psi': 35.6     # doesn't say in docs to input this
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

def run_one_ccf(species_label, vmr, arm, observation_epoch, template_wave, template_flux, template_wave_in, template_flux_in, planet_name, temperature_profile, do_inject_model, species_name_ccf, model_tag, f, method, do_make_new_model):

    

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

    #Make some diagnostic plots
    plotname = 'plots/' + planet_name + '.' + observation_epoch + '.' + arm + '.spectrum-resids.pdf'

    psarr(unp.nominal_values(residual_flux), wave, orbital_phase, 'wavelength (Angstroms)', 'orbital phase', 'flux residual', filename=plotname,flat=True, ctable='gist_gray')

    sysrem_file = 'data_products/' + planet_name + '.' + observation_epoch + '.' + arm + '.SYSREM-' + str(n_systematics[0]) + '+' + str(n_systematics[1])+model_tag+'.npy'

    
    if do_sysrem:
        
        corrected_flux, corrected_error = do_sysrem(wave, residual_flux, arm, airmass, n_spectra, niter, n_systematics, do_molecfit)
        #corrected_flux, corrected_error = unp.nominal_values(residual_flux), unp.std_devs(residual_flux)

        np.save(sysrem_file, corrected_flux)
        np.save(sysrem_file+'.corrected-error.npy', corrected_error)

        plotname = 'plots/' + planet_name + '.' + observation_epoch + '.' + arm + '.spectrum-SYSREM-'+ str(n_systematics[0]) + '+' + str(n_systematics[1])+'.pdf'

        psarr(corrected_flux, wave, orbital_phase, 'wavelength (Angstroms)', 'orbital phase', 'flux residual', filename=plotname,flat=True, ctable='gist_gray')

    else:
        corrected_flux = np.load(sysrem_file)
        corrected_error = np.load(sysrem_file+'.corrected-error.npy')

    if method == 'ccf':
        ccf_file = 'data_products/'+ planet_name + '.' + observation_epoch + '.' + arm + '.' + species_name_ccf + model_tag + '.CCFs-raw.npy'

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

        ccf_model = dopplerShadowRemove(drv, planet_name, exptime, orbital_phase, 'pepsi')
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
        # Subtract off
        ccf_model *= scale_factor
        cross_cor -= ccf_model
        
        #Make a plot
        plotname = 'plots/' + planet_name + '.' + observation_epoch + '.' + species_name_ccf + model_tag + '.CCFs-raw.pdf'
        psarr(cross_cor, drv, orbital_phase, 'v (km/s)', 'orbital phase', 'SNR', filename=plotname, ctable='gist_gray')

    
        snr, Kp, drv, cross_cor_display, sigma_shifted_ccfs, ccf_weights = combine_ccfs(drv, cross_cor, sigma_cross_cor, orbital_phase, n_spectra, ccf_weights, half_duration_phase, temperature_profile)

        
        
        plotsnr, amps, amps_error, rv, rv_error, width, width_error, selected_idx, drv_restricted, plotsnr_restricted, residual_restricted, pl = make_shifted_plot(snr, planet_name, observation_epoch, arm, species_name_ccf, model_tag, RV_abs, Kp_expected, V_sys_true, Kp_true, do_inject_model, drv, Kp, species_label, temperature_profile, sigma_shifted_ccfs, method, cross_cor_display, sigma_cross_cor, ccf_weights)

        get_peak_snr(snr, drv, Kp, do_inject_model, V_sys_true, Kp_true, RV_abs, Kp_expected, arm, observation_epoch, f, method)


    if 'likelihood' in method:
        like_file = 'data_products/'+ planet_name + '.' + observation_epoch + '.' + arm + '.' + species_name_ccf + model_tag + '.likelihood-raw.npy'
        drv, lnL = get_likelihood(wave, corrected_flux, corrected_error, template_wave, template_flux, n_spectra)

        np.save(like_file, lnL)
        np.save(like_file+'.phase.npy', orbital_phase)

        #Make a plot
        plotname = 'plots/' + planet_name + '.' + observation_epoch + '.' + species_name_ccf + model_tag + '.likelihoods-raw.pdf'
        psarr(lnL, drv, orbital_phase, 'v (km/s)', 'orbital phase', 'ln L', filename=plotname, ctable='gist_gray')

        #now need to combine the likelihoods along the planet orbit
        shifted_lnL, Kp, drv = combine_likelihoods(drv, lnL, orbital_phase, n_spectra, half_duration_phase, temperature_profile)

        plotsnr, amps, amps_error, rv, rv_error, width, width_error, selected_idx, drv_restricted, plotsnr_restricted, residual_restricted, pl = make_shifted_plot(snr, planet_name, observation_epoch, arm, species_name_ccf, model_tag, RV_abs, Kp_expected, V_sys_true, Kp_true, do_inject_model, drv, Kp, species_label, temperature_profile, sigma_shifted_ccfs, method, cross_cor_display, sigma_cross_cor, ccf_weights)

    return Kp, Kp_true, drv, species_label, planet_name, observation_epoch, arm, species_name_ccf, model_tag, plotsnr, cross_cor_display, sigma_cross_cor, orbital_phase, n_spectra, ccf_weights, half_duration_phase, temperature_profile, sigma_shifted_ccfs, amps, amps_error, rv, rv_error, width, width_error, selected_idx, drv_restricted, plotsnr_restricted, residual_restricted

def combine_observations(observation_epochs, arms, planet_name, temperature_profile, species_label, species_name_ccf, model_tag, RV_abs, Kp_expected, do_inject_model, f, method):

    Period, epoch, M_star, RV_abs, i, M_p, R_p, RA, Dec, Kp_expected, half_duration_phase = get_planet_parameters(planet_name)

    j=0
    for observation_epoch in observation_epochs:
        for arm in arms:
            if 'likelihood' in method:
                ccf_file_2 = 'data_products/'+ planet_name + '.' + observation_epoch + '.' + arm + '.' + species_name_ccf + model_tag + '.likelihood-raw.npy'
            if method == 'ccf':
                ccf_file_2 = 'data_products/'+ planet_name + '.' + observation_epoch + '.' + arm + '.' + species_name_ccf + model_tag + '.CCFs-raw.npy'
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
        snr, Kp, drv, cross_cor_display, sigma_shifted_ccfs, ccf_weights = combine_ccfs(drv, cross_cor, sigma_cross_cor, orbital_phase, len(orbital_phase), ccf_weights, half_duration_phase, temperature_profile)
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
            
    plotsnr, amps, amps_error, rv, rv_error, width, width_error, selected_idx, drv_restricted, plotsnr_restricted, residual_restricted, pl = make_shifted_plot(snr, planet_name, observation_epoch, arm, species_name_ccf, model_tag, RV_abs, Kp_expected, V_sys_true, Kp_true, do_inject_model, drv, Kp, species_label, temperature_profile, sigma_shifted_ccfs, method, cross_cor_display, sigma_cross_cor, ccf_weights)


    get_peak_snr(snr, drv, Kp, do_inject_model, V_sys_true, Kp_true, RV_abs, Kp_expected, all_arms, all_epochs, f, method)

    return Kp_true, orbital_phase, plotsnr, amps, amps_error, rv, rv_error, width, width_error, selected_idx, drv_restricted, plotsnr_restricted, residual_restricted

def run_all_ccfs(planet_name, temperature_profile, species_label, vmr, do_inject_model, do_run_all, do_make_new_model, method):
    
    fit_params = {}

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

    file_out = 'logs/'+ planet_name + '.' + species_name_ccf + model_tag + '.log'
    f = open(file_out,'w')

    f.write('Log file for ' + planet_name + ' for ' + species_name_ccf + ' \n')

    if do_make_new_model:
        template_wave, template_flux = make_new_model(instrument, species_name_ccf, vmr, spectrum_type, planet_name, temperature_profile, do_plot=True)
    else:
        template_wave, template_flux = get_atmospheric_model(planet_name, species_name_ccf, vmr, temperature_profile, True, True)

    # This may break an injection of an ionized species into a neutral species??
    if species_name_ccf[:2] != species_name_inject[:2]:
        if do_make_new_model:
            template_wave_in, template_flux_in, pressures, atmosphere, parameters = make_new_model(instrument, species_name_inject, vmr, spectrum_type, planet_name, temperature_profile)
        else:
            template_wave_in, template_flux_in = get_atmospheric_model(planet_name, species_name_inject, vmr, temperature_profile, True, True)
    else:
        template_wave_in, template_flux_in = template_wave, template_flux

    # Ensure species_label key is initialized in fit_params
    if species_label not in fit_params:
        fit_params[species_label] = {}
    if do_run_all:
        for observation_epoch in observation_epochs:
            for arm in arms:
                print('Now running the ',arm,' data for ',observation_epoch)
                Kp, Kp_true, drv, species_label, planet_name, observation_epoch, arm, species_name_ccf, model_tag, plotsnr, cross_cor_display, sigma_cross_cor, orbital_phase, n_spectra, ccf_weights, half_duration_phase, temperature_profile, sigma_shifted_ccfs, amps, amps_error, rv, rv_error, width, width_error, selected_idx, drv_restricted, plotsnr_restricted, residual_restricted = run_one_ccf(species_label, vmr, arm, observation_epoch, template_wave, template_flux, template_wave_in, template_flux_in, planet_name, temperature_profile, do_inject_model, species_name_ccf, model_tag, f, method, do_make_new_model)

               # Ensure observation_epoch key is initialized in fit_params[species_label]
                if observation_epoch not in fit_params[species_label]:
                    fit_params[species_label][observation_epoch] = {}
                # Assign the values

                fit_params[species_label][observation_epoch][arm] = {
                    'amps': amps,
                    'amps_error': amps_error,
                    'rv': rv,
                    'rv_error': rv_error,
                    'width': width,
                    'width_error': width_error,
                    'selected_idx': selected_idx,
                    'orbital_phase': orbital_phase,
                    'drv_restricted': drv_restricted,
                    'plotsnr_restricted': plotsnr_restricted,
                    'residual_restricted': residual_restricted
                }

    print('Now combining all of the data')

    Period, epoch, M_star, RV_abs, i, M_p, R_p, RA, Dec, Kp_expected, half_duration_phase = get_planet_parameters(planet_name)

    
    
    Kp_true, orbital_phase, plotsnr, amps, amps_error, rv, rv_error, width, width_error, selected_idx, drv_restricted, plotsnr_restricted, residual_restricted = combine_observations(observation_epochs, arms, planet_name, temperature_profile, species_label, species_name_ccf, model_tag, RV_abs, Kp_expected, do_inject_model, f, method)

    
    fit_params[species_label]['combined'] = {}
    fit_params[species_label]['combined']['combined'] = {
        'amps': amps,
        'amps_error': amps_error,
        'rv': rv,
        'rv_error': rv_error,
        'width': width,
        'width_error': width_error,
        'selected_idx': selected_idx,
        'orbital_phase': orbital_phase,
        'drv_restricted': drv_restricted,
        'plotsnr_restricted': plotsnr_restricted,
        'residual_restricted': residual_restricted
    }
    
    f.close()

    np.save('data_products/' + planet_name + '.' + observation_epoch + '.' + species_label + '.' + 'fit_params.npy', fit_params)

    return fit_params, observation_epochs, plotsnr_restricted 
    
    
    
    
    # Adopted from https://github.com/Andrew-S-Rosen/periodic_trends/blob/master/periodic_trends.py
def plotter(
    filename: str,
    show: bool = True,
    output_filename: str = None,
    width: int = 1050,
    cmap: str = "plasma",
    alpha: float = 0.65,
    extended: bool = True,
    periods_remove: List[int] = None,
    groups_remove: List[int] = None,
    log_scale: bool = False,
    cbar_height: float = None,
    cbar_standoff: int = 12,
    cbar_fontsize: int = 14,
    blank_color: str = "#c4c4c4",
    under_value: float = None,
    under_color: str = "#140F0E",
    over_value: float = None,
    over_color: str = "#140F0E",
    special_elements: List[str] = None,
    special_color: str = "#6F3023",
) -> figure:

    """
    Plot a heatmap over the periodic table of elements.

    Parameters
    ----------
    filename : str
        Path to the .csv file containing the data to be plotted.
    show : str
        If True, the plot will be shown.
    output_filename : str
        If not None, the plot will be saved to the specified (.html) file.
    width : float
        Width of the plot.
    cmap : str
        plasma, inferno, viridis, magma, cividis, turbo
    alpha : float
        Alpha value (transparency).
    extended : bool
        If True, the lanthanoids and actinoids will be shown.
    periods_remove : List[int]
        Period numbers to be removed from the plot.
    groups_remove : List[int]
        Group numbers to be removed from the plot.
    log_scale : bool
        If True, the colorbar will be logarithmic.
    cbar_height : int
        Height of the colorbar.
    cbar_standoff : int
        Distance between the colorbar and the plot.
    cbar_fontsize : int
        Fontsize of the colorbar label.
    blank_color : str
        Hexadecimal color of the elements without data.
    under_value : float
        Values <= under_value will be colored with under_color.
    under_color : str
        Hexadecimal color to be used for the lower bound color.
    over_value : float
        Values >= over_value will be colored with over_color.
    under_color : str
        Hexadecial color to be used for the upper bound color.
    special_elements: List[str]
        List of elements to be colored with special_color.
    special_color: str
        Hexadecimal color to be used for the special elements.

    Returns
    -------
    figure
        Bokeh figure object.
    """

    options.mode.chained_assignment = None

    # Assign color palette based on input argument
    if cmap == "plasma":
        cmap = plasma
        bokeh_palette = "Plasma256"
    elif cmap == "inferno":
        cmap = inferno
        bokeh_palette = "Inferno256"
    elif cmap == "magma":
        cmap = magma
        bokeh_palette = "Magma256"
    elif cmap == "viridis":
        cmap = viridis
        bokeh_palette = "Viridis256"
    elif cmap == "cividis":
        cmap = cividis
        bokeh_palette = "Cividis256"
    elif cmap == "turbo":
        cmap = turbo
        bokeh_palette = "Turbo256"
    else:
        ValueError("Invalid color map.")

    # Define number of and groups
    period_label = ["1", "2", "3", "4", "5", "6", "7"]
    group_range = [str(x) for x in range(1, 19)]

    # Remove any groups or periods
    if groups_remove:
        for gr in groups_remove:
            gr = gr.strip()
            group_range.remove(str(gr))
    if periods_remove:
        for pr in periods_remove:
            pr = pr.strip()
            period_label.remove(str(pr))

    # Read in data from CSV file
    data_elements = []
    data_list = []
    for row in reader(open(filename)):
        data_elements.append(row[0])
        data_list.append(row[1])
    data = [float(i) for i in data_list]

    if len(data) != len(data_elements):
        raise ValueError("Unequal number of atomic elements and data points")

    period_label.append("blank")
    period_label.append("La")
    period_label.append("Ac")

    if extended:
        count = 0
        for i in range(56, 70):
            elements.period[i] = "La"
            elements.group[i] = str(count + 4)
            count += 1

        count = 0
        for i in range(88, 102):
            elements.period[i] = "Ac"
            elements.group[i] = str(count + 4)
            count += 1

    # Define matplotlib and bokeh color map
    if log_scale:
        for datum in data:
            if datum < 0:
                raise ValueError(
                    f"Entry for element {datum} is negative but log-scale is selected"
                )
        color_mapper = LogColorMapper(
            palette=bokeh_palette, low=min(data), high=max(data)
        )
        norm = LogNorm(vmin=min(data), vmax=max(data))
    else:
        color_mapper = LinearColorMapper(
            palette=bokeh_palette, low=min(data), high=max(data)
        )
        norm = Normalize(vmin=min(data), vmax=max(data))
    color_scale = ScalarMappable(norm=norm, cmap=cmap).to_rgba(data, alpha=None)

    # Set blank color
    color_list = [blank_color] * len(elements)

    # Compare elements in dataset with elements in periodic table
    for i, data_element in enumerate(data_elements):
        element_entry = elements.symbol[
            elements.symbol.str.lower() == data_element.lower()
        ]
        if element_entry.empty == False:
            element_index = element_entry.index[0]
        else:
            warnings.warn("Invalid chemical symbol: " + data_element)
        if color_list[element_index] != blank_color:
            warnings.warn("Multiple entries for element " + data_element)
        elif under_value is not None and data[i] <= under_value:
            color_list[element_index] = under_color
        elif over_value is not None and data[i] >= over_value:
            color_list[element_index] = over_color
        else:
            color_list[element_index] = to_hex(color_scale[i])

    if special_elements:
        for k, v in elements["symbol"].iteritems():
            if v in special_elements:
                color_list[k] = special_color

    # Define figure properties for visualizing data
    source = ColumnDataSource(
        data=dict(
            group=[str(x) for x in elements["group"]],
            period=[str(y) for y in elements["period"]],
            sym=elements["symbol"],
            atomic_number=elements["atomic number"],
            type_color=color_list,
        )
    )

    # Plot the periodic table
    p = figure(x_range=group_range, y_range=list(reversed(period_label)), tools="save")
    p.width = width
    p.outline_line_color = None
    p.background_fill_color = None
    p.border_fill_color = None
    p.toolbar_location = "above"
    p.rect("group", "period", 0.9, 0.9, source=source, alpha=alpha, color="type_color")
    p.axis.visible = False
    text_props = {
        "source": source,
        "angle": 0,
        "color": "black",
        "text_align": "left",
        "text_baseline": "middle",
    }
    x = dodge("group", -0.4, range=p.x_range)
    y = dodge("period", 0.3, range=p.y_range)
    p.text(
        x=x,
        y="period",
        text="sym",
        text_font_style="bold",
        text_font_size="16pt",
        **text_props,
    )
    p.text(x=x, y=y, text="atomic_number", text_font_size="11pt", **text_props)

    color_bar = ColorBar(
        color_mapper=color_mapper,
        ticker=BasicTicker(desired_num_ticks=10),
        border_line_color=None,
        label_standoff=cbar_standoff,
        location=(0, 0),
        orientation="vertical",
        scale_alpha=alpha,
        major_label_text_font_size=f"{cbar_fontsize}pt",
    )

    if cbar_height is not None:
        color_bar.height = cbar_height

    p.add_layout(color_bar, "right")
    p.grid.grid_line_color = None

    if output_filename:
        output_file(output_filename)

    if show:
        show_(p)

    return p

# Make plot stacking SYSREM Resids and a single synthetic transmission spectra for the paper
def overlayArms(planet_name, temperature_profile, species_label, vmr, do_inject_model, do_run_all, do_make_new_model, method):
    
    drv_restricted, plotsnr_restricted, residual_restricted = {}, {}, {}
    arms = ['blue', 'red']
    fit_params, observation_epochs,_ = run_all_ccfs(planet_name, temperature_profile, species_label, vmr, do_inject_model, do_run_all, do_make_new_model, method)

    for arm in arms:
        for observation_epoch in observation_epochs:
            # This function is not ready for multiple observation epochs yet
            drv_restricted[arm]= fit_params[species_label][observation_epoch][arm]['drv_restricted']
            plotsnr_restricted[arm] = fit_params[species_label][observation_epoch][arm]['plotsnr_restricted']
            residual_restricted[arm] = fit_params[species_label][observation_epoch][arm]['residual_restricted']
            
            if arm == 'combined':
                max_index = np.argmax(plotsnr_restricted[arm])
                ax1.axvline(x=drv_restricted[arm][max_index], color='k')
                
    drv_restricted['combined']= fit_params[species_label]['combined']['combined']['drv_restricted']
    plotsnr_restricted['combined'] = fit_params[species_label]['combined']['combined']['plotsnr_restricted']
    residual_restricted['combined'] = fit_params[species_label]['combined']['combined']['residual_restricted']

    new_arms = ['blue', 'red', 'combined']
        
    #Check if drv_restricteds are the same
    if np.array_equal(drv_restricted['blue'], drv_restricted['red']) and np.array_equal(drv_restricted['red'], drv_restricted['combined']):        

        # Initialize Figure and GridSpec objects
        fig = pl.figure(figsize=(8,8))
        gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])

        # Create Axes for the main plot and the residuals plot
        ax1 = pl.subplot(gs[0])
        ax2 = pl.subplot(gs[1], sharex=ax1)

        ax1.text(0.05, 0.99, species_label, transform=ax1.transAxes, verticalalignment='top', horizontalalignment='left', fontsize=12)
        pl.setp(ax1.get_xticklabels(), visible=False)
        ax1.set_ylabel('SNR')
        # Add the horizontal line at 4 SNR
        ax1.axhline(y=4, color='g', linestyle='--', label=r'4 $\sigma$')    
        ax2.set_xlabel('$\Delta$V (km/s)')
        ax2.set_ylabel('Residuals')
        
        color_map = {
            'blue': 'b',
            'red': 'r',
            'combined': 'k'
        }
        line_style_map = {
            'blue': '--',
            'red': '--',
            'combined': '-'
        }

        for arm in new_arms:
            color = color_map[arm]
            line_style = line_style_map[arm]
            ax1.plot(drv_restricted[arm], plotsnr_restricted[arm], f'o{line_style}{color}', label='data', markersize=2)
            ax2.plot(drv_restricted[arm], residual_restricted[arm], f'o{line_style}{color}', markersize=1)
                
        # Consider a clearer naming scheme
        overlay_fits = 'plots/'+ planet_name + '.' + observation_epoch + '.' + arm + '.' + species_label + '.line-profiles-overlaidarms.pdf'
        # Save the plot
        fig.savefig(overlay_fits, dpi=300, bbox_inches='tight')
        pl.close(fig)
            
    else:
        print('The drv_restricted arrays are not the same for the different arms, there is a bug.')

#def overlayTransitPeriods(planet_name, temperature_profile, species_label, vmr, do_inject_model, do_run_all, do_make_new_model, method):

def multiSpeciesCCF(planet_name, temperature_profile, species_dict, do_inject_model, do_run_all, do_make_new_model, method):
    # This function only works with a single observation epoch! FIX THIS!

    ccf_params = {}
    if planet_name == 'KELT-20b': observation_epoch = '20190504'

    for species_label, params in species_dict.items():
        vmr = params.get('vmr')
        arm = str(params.get('arm'))

        species_name_ccf = get_species_label(species_label)
        fit_params, observation_epochs, plotsnr_restricted = run_all_ccfs(planet_name, temperature_profile, species_label, vmr, do_inject_model, do_run_all, do_make_new_model, method)
        
        if arm != 'combined':
            selected_idx = fit_params[species_label][observation_epoch][arm]['selected_idx']
            ccf_params[species_label] = {
                'amps' : fit_params[species_label][observation_epoch][arm]['amps'][selected_idx],
                'amps_error' : fit_params[species_label][observation_epoch][arm]['amps_error'][selected_idx],
                'rv' : fit_params[species_label][observation_epoch][arm]['rv'][selected_idx],
                'rv_error' : fit_params[species_label][observation_epoch][arm]['rv_error'][selected_idx],
                'width' : fit_params[species_label][observation_epoch][arm]['width'][selected_idx],
                'width_error' : fit_params[species_label][observation_epoch][arm]['width_error'][selected_idx],
            }

            # Store the results in the dictionary with species_label as the key
        elif arm == 'combined':
            selected_idx = fit_params[species_label]['combined']['combined']['selected_idx']
            ccf_params[species_label] = {
                'amps': fit_params[species_label]['combined']['combined']['amps'][selected_idx],
                'amps_error': fit_params[species_label]['combined']['combined']['amps_error'][selected_idx],
                'rv': fit_params[species_label]['combined']['combined']['rv'][selected_idx],
                'rv_error': fit_params[species_label]['combined']['combined']['rv_error'][selected_idx],
                'width': fit_params[species_label]['combined']['combined']['width'][selected_idx],
                'width_error': fit_params[species_label]['combined']['combined']['width_error'][selected_idx],
            }

    species_labels = list(ccf_params.keys())
    species_labels.sort()

    # Prepare colors based on 'amps' in ccf_params
    amps = np.array([ccf_params[species]['amps'] for species in species_labels])
    cmap = pl.cm.viridis
    colors = cmap(amps / amps.max())

    fig, ax = pl.subplots()

    # Create a normal plot with error bars for each species
    for i, species in enumerate(species_labels):
        rv = ccf_params[species]['rv']
        rv_error = ccf_params[species]['rv_error']
        ax.errorbar(rv, i, xerr=rv_error, fmt='o', color=colors[i], markersize=5, markeredgewidth=1, markeredgecolor='black', capsize=5)

    ax.set_yticks(range(len(species_labels)))
    ax.set_yticklabels(species_labels)
    ax.set_xlabel('$\Delta$V(km/s)')

    # Adding a colorbar
    sm = pl.cm.ScalarMappable(cmap=cmap, norm=pl.Normalize(vmin=amps.min(), vmax=amps.max()))
    sm.set_array([])
    cbar = pl.colorbar(sm, ax=ax)
    cbar.set_label('SNR')

    # Save the plot
    plotname = 'plots/' + planet_name + '.' + temperature_profile + '.CombinedRVs.pdf'
    fig.savefig(plotname, dpi=300, bbox_inches='tight')
    pl.close(fig)

def combinedPhaseResolvedLineProfiles(planet_name, temperature_profile, species_dict, do_inject_model, do_run_all, do_make_new_model, method, snr_coloring=True):
    """
    Calculate and plot the combined wind characteristics for different species.

    Parameters:
    planet_name (str): The name of the planet.
    temperature_profile (str): The temperature profile.
    species_dict (dict): A dictionary containing species labels and their corresponding parameters.
    do_inject_model (bool): Flag indicating whether to inject a model.
    do_run_all (bool): Flag indicating whether to run all calculations.
    do_make_new_model (bool): Flag indicating whether to make a new model.
    method (str): The method to use for calculations.
    snr_coloring (bool): If True, color points by SNR.

    Returns:
    None
    """
    
    # Initialize dicts
    line_profile = {}
    all_amps = []

    # Loop through each species
    for species_label, params in species_dict.items():
        vmr = params['vmr']
        amps, amps_error, rv, rv_error, width, width_error, selected_idx, orbital_phase, fit_params, observation_epochs, plotsnr_restricted = run_all_ccfs(planet_name, temperature_profile, species_label, vmr, do_inject_model, do_run_all, do_make_new_model, method)

        # Initialize 'combined' key for each species
        if species_label not in line_profile:
            line_profile[species_label] = {'combined': {'combined': {}}}

        # Mask 'rv' and 'rv_error' values exceeding Â±20 km/s
        rv_masked = np.where(np.abs(rv) <= 10, rv, 0)
        rv_error_masked = np.where(np.abs(rv_error) <= 10, rv_error, 0)

        # ############################################################################################################################################################################################
        # This will do for now, but later on I need to pare down the outputs of run_all_ccfs to fit_params and change how the outputs are processed in multiSpeciesCCF and combinedWindCharacteristics
        
        # Creating a whole new dictionary is unnecessary. 

        # Accessing results from run_all_ccfs and storing results for per-arm CCFs
        if species_label not in line_profile:
            line_profile[species_label] = {}

            for observation_epoch in observation_epochs:
                for arm in ['blue', 'red']:
                    if observation_epoch not in line_profile[species_label]:

                        line_profile[species_label][observation_epoch] = {}
                        line_profile[species_label][observation_epoch][arm] = {}

                        line_profile[species_label][observation_epoch][arm]['amps'] = fit_params[species_label][observation_epoch][arm]['amps']
                        line_profile[species_label][observation_epoch][arm]['amps_error'] = fit_params[species_label][observation_epoch][arm]['amps_error']
                        line_profile[species_label][observation_epoch][arm]['rv'] = fit_params[species_label][observation_epoch][arm]['rv']
                        line_profile[species_label][observation_epoch][arm]['rv_error'] = fit_params[species_label][observation_epoch][arm]['rv_error']
                        line_profile[species_label][observation_epoch][arm]['width'] = fit_params[species_label][observation_epoch][arm]['width']
                        line_profile[species_label][observation_epoch][arm]['width_error'] = fit_params[species_label][observation_epoch][arm]['width_error']
                        line_profile[species_label][observation_epoch][arm]['selected_idx'] = fit_params[species_label][observation_epoch][arm]['selected_idx']
                        line_profile[species_label][observation_epoch][arm]['orbital_phase'] = fit_params[species_label][observation_epoch][arm]['orbital_phase']
    
        # Storing results for combined CCFs
        line_profile[species_label]['combined']['combined'] = {
            'amps': amps,
            'amps_error': amps_error,
            'rv': rv_masked,
            'rv_error': rv_error_masked,
           'width': width,
             'width_error': width_error
        }

        all_amps.append(amps[selected_idx])

    # Initialize the figure
        fig, ax = pl.subplots(figsize=(8, 8))

        # x-axis for plot
        phase_min = np.min(orbital_phase)
        phase_max = np.max(orbital_phase)
        phase_array = np.linspace(phase_min, phase_max, np.shape(rv)[0])

    if snr_coloring:
        # Normalize the amplitude values to get a continuous range of colors
        norm = colors.Normalize(vmin=min(all_amps), vmax=max(all_amps))
        scalar_map = cm.ScalarMappable(norm=norm, cmap=cm.magma)  # Using the 'plasma' colormap

        # Plotting for each species
        for species, species_data in line_profile.items():
            combined_data = species_data['combined']['combined']
            
            # Generate a color for each data point based on its amplitude
            colors_array = [scalar_map.to_rgba(amp) for amp in combined_data['amps']]

            # Iterate over each point, applying a color based on its amplitude
            for i in range(len(phase_array)):
                color_val = colors_array[i]
                
                # Plotting the point with the same color for fill and edge
                ax.plot(phase_array[i], combined_data['rv'][i], 'o', color=color_val, markeredgecolor=color_val, markersize=5)

            # Creating a gradient fill
            for i in range(len(phase_array) - 1):
                ax.fill_between([combined_data['rv'][i] - combined_data['rv_error'][i], combined_data['rv'][i] + combined_data['rv_error'][i]], phase_array[i], phase_array[i + 1], color=colors_array[i], alpha=0.5)

            # Labeling the species at the beginning of the curve with offset and style
            start_phase = phase_array[0] - 0.05 * (phase_max - phase_min)  # Adjusting horizontal offset
            start_center = combined_data['rv'][0]
            ax.text(start_phase, start_center, species, color='red', fontsize=12, 
                    path_effects=[patheffects.withStroke(linewidth=3, foreground='black')],
                    ha='right', va='center')

        # Create a color bar
        cbar = pl.colorbar(scalar_map, ax=ax)
        cbar.set_label('SNR')

    else:
        # Set up a color cycle using a matplotlib colormap
        num_species = len(line_profile)
        color_map = pl.cm.get_cmap('nipy_spectral', num_species)  # Choosing a colormap with sufficient distinct colors
        
        species_colors = {species: color_map(i) for i, species in enumerate(line_profile.keys())}
        
        # Plotting for each species with assigned color
        for species, species_data in enumerate(line_profile.items()):
            # Access the 'combined' data
            combined_data = species_data['combined']['combined']
            species_color = species_colors[species]

            # Use the assigned color for plotting
            ax.plot(phase_array, combined_data['rv'], 'o', label=f'{species} (Combined)', color=species_color)
            ax.fill_between(phase_array, combined_data['rv'] - combined_data['rv_error'], combined_data['rv'] + combined_data['rv_error'], color=species_color, alpha=0.2)

            # Plotting for individual observation epochs and arms
            for observation_epoch, epoch_data in species_data.items():
                if observation_epoch != 'combined':
                    for arm, arm_data in epoch_data.items():
                        ax.plot(phase_array, arm_data['rv'], 'o', label=f'{species} ({observation_epoch} - {arm})', color=species_color)
                        ax.fill_between(phase_array, arm_data['rv'] - arm_data['rv_error'], arm_data['rv'] + arm_data['rv_error'], color=species_color, alpha=0.2)

        # Create custom lines for the legend to match the species' colors
        custom_lines = [pl.Line2D([0], [0], color=species_colors[species], lw=4) for species in line_profile.keys()]

        # Create the legend
        ax.legend(custom_lines, [f'{species}' for species in line_profile.keys()])

    # Save the plot
    plotname_combined = 'plots/' + planet_name + '.' + temperature_profile + '.CombinedWindCharacteristics_Combined.pdf'   
    fig.savefig(plotname_combined, dpi=300, bbox_inches='tight')
    pl.close(fig)

    # Save the plot for each arm
    for arm in ['blue', 'red']:
        plotname_arm = 'plots/' + planet_name + '.' + temperature_profile + f'.CombinedWindCharacteristics_{arm}.pdf'
        fig.savefig(plotname_arm, dpi=300, bbox_inches='tight')
        pl.close(fig)

#def transitAsymmetries(planet_name, temperature_profile, species_label, vmr, do_inject_model, do_run_all, do_make_new_model, method):

#Make plot stacking all of the synthetic transmission spectra for appendix
def make_spectrum_plots(species_dict):
    
    instrument = 'PEPSI'
    planet_name = 'KELT-20b'
    temperature_profile = 'inverted-transmission-better'
    spectrum_type = 'transmission'

    parameters = {}
    parameters['Hm'] = 1e-9
    parameters['em'] = 0.0008355 #constant for now, link to FastChem later
    parameters['H0'] = 2.2073098e-12 #ditto
    parameters['P0'] = 1.0
    
    
    if instrument == 'PEPSI':
        if (planet_name == 'WASP-189b' or planet_name == 'KELT-20b'):
            instrument_here = 'PEPSI-25'
        else:
            instrument_here = 'PEPSI-35'
    else:
        instrument_here = instrument
          

    
    if planet_name == 'KELT-20b':
            parameters['Teq'] = 3000.
            
            if temperature_profile == 'inverted-emission-better' or temperature_profile == 'inverted-transmission-better':
                parameters['kappa'] = 0.04
                parameters['gamma'] = 30.
                ptprofile = 'guillot'
        
        
    n_spectra = len(species_dict.keys())  # Number of spectra to plot
    
    fig, axs = pl.subplots(n_spectra, sharex=True, figsize=(6, n_spectra*2))  # Create subplots based on the number of spectra 
    fig.subplots_adjust(hspace=0.125)  
    for i, (species, params) in enumerate(species_dict.items()):
        species_name_inject, species_name = get_species_keys(species)
        vmr = params.get('vmr')        
        template_wave, template_flux = make_new_model(instrument, species_name, vmr, spectrum_type, planet_name, temperature_profile)

        #if planet_name == 'WASP-189b' or planet_name == 'KELT-20b':
        #    axs[i].fill([4265,4265,4800,4800],[np.nanmin(template_flux),np.nanmax(template_flux),np.nanmax(template_flux),np.nanmin(template_flux)],color='cyan',alpha=0.25)
        if planet_name != 'WASP-189b':
            axs[i].fill([4800,4800,5441,5441],[np.nanmin(template_flux),np.nanmax(template_flux),np.nanmax(template_flux),np.nanmin(template_flux)],color='blue',alpha=0.25)
            
        axs[i].text(0.125, 0.85, species_name, transform=axs[i].transAxes)
        axs[i].plot(template_wave, template_flux, color='black')

        axs[i].fill([6278,6278,7419,7419],[np.nanmin(template_flux),np.nanmax(template_flux),np.nanmax(template_flux),np.nanmin(template_flux)],color='red',alpha=0.25)

        if i == n_spectra - 1:  # If this is the last subplot
            axs[i].tick_params(axis='x', which='both', labeltop=False, labelbottom=True)  # Only show x-axis label on the bottom
            axs[i].set_xlabel('Wavelength (Ã…)')
        else:
            axs[i].tick_params(axis='x', which='both', labeltop=False, labelbottom=False)  # Don't show x-axis label on other subplots
        axs[i].set_ylabel('normalized flux')

        plotout = 'plots/spectra.' + planet_name +  '.' + temperature_profile + '.pdf'
        pl.savefig(plotout,format='pdf')

# Make plot stacking PT profiles for each species

def fastchem_plot(abundance_species):


    #Do the chemistry calculations
    #this loads the temperatures and pressures produced by petitRADTRANS, you may need to modify these lines if you store these data products somewhere else
    temperatures = np.load('data_products/radtrans_temperature.npy')
    pressures = np.load('data_products/radtrans_pressure.npy')

    fastchem = pyfastchem.FastChem('/home/calder/Documents/FastChem/input/element_abundances/asplund_2020_extended.dat', 
                                '/home/calder/Documents/FastChem/input/logK/logK.dat', 
                                1)
    
    input_data = pyfastchem.FastChemInput()
    output_data = pyfastchem.FastChemOutput()

    input_data.temperature = temperatures
    input_data.pressure = pressures

    fastchem_flag = fastchem.calcDensities(input_data, output_data)

    number_densities = np.array(output_data.number_densities)
    gas_number_density = pressures*1e6 / (const.k_B.cgs * temperatures)

    #set the quench pressure to 1 bar
    quench = np.argmin(np.abs(pressures-1e1))

    a_index = []
    abundance_species_indices, abundance_species_masses_ordered = [], []
    n_species = fastchem.getElementNumber()



    if np.amin(output_data.element_conserved[:]) == 1:
        print("  - element conservation: ok")
    else:
        print("  - element conservation: fail")


    #save the monitor output to a file

    line_styles = ['-', '--', '-.', ':']

    for i, species in enumerate(abundance_species):
        index = fastchem.getGasSpeciesIndex(species)
        if index != pyfastchem.FASTCHEM_UNKNOWN_SPECIES:
            abundance_species_indices.append(index) 
            this_species = number_densities[quench, index]/gas_number_density[quench]
            # Plot the species with different line styles and add a label
            pl.plot(number_densities[:, index]/gas_number_density[:],pressures, linestyle=line_styles[i % len(line_styles)], label=species)
        else:
            print("Species", species, "to plot not found in FastChem")

    pl.xscale('log')
    pl.yscale('log')

    # label the axes
    pl.xlabel('VMR')
    pl.ylabel('Pressure (bar)')

    # add ticks to the plot
    pl.tick_params(axis='both', which='both', direction='in', top=True, right=True)

    # add a legend to the plot without bounding box and small, and dont make it transparent
    pl.legend(loc='lower left', fontsize='small', frameon=False, facecolor='white', edgecolor='black')   


    # increase the size of the plot
    pl.gcf().set_size_inches(6, 6)

    # set y limit from 10^-12 to 10^0
    pl.ylim(1e-8, 1e0)
    pl.gca().invert_yaxis()

    pl.savefig('plots/'+'PT-plots.pdf')  # Save the plot as a PDF

def calculate_observability_score(instrument_here, opacities_all, opacities_without_species, wavelengths):

    observability_scores = {}

    # Define the wavelength range in Ã…
    lambda_low, lambda_high = get_wavelength_range(instrument_here)
    
     # Mask to select the wavelength range
    mask = (wavelengths >= lambda_low) & (wavelengths <= lambda_high)
    
    # Calculate the total opacity with all species included
    tau_all = np.log(opacities_all[mask])

    for species, opacities in opacities_without_species.items():
        # Calculate the opacity without the current species
        tau_without_species = np.log(opacities[mask])
        
        # Calculate the observability score using the provided formula
        score = simps(tau_all - tau_without_species, wavelengths[mask])
        observability_scores[species] = score
    
    # Normalize the scores so the most observable species is 1
    max_score = max(observability_scores.values())
    for species in observability_scores:
        observability_scores[species] /= max_score
    return observability_scores

def create_atmospheres(planet_name, temperature_profile, instrument, species_dict, ptprofile):

    #will need to generalize the below to any planet that is not KELT-20!
    parameters = {}
    vmrs = []
    mass = []
    abundances = {}
    species_labels = []

    
    parameters['Hm'] = 1e-9
    parameters['em'] = 0.0008355 #constant for now, link to FastChem later
    parameters['H0'] = 2.2073098e-12 #constant for now, link to FastChem later
    parameters['P0'] = 1.0
    if planet_name == 'KELT-20b':
        parameters['Teq'] = 2262.
        if temperature_profile == 'inverted-emission-better' or temperature_profile == 'inverted-transmission-better':
            parameters['kappa'] = 0.04
            parameters['gamma'] = 30.
            ptprofile = 'guillot'
            
    for species in species_dict.keys():
        species_name_inject, species_name_ccf = get_species_keys(species)
        species_labels.append(species_name_ccf)
        parameters[species_name_ccf] = species_dict[species]['vmr']
        mass.append(get_species_mass(species_name_ccf))
        vmrs.append(parameters[species_name_ccf])

    MMW = 2.33

    mass = np.array(mass)
    vmrs = np.array(vmrs)
    species_abundance = mass/MMW * vmrs

    H2_abundance = 1.008
    He_abundance = 4.00126 * (10.**(10.925-12))
    Hm_abundance = (1.00784/MMW) * parameters['Hm']
    e_abundance = ((1./1822.8884845)/MMW) * parameters['em']
    H_abundance = (1.00784/MMW) * parameters['H0']

    total_abundance = np.sum(species_abundance) + H2_abundance + He_abundance + e_abundance + H_abundance + Hm_abundance

    i=0
    for species in species_labels:
        abundances[species] = species_abundance[i]/total_abundance
        i+=1

    abundances['H2'] = H2_abundance/total_abundance
    abundances['He'] = He_abundance/total_abundance
    abundances['H'] = H_abundance/total_abundance
    abundances['H-'] = Hm_abundance/total_abundance
    abundances['e-'] = e_abundance/total_abundance

    if instrument == 'PEPSI':
        #if (planet_name == 'WASP-189b' or planet_name == 'KELT-20b'):
        if planet_name == 'WASP-189b':
            instrument_here = 'PEPSI-25'
        else:
            instrument_here = 'PEPSI-35'
    else:
        instrument_here = instrument

    R_host, R_pl, M_pl, Teff, gravity = get_planetary_parameters(planet_name)
    lambda_low, lambda_high = get_wavelength_range(instrument_here)

    pressures = np.logspace(-8, 2, 100)
    atmosphere = instantiate_radtrans(species_labels, lambda_low, lambda_high, pressures, downsample_factor=5)
    
    kappa_IR = parameters['kappa']
    gamma = parameters['gamma']
    T_equ = parameters['Teq']
    T_int = 100.
    P0 = parameters['P0']
    
    if ptprofile == 'guillot':
        temperature = guillot_global(pressures, kappa_IR, gamma, gravity, T_int, T_equ)
    
    atmosphere.setup_opa_structure(pressures)

    for key in abundances:
        if isinstance(abundances[key], np.ndarray):
            abundances[key] = abundances[key][0]
        abundances[key] *= np.ones_like(temperature)
        

    MMW = 2.33 * np.ones_like(temperature)

    
    atmosphere.calc_transm(temperature, abundances, gravity, MMW, R_pl=R_pl, P0_bar=P0)
    opacities_all = atmosphere.transm_rad

    wavelengths = nc.c / atmosphere.freq / 1e-8  # Convert frequency to wavelength in Ã…

    opacities_without_species = {}
    
    for species in species_labels:
        abundances_temp = abundances.copy()
        abundances_temp[str(species)] = 0.0
        atmosphere.calc_transm(temperature, abundances_temp, gravity, MMW, R_pl=R_pl, P0_bar=P0)
        opacities_without_species[str(species)] = atmosphere.transm_rad
            
    return opacities_all, opacities_without_species, wavelengths, instrument_here

def generate_observability_table(planet_name, temperature_profile, instrument, species_dict, ptprofile):
    
    opacities_all, opacities_without_species, wavelengths, instrument_here = create_atmospheres(planet_name,temperature_profile, instrument, species_dict, ptprofile)
    observability_scores = calculate_observability_score(instrument_here, opacities_all, opacities_without_species, wavelengths)
    
    # Save observability scores and their corresponding species to a CSV file
    filename = 'observability_scores.csv'
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        for species, score in observability_scores.items():
            writer.writerow([species, score])
    p = plotter(filename ,cmap='magma',extended=False, log_scale=False)
    export_png(p, filename='plots/observability_scores.pdf', webdriver=webdriver.Chrome())