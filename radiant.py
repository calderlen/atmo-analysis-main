import numpy as np
import petitRADTRANS.nat_cst as nc
from petitRADTRANS import Radtrans
from astropy import units as u
from astropy.io import ascii

from create_model import create_model, instantiate_radtrans
from atmo_utilities import *
from run_all_ccfs import *

import emcee
import argparse
import os
import sys


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
    # Add AMUs here for new species
    
    if 'TiO' in species_name: return 47.867+15.999
    if 'Ti' in species_name: return 47.867
    if 'VO' in species_name: return 50.9415+15.999
    if 'FeH' in species_name: return 55.845+1.00784
    if 'Fe' or 'Fe+' in species_name: return 55.845
    if 'Ni' in species_name: return 58.6934
    if 'SiO' in species_name: return 28.085+15.999
    if 'Si' in species_name: return 28.085
    if 'OH' in species_name: return 15.999+1.00784
    if 'H2O' in species_name: return 15.999+2.*1.00784
    if 'H2S' in species_name: return 32.06+2.*1.00784
    if 'Cr' in species_name: return 51.9961
    if 'CaH' in species_name: return 40.078+1.00784
    if 'CO' in species_name: return 12.011+15.999
    if 'MgH' in species_name: return 24.305+1.00784
    if 'Mg' in species_name: return 24.305
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
        R_host = 1.6174400*nc.r_sun # NEA: Lund et al. 2017, (+0.057, -0.064)
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

    np.save('data_products/radtrans_temperature.npy', temperature)
    np.save('data_products/radtrans_pressure.npy', pressures)
    
    
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
