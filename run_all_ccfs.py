#import packages
import numpy as np
import matplotlib.pyplot as pl
import matplotlib.gridspec as gridspec
from matplotlib import cm, colors
from matplotlib import patheffects

from glob import glob
from astropy.io import fits

from scipy.stats import chisquare
from scipy.optimize import curve_fit

import os.path

from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
from astropy.table import Table

from uncertainties import ufloat
from uncertainties import unumpy as unp

from atmo_utilities import ccf, one_log_likelihood, log_likelihood_CCF, vacuum2air, log_likelihood_opt_beta

import time

from dtutils import psarr

from radiant import *
from create_model import create_model, instantiate_radtrans
import horus

# global varaibles defined for harcoded path to data on my computer
path_modifier_plots = '/home/calder/Documents/atmo-analysis-main/'  #linux
path_modifier_data = '/home/calder/Documents/petitRADTRANS_data/'   #linux
path_modifier_plots = '/Users/calder/Documents/atmo-analysis-main/' #mac
path_modifier_data = '/Users/calder/Documents/petitRADTRANS_data/'  #mac


def run_one_ccf(species_label, vmr, arm, observation_epoch, template_wave, template_flux, template_wave_in, template_flux_in, planet_name, temperature_profile, do_inject_model, species_name_ccf, model_tag, f, method, do_make_new_model):
    
    niter = 10
    n_systematics = np.array(get_sysrem_parameters(arm, observation_epoch, species_label, planet_name))
    ckms = 2.9979e5

    if arm == 'red':
        do_molecfit = True
    else:
        do_molecfit = False

    Period, epoch, M_star, RV_abs, i, M_p, R_p, RA, Dec, Kp_expected, half_duration_phase, Ks_expected = get_planet_parameters(planet_name)

    wave, fluxin, errorin, jd, snr_spectra, exptime, airmass, n_spectra, npix = get_pepsi_data(arm, observation_epoch, planet_name, do_molecfit)

    orbital_phase = get_orbital_phase(jd, epoch, Period, RA, Dec)

    if planet_name == 'TOI-1431b': wave = correct_for_reflex_motion(Ks_expected, orbital_phase, wave, n_spectra)

    if do_inject_model:
        fluxin, Kp_true, V_sys_true = inject_model(Kp_expected, orbital_phase, wave, fluxin, template_wave_in, template_flux_in, n_spectra)
    else:
        Kp_true, V_sys_true = Kp_expected.n, RV_abs

    wave, flux, ccf_weights = regrid_data(wave, fluxin, errorin, n_spectra, template_wave, template_flux, snr_spectra, temperature_profile, do_make_new_model)

    #residual_flux = flux[:]
    residual_flux = flatten_spectra(flux, npix, n_spectra)

    #Make some diagnostic plots
    plotname = 'plots/' + planet_name + '.' + observation_epoch + '.' + arm + '.spectrum-resids.pdf'

    #import pdb; pdb.set_trace()

    psarr(unp.nominal_values(residual_flux), wave, orbital_phase, 'wavelength (Angstroms)', 'orbital phase', 'flux residual', filename=plotname,flat=True, ctable='gist_gray')

    sysrem_file = path_modifier_plots + 'data_products/' + planet_name + '.' + observation_epoch + '.' + arm + '.SYSREM-' + str(n_systematics[0]) + '+' + str(n_systematics[1])+model_tag+'.npy'

    #if os.path.isfile(sysrem_file):
    #    do_sysrem = True
    #else:
    #    do_sysrem = False

    
    if not os.path.isfile(sysrem_file+'.Umatrix.npy'):
        
        corrected_flux, corrected_error, U_sysrem, telluric_free = do_sysrem(wave, residual_flux, arm, airmass, n_spectra, niter, n_systematics, do_molecfit)
        #corrected_flux, corrected_error = unp.nominal_values(residual_flux), unp.std_devs(residual_flux)

        np.save(sysrem_file, corrected_flux)
        np.save(sysrem_file+'.corrected-error.npy', corrected_error)
        np.save(sysrem_file+'.Umatrix.npy', U_sysrem)
        np.save(sysrem_file+'.telluric-region.npy', telluric_free)

        plotname = path_modifier_plots + 'plots/' + planet_name + '.' + observation_epoch + '.' + arm + '.spectrum-SYSREM-'+ str(n_systematics[0]) + '+' + str(n_systematics[1])+'.pdf'

        psarr(corrected_flux, wave, orbital_phase, 'wavelength (Angstroms)', 'orbital phase', 'flux residual', filename=plotname,flat=True, ctable='gist_gray')

    else:
        corrected_flux = np.load(sysrem_file)
        corrected_error = np.load(sysrem_file+'.corrected-error.npy')
        U_sysrem = np.load(sysrem_file+'.Umatrix.npy')
        telluric_free = np.load(sysrem_file+'.telluric-region.npy')

    if method == 'ccf':
        ccf_file = path_modifier_plots + 'data_products/'+ planet_name + '.' + observation_epoch + '.' + arm + '.' + species_name_ccf + model_tag + '.CCFs-raw.npy'

        drv, cross_cor, sigma_cross_cor = get_ccfs(wave, corrected_flux, corrected_error, template_wave, template_flux, n_spectra, U_sysrem, telluric_free)

        
        
        np.save(ccf_file, cross_cor)
        np.save(ccf_file+'.sigma.npy', sigma_cross_cor)
        np.save(ccf_file+'.phase.npy', orbital_phase)
        np.save(ccf_file+'.ccf_weights', ccf_weights)

   
        #Normalize the CCFs
        for i in range (n_spectra): 
            cross_cor[i,:]-=np.mean(cross_cor[i,:])
            sigma_cross_cor[i,:] = np.sqrt(sigma_cross_cor[i,:]**2 + np.sum(sigma_cross_cor[i,:]**2)/len(sigma_cross_cor[i,:])**2)
            #I guess the following is OK as long as there isn't a strong peak, which there shouldn't be in any of the individual CCFs
            cross_cor[i,:]/=np.std(cross_cor[i,:])

        # Specifically for KELT-20b
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
        plotname = path_modifier_plots + 'plots/' + planet_name + '.' + observation_epoch + '.' + species_name_ccf + model_tag + '.' + arm + '.CCFs-raw.pdf'
        
        psarr(cross_cor, drv, orbital_phase, 'v (km/s)', 'orbital phase', 'SNR', filename=plotname, ctable='gist_gray', carr = Kp_true * np.sin(2.*np.pi*orbital_phase))

        #blank out the non-radial pulsations for now
        if planet_name == 'WASP-33b' or planet_name == 'TOI-1431b':
            if planet_name == 'WASP-33b': badwidth = 100.
            if planet_name == 'TOI-1431b': badwidth = 10.
            bads = np.abs(drv) <= badwidth
            cross_cor[:, bads] = 0.0
            sigma_cross_cor[:, bads] = 1e5

            plotname = 'plots/' + planet_name + '.' + observation_epoch + '.' + species_name_ccf + model_tag + '.' + arm + '.blanked.CCFs-raw.pdf'

            psarr(cross_cor, drv, orbital_phase, 'v (km/s)', 'orbital phase', 'SNR', filename=plotname, ctable='gist_gray', carr = Kp_true * np.sin(2.*np.pi*orbital_phase))

            np.save(ccf_file, cross_cor)
            np.save(ccf_file+'.sigma.npy', sigma_cross_cor)


    
        snr, Kp, drv, cross_cor_display = combine_ccfs(drv, cross_cor, sigma_cross_cor, orbital_phase, n_spectra, ccf_weights, half_duration_phase, temperature_profile)

        
        make_shifted_plot(snr, planet_name, observation_epoch, arm, species_name_ccf, model_tag, RV_abs, Kp_expected, V_sys_true, Kp_true, do_inject_model, True, drv, Kp, species_label, temperature_profile, method)

        get_peak_snr(snr, drv, Kp, do_inject_model, V_sys_true, Kp_true, RV_abs, Kp_expected, arm, observation_epoch, f, method)


    if 'likelihood' in method:
        like_file = path_modifier_plots + 'data_products/'+ planet_name + '.' + observation_epoch + '.' + arm + '.' + species_name_ccf + model_tag + '.likelihood-raw.npy'
        drv, lnL = get_likelihood(wave, corrected_flux, corrected_error, template_wave, template_flux, n_spectra, U_sysrem, telluric_free)

        np.save(like_file, lnL)
        np.save(like_file+'.phase.npy', orbital_phase)

        #Make a plot
        plotname = path_modifier_plots + 'plots/' + planet_name + '.' + observation_epoch + '.' + species_name_ccf + model_tag + '.likelihoods-raw.pdf'
        psarr(lnL, drv, orbital_phase, 'v (km/s)', 'orbital phase', 'ln L', filename=plotname, ctable='inferno')

        #now need to combine the likelihoods along the planet orbit
        shifted_lnL, Kp, drv = combine_likelihoods(drv, lnL, orbital_phase, n_spectra, half_duration_phase, temperature_profile)

        make_shifted_plot(shifted_lnL, planet_name, observation_epoch, arm, species_name_ccf, model_tag, RV_abs, Kp_expected, V_sys_true, Kp_true, do_inject_model, True, drv, Kp, species_label, temperature_profile, method)

        combine_ccfs_binned(drv, cross_cor, sigma_cross_cor, orbital_phase, n_spectra, ccf_weights, half_duration_phase, temperature_profile, Kp_true, species_name_ccf, planet_name)
    return orbital_phase

def combine_observations(observation_epochs, arms, planet_name, temperature_profile, species_label, species_name_ccf, model_tag, RV_abs, Kp_expected, do_inject_model, f, method):

    Period, epoch, M_star, RV_abs, i, M_p, R_p, RA, Dec, Kp_expected, half_duration_phase, Ks_expected = get_planet_parameters(planet_name)

    j=0
    for observation_epoch in observation_epochs:
        for arm in arms:
            if 'likelihood' in method:
                ccf_file_2 = path_modifier_plots + 'data_products/'+ planet_name + '.' + observation_epoch + '.' + arm + '.' + species_name_ccf + model_tag + '.likelihood-raw.npy'
            if method == 'ccf':
                ccf_file_2 = path_modifier_plots + 'data_products/'+ planet_name + '.' + observation_epoch + '.' + arm + '.' + species_name_ccf + model_tag + '.CCFs-raw.npy'
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

        drv_original = drv[:]
        
        snr, Kp, drv, cross_cor_display = combine_ccfs(drv, cross_cor, sigma_cross_cor, orbital_phase, len(orbital_phase), ccf_weights, half_duration_phase, temperature_profile)

        ind = np.unravel_index(np.argmax(snr, axis=None), snr.shape)
        Kp_best, drv_best = Kp[ind[0]], drv[ind[1]]

        if planet_name == 'KELT-20b': ccfs_binned = combine_ccfs_binned(drv_original, cross_cor, sigma_cross_cor, orbital_phase, len(orbital_phase), ccf_weights, half_duration_phase, temperature_profile, Kp_best, species_name_ccf, planet_name)

        #Make a plot
        if len(arms) > 1:
            which_arms = 'combined'
        else:
            which_arms = arms[0]
        plotname = 'plots/' + planet_name + '.' + which_arms + '.' + species_name_ccf + model_tag + '.' + arm + '.CCFs-raw.pdf'

        phase_order = np.argsort(orbital_phase)
        
        psarr(cross_cor_display[phase_order,:], drv, orbital_phase[phase_order], 'v (km/s)', 'orbital phase', 'SNR', filename=plotname, ctable='gist_gray', carr = Kp_true * np.sin(2.*np.pi*orbital_phase[phase_order]))

        
        
    if 'likelihood' in method:
        snr, Kp, drv = combine_likelihoods(drv, cross_cor, orbital_phase, len(orbital_phase), half_duration_phase, temperature_profile)


    all_epochs = observation_epochs[0]
    if len(observation_epochs) > 1:
        for i in range (1, len(observation_epochs)):
            all_epochs += '+'+observation_epochs[i]

    make_shifted_plot(snr, planet_name, all_epochs, which_arms, species_name_ccf, model_tag, RV_abs, Kp_expected, V_sys_true, Kp_true, do_inject_model, True, drv, Kp, species_label, temperature_profile, method)

    

    #import pdb; pdb.set_trace()

    get_peak_snr(snr, drv, Kp, do_inject_model, V_sys_true, Kp_true, RV_abs, Kp_expected, which_arms, all_epochs, f, method)
    
    return Kp_true, orbital_phase
            

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
        if planet_name == 'KELT-9b': observation_epochs = ['20180703', '20190622']
        if planet_name == 'MASCARA-1b': observation_epochs = ['20220925']
    else:
        spectrum_type = 'emission'
        if planet_name == 'KELT-20b': observation_epochs = ['20210501', '20210518']
        if planet_name == 'WASP-12b': observation_epochs = ['20210303', '20220208']
        if planet_name == 'KELT-9b': observation_epochs = ['20210628']
        if planet_name == 'WASP-76b': observation_epochs = ['20211031']
        if planet_name == 'WASP-33b': observation_epochs = ['20220929', '20221202']
        if planet_name == 'WASP-189b': observation_epochs = ['20230327']
        if planet_name == 'TOI-1431b': observation_epochs = ['20231023']
        if planet_name == 'TOI-1518b': observation_epochs = ['20231106'] #, '20240114']
        


    if species_label == 'FeH' or species_label == 'CrH':
        arms = ['red']
    elif species_label == 'CaH':
        arms = ['blue']
    else:
        arms = ['blue','red']
            
    species_name_inject, species_name_ccf = get_species_keys(species_label)

    file_out = path_modifier_plots + 'logs/'+ planet_name + '.' + species_name_ccf + model_tag + '.log'
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

    # Ensure species_label key is initialized in fit_params
    if species_label not in fit_params:
        fit_params[species_label] = {}

    if do_run_all:
        for observation_epoch in observation_epochs:
            for arm in arms:
                print('Now running the ',arm,' data for ',observation_epoch)
                orbital_phase = run_one_ccf(species_label, vmr, arm, observation_epoch, template_wave, template_flux, template_wave_in, template_flux_in, planet_name, temperature_profile, do_inject_model, species_name_ccf, model_tag, f, method, do_make_new_model)

    print('Now combining all of the data')

    Period, epoch, M_star, RV_abs, i, M_p, R_p, RA, Dec, Kp_expected, half_duration_phase, Ks_expected = get_planet_parameters(planet_name)
    
    # ############################################################################################################################################################################################
    # This will do for now, but later on I need to pare down the outputs of run_all_ccfs to fit_params and change how the outputs are processes in multiSpeciesCCF and combinedWindCharacteristics
    Kp_true, orbital_phase = combine_observations(observation_epochs, arms, planet_name, temperature_profile, species_label, species_name_ccf, model_tag, RV_abs, Kp_expected, do_inject_model, f, method)

    if species_label != 'FeH': combine_observations(observation_epochs, ['blue'], planet_name, temperature_profile, species_label, species_name_ccf, model_tag, RV_abs, Kp_expected, do_inject_model, f, method)

    if species_label != 'CaH': combine_observations(observation_epochs, ['red'], planet_name, temperature_profile, species_label, species_name_ccf, model_tag, RV_abs, Kp_expected, do_inject_model, f, method)

    f.close()
    orbital_phase, observation_epochs

def multiSpeciesCCF(planet_name, temperature_profile, species_dict, do_inject_model, do_run_all, do_make_new_model, method):
    """
    Runs all cross-correlation functions (CCFs) for a given planet, temperature profile, and all species labels with their
    respective VMRs and p0_gaussian values given in the species_dict.

    Args:
        planet_name (str): The name of the planet.
        temperature_profile (str): The temperature profile.
        species_dict (dict): A dictionary where each key is a species label and its value is another dictionary
                             containing 'vmr' and 'p0_gaussian' for the species.
        do_inject_model (bool): Flag indicating whether to inject a model.
        do_run_all (bool): Flag indicating whether to run all CCFs.
        do_make_new_model (bool): Flag indicating whether to make a new model.
        method (str): The method to use for the CCF.

    Returns:
        None
    """

    ccf_arrays = {}
    ccf_params = {}

    for species_label, params in species_dict.items():
        vmr = params.get('vmr')
        #p0_gaussian = params.get('p0_gaussian')
        amps, amps_error, rv, rv_error, width, width_error, selected_idx, orbital_phase, fit_params, observation_epochs = run_all_ccfs(
            planet_name, temperature_profile, species_label, vmr, do_inject_model, do_run_all, do_make_new_model, method)
        
        # Store the results in the dictionary with species_label as the key
        ccf_arrays[species_label] = {
            'amps': amps,
            'amps_error': amps_error,
            'rv': rv,
            'rv_error': rv_error,
            'width': width,
            'width_error': width_error
        }

        ccf_params[species_label] = {
            'amps': amps[selected_idx],
            'amps_error': amps_error[selected_idx],
            'rv': rv[selected_idx],
            'rv_error': rv_error[selected_idx],
            'width': width[selected_idx],
            'width_error': width_error[selected_idx]
        }

    species_labels = list(ccf_arrays.keys())

    # Extracting 'rv' for each species
    boxplot_data = [ccf_arrays[species]['rv'] for species in species_labels]

    # Prepare colors based on 'amps' in ccf_params
    amps = [ccf_params[species]['amps'] for species in species_labels]
    norm = pl.Normalize(min(amps), max(amps))
    cmap = pl.cm.viridis

    fig, ax = pl.subplots()

    # Create a horizontal boxplot for each species
    bp = ax.boxplot(boxplot_data,
                    vert=False,
                    patch_artist=True)
    
    # Apply color to each box
    for patch, color in zip(bp['boxes'], cmap(norm(amps))):
        patch.set_facecolor(color)

    ax.set_yticklabels(species_labels)
    ax.set_xlabel('$v_{sys}$ (km/s)')      #Vsys
    ax.set_ylabel('Species')
    ax.set_title('Variation in $V_{sys}$ for each species across all observations by SNR')
    
    # Adding a colorbar
    sm = pl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = pl.colorbar(sm, ax=ax)
    cbar.set_label('SNR')

    # Save the plot
    plotname = path_modifier_plots + 'plots/' + planet_name + '.' + temperature_profile + '.CombinedSpeciesSNRs.pdf'
    fig.savefig(plotname, dpi=300, bbox_inches='tight')

# example usage
# multiSpeciesCCF('KELT-20b', 'inverted-transmission-better', {'Fe I' : 5.39e-05, 'Fe II' : 5.39e-05, 'Ni I' :  2.676e-06, 'V I' :  5.623e-09, 'Ca I' :  2.101e-08, 'Co I' : 1.669e-07, 'Mn I' : 2.350e-07, 'Na I' : 2.937e-06, 'H I' : 2.646e-04}, False, True, True, 'ccf')

def combinedWindCharacteristics(planet_name, temperature_profile, species_dict, do_inject_model, do_run_all, do_make_new_model, method, snr_coloring=True):
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
    wind_chars = {}
    all_amps = []

    # Loop through each species
    for species_label, params in species_dict.items():
        vmr = params['vmr']
        #p0_gaussian = params['p0_gaussian']
        amps, amps_error, rv, rv_error, width, width_error, selected_idx, orbital_phase, fit_params, observation_epochs = run_all_ccfs(planet_name, temperature_profile, species_label, vmr, do_inject_model, do_run_all, do_make_new_model, method)

        # Initialize 'combined' key for each species
        if species_label not in wind_chars:
            wind_chars[species_label] = {'combined': {'combined': {}}}

        # Mask 'rv' and 'rv_error' values exceeding ±20 km/s
        rv_masked = np.where(np.abs(rv) <= 20, rv, 0)
        rv_error_masked = np.where(np.abs(rv_error) <= 20, rv_error, 0)

        # ############################################################################################################################################################################################
        # This will do for now, but later on I need to pare down the outputs of run_all_ccfs to fit_params and change how the outputs are processed in multiSpeciesCCF and combinedWindCharacteristics
        
        # Creating a whole new dictionary is unnecessary. 

        # Accessing results from run_all_ccfs and storing results for per-arm CCFs
        if species_label not in wind_chars:
            wind_chars[species_label] = {}

            for observation_epoch in observation_epochs:
                for arm in ['blue', 'red']:
                    if observation_epoch not in wind_chars[species_label]:

                        wind_chars[species_label][observation_epoch] = {}
                        wind_chars[species_label][observation_epoch][arm] = {}

                        wind_chars[species_label][observation_epoch][arm]['amps'] = fit_params[species_label][observation_epoch][arm]['amps']
                        wind_chars[species_label][observation_epoch][arm]['amps_error'] = fit_params[species_label][observation_epoch][arm]['amps_error']
                        wind_chars[species_label][observation_epoch][arm]['rv'] = fit_params[species_label][observation_epoch][arm]['rv']
                        wind_chars[species_label][observation_epoch][arm]['rv_error'] = fit_params[species_label][observation_epoch][arm]['rv_error']
                        wind_chars[species_label][observation_epoch][arm]['width'] = fit_params[species_label][observation_epoch][arm]['width']
                        wind_chars[species_label][observation_epoch][arm]['width_error'] = fit_params[species_label][observation_epoch][arm]['width_error']
                        wind_chars[species_label][observation_epoch][arm]['selected_idx'] = fit_params[species_label][observation_epoch][arm]['selected_idx']
                        wind_chars[species_label][observation_epoch][arm]['orbital_phase'] = fit_params[species_label][observation_epoch][arm]['orbital_phase']
    
        # Storing results for combined CCFs
        wind_chars[species_label]['combined']['combined'] = {
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
        for species, species_data in wind_chars.items():
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
                ax.fill_betweenx([combined_data['rv'][i] - combined_data['rv_error'][i], combined_data['rv'][i] + combined_data['rv_error'][i]], phase_array[i], phase_array[i + 1], color=colors_array[i], alpha=0.5)

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
        num_species = len(wind_chars)
        color_map = pl.cm.get_cmap('nipy_spectral', num_species)  # Choosing a colormap with sufficient distinct colors
        
        species_colors = {species: color_map(i) for i, species in enumerate(wind_chars.keys())}
        
        # Plotting for each species with assigned color
        for species, species_data in enumerate(wind_chars.items()):
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
        custom_lines = [pl.Line2D([0], [0], color=species_colors[species], lw=4) for species in wind_chars.keys()]

        # Create the legend
        ax.legend(custom_lines, [f'{species}' for species in wind_chars.keys()])

    # Save the plot
    plotname_combined = path_modifier_plots + 'plots/' + planet_name + '.' + temperature_profile + '.CombinedWindCharacteristics_Combined.pdf'   
    fig.savefig(plotname_combined, dpi=300, bbox_inches='tight')

    # Save the plot for each arm
    for arm in ['blue', 'red']:
        plotname_arm = path_modifier_plots + 'plots/' + planet_name + '.' + temperature_profile + f'.CombinedWindCharacteristics_{arm}.pdf'
        fig.savefig(plotname_arm, dpi=300, bbox_inches='tight')