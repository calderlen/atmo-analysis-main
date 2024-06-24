#import packages
import numpy as np
import matplotlib.pyplot as pl
from glob import glob
from astropy.io import fits
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

from radiant_fastchemprofile import *
from create_model import create_model, instantiate_radtrans

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

    sysrem_file = 'data_products/' + planet_name + '.' + observation_epoch + '.' + arm + '.SYSREM-' + str(n_systematics[0]) + '+' + str(n_systematics[1])+model_tag+'.npy'

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

        plotname = 'plots/' + planet_name + '.' + observation_epoch + '.' + arm + '.spectrum-SYSREM-'+ str(n_systematics[0]) + '+' + str(n_systematics[1])+'.pdf'

        psarr(corrected_flux, wave, orbital_phase, 'wavelength (Angstroms)', 'orbital phase', 'flux residual', filename=plotname,flat=True, ctable='gist_gray')

    else:
        corrected_flux = np.load(sysrem_file)
        corrected_error = np.load(sysrem_file+'.corrected-error.npy')
        U_sysrem = np.load(sysrem_file+'.Umatrix.npy')
        telluric_free = np.load(sysrem_file+'.telluric-region.npy')

    if method == 'ccf':
        ccf_file = 'data_products/'+ planet_name + '.' + observation_epoch + '.' + arm + '.' + species_name_ccf + model_tag + '.CCFs-raw.npy'

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




        #Make a plot
        plotname = 'plots/' + planet_name + '.' + observation_epoch + '.' + species_name_ccf + model_tag + '.' + arm + '.CCFs-raw.pdf'
        
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

        #import pdb; pdb.set_trace()

    if 'likelihood' in method:
        like_file = 'data_products/'+ planet_name + '.' + observation_epoch + '.' + arm + '.' + species_name_ccf + model_tag + '.likelihood-raw.npy'
        drv, lnL = get_likelihood(wave, corrected_flux, corrected_error, template_wave, template_flux, n_spectra, U_sysrem, telluric_free)

        np.save(like_file, lnL)
        np.save(like_file+'.phase.npy', orbital_phase)

        #Make a plot
        plotname = 'plots/' + planet_name + '.' + observation_epoch + '.' + species_name_ccf + model_tag + '.likelihoods-raw.pdf'
        psarr(lnL, drv, orbital_phase, 'v (km/s)', 'orbital phase', 'ln L', filename=plotname, ctable='gist_gray')

        #now need to combine the likelihoods along the planet orbit
        shifted_lnL, Kp, drv = combine_likelihoods(drv, lnL, orbital_phase, n_spectra, half_duration_phase, temperature_profile)

        make_shifted_plot(shifted_lnL, planet_name, observation_epoch, arm, species_name_ccf, model_tag, RV_abs, Kp_expected, V_sys_true, Kp_true, do_inject_model, True, drv, Kp, species_label, temperature_profile, method)

        
        

def combine_observations(observation_epochs, arms, planet_name, temperature_profile, species_label, species_name_ccf, model_tag, RV_abs, Kp_expected, do_inject_model, f, method):

    Period, epoch, M_star, RV_abs, i, M_p, R_p, RA, Dec, Kp_expected, half_duration_phase, Ks_expected = get_planet_parameters(planet_name)

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
    
                
            

def run_all_ccfs(planet_name, temperature_profile, species_label, vmr, do_inject_model, do_run_all, do_make_new_model, method):

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
        if planet_name == 'KELT-20b': observation_epochs = ['20210501', '20210518', '20230430', '20230615']
        if planet_name == 'WASP-12b': observation_epochs = ['20210303', '20220208']
        if planet_name == 'KELT-9b': observation_epochs = ['20210628']
        if planet_name == 'WASP-76b': observation_epochs = ['20211031']
        if planet_name == 'WASP-33b': observation_epochs = ['20220929', '20221202']
        if planet_name == 'WASP-189b': observation_epochs = ['20230327']
        if planet_name == 'TOI-1431b': observation_epochs = ['20231023']
        if planet_name == 'TOI-1518b': observation_epochs = ['20231106']
        


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

    Period, epoch, M_star, RV_abs, i, M_p, R_p, RA, Dec, Kp_expected, half_duration_phase, Ks_expected = get_planet_parameters(planet_name)

    
    
    combine_observations(observation_epochs, arms, planet_name, temperature_profile, species_label, species_name_ccf, model_tag, RV_abs, Kp_expected, do_inject_model, f, method)

    if species_label != 'FeH': combine_observations(observation_epochs, ['blue'], planet_name, temperature_profile, species_label, species_name_ccf, model_tag, RV_abs, Kp_expected, do_inject_model, f, method)

    if species_label != 'CaH': combine_observations(observation_epochs, ['red'], planet_name, temperature_profile, species_label, species_name_ccf, model_tag, RV_abs, Kp_expected, do_inject_model, f, method)

    f.close()
    