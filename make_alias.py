from run_all_ccfs import *
from uncertainties import unumpy as unp
path_modifier_plots = '/home/calder/Documents/atmo-analysis-main/'

def make_alias(instrument="PEPSI", planet_name="KELT-20b", spectrum_type="transmission", temperature_profile="inverted-transmission-better", model_tag="alias", spec_one="Fe+", vmr_one=4.95e-5, spec_two="Mg", vmr_two=6.08e-5, arm='blue', observation_epoch = 'mock-obs'):    
    """
    Generate an alias spectrum for a given planet and two species of interest.

    Args:
        instrument (str, optional): The name of the instrument. Defaults to "PEPSI".
        planet_name (str, optional): The name of the planet. Defaults to "KELT-20b".
        spectrum_type (str, optional): The type of spectrum. Defaults to "transmission".
        temperature_profile (str, optional): The temperature profile. Defaults to "inverted-transmission-better".
        model_tag (str, optional): The model tag. Defaults to "alias".
        spec_one (str, optional): The name of the first species. Defaults to "Fe+".
        vmr_one (float, optional): The volume mixing ratio of the first species. Defaults to 5.39e-5.
        spec_two (str, optional): The name of the second species. Defaults to "Ni".
        vmr_two (float, optional): The volume mixing ratio of the second species. Defaults to 2.676e-06.
        arm (str, optional): The arm of the instrument. Defaults to 'blue'.

    Returns:
        None
    """
    #make the models for the two species of interest

    template_wave_1, template_flux_1,_, _, _ = make_new_model(instrument, spec_one, vmr_one, spectrum_type, planet_name, temperature_profile, do_plot=True)

    template_wave_2, template_flux_2, _, _, _ = make_new_model(instrument, spec_two, vmr_two, spectrum_type, planet_name, temperature_profile, do_plot=True)

    #load in the orbital phase of the data

    orbital_phase = np.load('data_products/KELT-20b.20190504.' + arm + '.' + spec_two + '.CCFs-raw.npy.phase.npy')
    
    Period, epoch, M_star, RV_abs, i, M_p, R_p, RA, Dec, Kp_expected, half_duration_phase, Ks_expected = get_planet_parameters(planet_name)

    n_spectra = len(orbital_phase)

    #limit to the wavelength range of one PEPSI CD

    goods = (template_wave_1 >= 4800.) & (template_wave_1 < 5441.)
    template_wave_1, template_flux_1 = template_wave_1[goods], template_flux_1[goods]

    goods = (template_wave_2 >= 4800.) & (template_wave_2 < 5441.)
    template_wave_2, template_flux_2 = template_wave_2[goods], template_flux_2[goods]

    mock_spectra = np.zeros((n_spectra, len(template_wave_1)))

    mock_wave = np.zeros((n_spectra, len(template_wave_1)))

    
    for i in range (n_spectra): mock_wave[i,:] = template_wave_1
    
    #fluxin, Kp_true, V_sys_true = inject_model(Kp_expected, orbital_phase, mock_wave, mock_spectra, template_wave_1, template_flux_1 + np.interp(template_wave_1, template_wave_2, template_flux_2), n_spectra)
    fluxin, Kp_true, V_sys_true = inject_model(Kp_expected, orbital_phase, mock_wave, mock_spectra, template_wave_1, template_flux_1, n_spectra)

    #make a model spectrum with the template appropriately shifted to the planetary orbital motion
    #fluxin, Kp_true, V_sys_true = inject_model(Kp_expected, orbital_phase, mock_wave, mock_spectra, template_wave_2, template_flux_2, n_spectra)

    #do the cross-correlation and associated processing
    drv, cross_cor, sigma_cross_cor = get_ccfs(mock_wave, mock_spectra, np.ones_like(mock_spectra), template_wave_2, template_flux_2, n_spectra, mock_spectra, np.where(template_wave_1 > 0.))
    
    for i in range (n_spectra):
        cross_cor[i,:]-=np.mean(cross_cor[i,:])
        sigma_cross_cor[i,:] = np.sqrt(sigma_cross_cor[i,:]**2 + np.sum(sigma_cross_cor[i,:]**2)/len(sigma_cross_cor[i,:])**2)
        #I guess the following is OK as long as there isn't a strong peak, which there shouldn't be in any of the individual CCFs
        cross_cor[i,:]/=np.std(cross_cor[i,:])

    snr, Kp, drv, cross_cor, sigma_shifted_ccfs, ccf_weights = combine_ccfs(drv, cross_cor, sigma_cross_cor, orbital_phase, n_spectra, np.ones_like(orbital_phase), half_duration_phase, temperature_profile)
    
    plotsnr, amps, amps_error, rv, rv_error, width, width_error, idx, drv_restricted, plotsnr_restricted, residual_restricted, pl = make_shifted_plot(snr, planet_name, observation_epoch, arm, spec_one+spec_two + 'ACF', model_tag, RV_abs, Kp_expected, V_sys_true, Kp_true, False, drv, Kp,  spec_one + spec_two + 'ACF', temperature_profile, sigma_shifted_ccfs, 'ccf', cross_cor, sigma_cross_cor, ccf_weights, plotformat = 'pdf')
    
    keep = Kp == np.round(unp.nominal_values(Kp_expected))
    ccf_1d = snr[keep,:]
    np.save('alias-drv.npy',drv)
    np.save(spec_two + '-alias-ccf.npy', ccf_1d)