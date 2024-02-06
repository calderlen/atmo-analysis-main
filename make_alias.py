from run_all_ccfs import *

def make_alias(instrument="PEPSI", planet_name="KELT-20b", spectrum_type="transmission", temperature_profile="inverted-transmission-better", model_tag="alias", spec_one="Fe+", vmr_one = 5.39e-5,  spec_two="Ni", vmr_two = 2.676e-06):
    '''
    '''
    
    #make the models for the two species of interest
    template_wave_one, template_flux_one = make_new_model(instrument, spec_one, vmr_one, spectrum_type, planet_name, temperature_profile, do_plot=True)
    template_wave_two, template_flux_two = make_new_model(instrument, spec_two, vmr_two, spectrum_type, planet_name, temperature_profile, do_plot=True)

    #load in the orbital phases of the data
    orbital_phase = np.load('data_products/KELT-20b.20190504.blue.' + spec_two + '.CCFs-raw.npy.phase.npy')

    Period, epoch, M_star, RV_abs, i, M_p, R_p, RA, Dec, Kp_expected, half_duration_phase = get_planet_parameters(planet_name)

    n_spectra = len(orbital_phase)

    #limit to the wavelength range of one PEPSI CD

    goods = (template_wave_one >= 4800.) & (template_wave_one < 5441.)
    template_wave_one, template_flux_one = template_wave_one[goods], template_flux_one[goods]

    goods = (template_wave_two >= 4800.) & (template_wave_two < 5441.)
    template_wave_two, template_flux_two = template_wave_two[goods], template_flux_two[goods]

    mock_spectra = np.zeros((n_spectra, len(template_wave_one)))

    mock_wave = np.zeros((n_spectra, len(template_wave_one)))

    for i in range (n_spectra): mock_wave[i,:] = template_wave_one

    #fluxin, Kp_true, V_sys_true = inject_model(Kp_expected, orbital_phase, mock_wave, mock_spectra, template_wave_one, template_flux_one + np.interp(template_wave_one, template_wave_two, template_flux_two), n_spectra)
    #fluxin, Kp_true, V_sys_true = inject_model(Kp_expected, orbital_phase, mock_wave, mock_spectra, template_wave_one, template_flux_one, n_spectra)

    #make a model spectrum with the template appropriately shifted to the planetary orbital motion
    fluxin, Kp_true, V_sys_true = inject_model(Kp_expected, orbital_phase, mock_wave, mock_spectra, template_wave_two, template_flux_two, n_spectra)

    #do the cross-correlation and associated processing
    drv, cross_cor, sigma_cross_cor = get_ccfs(mock_wave, mock_spectra, np.ones_like(mock_spectra), template_wave_two, template_flux_two, n_spectra, mock_spectra, np.where(template_wave_one > 0.))

    for i in range (n_spectra): 
        cross_cor[i,:]-=np.mean(cross_cor[i,:])
        sigma_cross_cor[i,:] = np.sqrt(sigma_cross_cor[i,:]**2 + np.sum(sigma_cross_cor[i,:]**2)/len(sigma_cross_cor[i,:])**2)
        #I guess the following is OK as long as there isn't a strong peak, which there shouldn't be in any of the individual CCFs
        cross_cor[i,:]/=np.std(cross_cor[i,:])

    snr, Kp, drv, cross_cor_display = combine_ccfs(drv, cross_cor, sigma_cross_cor, orbital_phase, n_spectra, np.ones_like(orbital_phase), half_duration_phase, temperature_profile)

    make_shifted_plot(snr, planet_name, 'mock-obs', 'blue', spec_two + 'ACF', model_tag, RV_abs, Kp_expected, V_sys_true, Kp_true, True, True, drv, Kp, spec_two + 'ACF', temperature_profile, 'ccf')

    breakpoint()

    keep = Kp == np.round(Kp_expected)
    ccf_1d = snr[keep,:]

    np.save('alias-drv.npy',drv)
    np.save(spec_two + '-alias-ccf.npy',ccf_1d)