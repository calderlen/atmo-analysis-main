from run_all_ccfs import *
from uncertainties import unumpy as unp
path_modifier_plots = '/home/calder/Documents/atmo-analysis-main/'

def make_shifted_plot_alias(snr, planet_name, observation_epoch, arm, species_name_ccf, model_tag, RV_abs, Kp_expected, V_sys_true, Kp_true, do_inject_model, do_combine, drv, Kp, species_label, temperature_profile, method, plotformat='pdf', p0_gaussian=[35,0,4]):
    
    if method == 'ccf':
        outtag, zlabel = 'CCFs-shifted', 'SNR'
        plotsnr = snr[:]
    if 'likelihood' in method:
        outtag, zlabel = 'likelihood-shifted', '$\Delta\ln \mathcal{L}$'
        plotsnr=snr - np.max(snr)
    plotname = path_modifier_plots + 'plots/'+ planet_name + '.' + observation_epoch + '.' + arm + '.' + species_name_ccf + model_tag + '.' + outtag + '.' + plotformat

    if do_combine:
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

    keeprv = np.abs(drv-apoints[0]) <= 100.
    plotsnr, drv = plotsnr[:, keeprv], drv[keeprv]
    keepKp = np.abs(Kp-apoints[1]) <= 100.
    plotsnr, Kp = plotsnr[keepKp, :], Kp[keepKp]

    psarr(plotsnr, drv, Kp, '$V_{\mathrm{sys}}$ (km/s)', '$K_p$ (km/s)', zlabel, filename=plotname, ctable=ctable, alines=True, apoints=apoints, acolor='cyan', textstr=species_label+' '+model_label, textloc = np.array([apoints[0]-75.,apoints[1]+75.]), textcolor='cyan', fileformat=plotformat)

    # Initializing lists to store fit parameters
    amps = []
    amps_err = []
    centers = []
    centers_err = []
    sigmas = []
    sigmas_err = []

    Kp_slices = []
    Kp_slice_peak = []

    residuals = []
    chi2_red = []

    # Fitting gaussian to all 1D Kp slices
    for i in range(plotsnr.shape[0]):
        current_slice = plotsnr[i,:]
        Kp_slices.append(current_slice)
        Kp_slice_peak.append(np.max(current_slice[80:121]))

        popt, pcov = curve_fit(gaussian, drv, current_slice, p0=p0_gaussian)

        amps.append(popt[0])
        centers.append(popt[1])
        sigmas.append(popt[2])

        # Storing errors (standard deviations)
        amps_err.append(np.sqrt(pcov[0, 0]))
        centers_err.append(np.sqrt(pcov[1, 1]))
        sigmas_err.append(np.sqrt(pcov[2, 2]))

    # Selecting a specific Kp slice
    selected_idx = np.where(Kp == int((np.floor(Kp_true))))[0][0] #Kp slice corresponding to expected Kp
    selected_idx = np.argmax(Kp_slice_peak)                       #Kp slice corresponding to max SNR
    
    # Fitting a Gaussian to the selected slice
    popt_selected = [amps[selected_idx], centers[selected_idx], sigmas[selected_idx]]
    print('Selected SNR:', amps[selected_idx], '\n Selected Vsys:', centers[selected_idx], '\n Selected sigma:', sigmas[selected_idx], '\n Selected Kp:', Kp[selected_idx])

    # Computing residuals and chi-squared for selected slice
    residual = plotsnr[selected_idx, :] - gaussian(drv, *popt_selected)
    # chi2 = np.sum((residual / np.std(residual))**2)/(len(drv)-len(popt))

    # Initialize Figure and GridSpec objects
    fig = pl.figure(figsize=(12,8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])

    # Create Axes for the main plot and the residuals plot
    ax1 = pl.subplot(gs[0])
    ax2 = pl.subplot(gs[1], sharex=ax1)
    
    # Main Plot (ax1)
    ax1.plot(drv, plotsnr[selected_idx, :], 'k--', label='data', markersize=2)
    ax1.plot(drv, gaussian(drv, *popt_selected), 'r-', label='fit')

    # Species Label
    ax1.text(0.05, 0.99, species_label, transform=ax1.transAxes, verticalalignment='top', horizontalalignment='left', fontsize=12)

    pl.setp(ax1.get_xticklabels(), visible=False)
    ax1.set_ylabel('SNR')
    # Annotating the arm and species on the plot
    
    # Additional text information for the main plot
    params_str = f"Peak (a): {popt_selected[0]:.2f}\nMean (mu): {popt_selected[1]:.2f}\nSigma: {popt_selected[2]:.2f}\nKp: {Kp[selected_idx]:.0f}"
    ax1.text(0.01, 0.95, params_str, transform=ax1.transAxes, verticalalignment='top', fontsize=10)

    arm_species_text = f'Arm: {arm}'
    ax1.text(0.15, 0.95, arm_species_text, transform=ax1.transAxes, verticalalignment='top', fontsize=10)
    
    # Vertical line for the Gaussian peak center
    ax1.axvline(x=centers[selected_idx], color='b', linestyle='-', label='Center')

    # Vertical lines for sigma width (center ± sigma)
    #sigma_left = centers[selected_idx] - sigmas[selected_idx]
    #sigma_right = centers[selected_idx] + sigmas[selected_idx]
    #ax1.axvline(x=sigma_left, color='purple', linestyle='--', label='- Sigma')
    #ax1.axvline(x=sigma_right, color='purple', linestyle='--', label='+ Sigma')

    ax1.legend()

    # Add the horizontal line at 4 SNR
    ax1.axhline(y=4, color='g', linestyle='--', label=r'4 $\sigma$')    

    # Inset for residuals (ax2)
    ax2.plot(drv, residual, 'o-', markersize=1)
    ax2.set_xlabel('Velocity (km/s)')
    ax2.set_ylabel('Residuals')

    # Consider a clearer naming scheme
    snr_fit = path_modifier_plots + '/plots/' + planet_name + '.' + observation_epoch + '.' + arm + '.' + species_name_ccf + model_tag + '.SNR-Gaussian.pdf'
    # Save the plot
    fig.savefig(snr_fit, dpi=300, bbox_inches='tight')

    if arm == 'red':
        do_molecfit = True
    else:
        do_molecfit = False

    Period, epoch, M_star, RV_abs, i, M_p, R_p, RA, Dec, Kp_expected, half_duration_phase = get_planet_parameters(planet_name)

    if observation_epoch != 'mock-obs':
        wave, fluxin, errorin, jd, snr_spectra, exptime, airmass, n_spectra, npix = get_pepsi_data(arm, observation_epoch, planet_name, do_molecfit)
    else:
        # check for my application of the function to KELT-20b's transmission spectra observation. Remove when done.s
        wave, fluxin, errorin, jd, snr_spectra, exptime, airmass, n_spectra, npix = get_pepsi_data(arm, '20190504', planet_name, do_molecfit)
            
        
    orbital_phase = get_orbital_phase(jd, epoch, Period, RA, Dec)

    phase_min = np.min(orbital_phase)
    phase_max = np.max(orbital_phase)
    phase_array = np.linspace(phase_min, phase_max, np.shape(centers)[0])

    fig, ax1 = pl.subplots(figsize=(15,10))

    ax1.text(0.05, 0.99, species_label, transform=ax1.transAxes, verticalalignment='top', horizontalalignment='left', fontsize=12)


    ax1.errorbar(phase_array, centers, yerr=centers_err, fmt='o-', label='Center')
    ax1.set_xlabel('Orbital Phase')
    ax1.set_ylabel('Vsys', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_title('Vsys and Sigma vs. Orbital Phase')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(phase_array, sigmas, 'r-', label='Sigma')
    ax2.set_ylabel('Sigma', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.legend(loc='upper right')


    # Consider a clearer naming scheme
    wind_chars = path_modifier_plots +  planet_name + '.' + observation_epoch + '.' + arm + '.' + species_name_ccf + model_tag + '.Wind-characteristics.pdf'
    # Save the plot
    fig.savefig(wind_chars, dpi=300, bbox_inches='tight')

def make_alias(instrument="PEPSI", planet_name="KELT-20b", spectrum_type="transmission", temperature_profile="inverted-transmission-better", model_tag="alias", spec_one="Fe+", vmr_one=5.39e-5, spec_two="Ni", vmr_two=2.676e-06, arm='blue', p0_gaussian=[0,5,1]):    
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

    template_wave_1, template_flux_1 = make_new_model(instrument, spec_one, vmr_one, spectrum_type, planet_name, temperature_profile, do_plot=True)

    template_wave_2, template_flux_2 = make_new_model(instrument, spec_two, vmr_two, spectrum_type, planet_name, temperature_profile, do_plot=True)

    #load in the orbital phase of the data

    orbital_phase = np.load('data_products/KELT-20b.20190504.' + arm + '.' + spec_two + '.CCFs-raw.npy.phase.npy')
    
    Period, epoch, M_star, RV_abs, i, M_p, R_p, RA, Dec, Kp_expected, half_duration_phase = get_planet_parameters(planet_name)

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
    #fluxin, Kp_true, V_sys_true = inject_model(Kp_expected, orbital_phase, mock_wave, mock_spectra, template_wave_1, template_flux_1, n_spectra)

    #make a model spectrum with the template appropriately shifted to the planetary orbital motion
    fluxin, Kp_true, V_sys_true = inject_model(Kp_expected, orbital_phase, mock_wave, mock_spectra, template_wave_2, template_flux_2, n_spectra)

    #do the cross-correlation and associated processing
    drv, cross_cor, sigma_cross_cor = get_ccfs(mock_wave, mock_spectra, np.ones_like(mock_spectra), template_wave_2, template_flux_2, n_spectra)
    
    for i in range (n_spectra): 
        cross_cor[i,:]-=np.mean(cross_cor[i,:])
        sigma_cross_cor[i,:] = np.sqrt(sigma_cross_cor[i,:]**2 + np.sum(sigma_cross_cor[i,:]**2)/len(sigma_cross_cor[i,:])**2)
        #I guess the following is OK as long as there isn't a strong peak, which there shouldn't be in any of the individual CCFs
        cross_cor[i,:]/=np.std(cross_cor[i,:])

    snr, Kp, drv = combine_ccfs(drv, cross_cor, sigma_cross_cor, orbital_phase, n_spectra, np.ones_like(orbital_phase), half_duration_phase, temperature_profile)

    make_shifted_plot_alias(snr, planet_name, 'mock-obs', arm, spec_two + 'ACF', model_tag, RV_abs, Kp_expected, V_sys_true, Kp_true, True, True, drv, Kp, spec_two + 'ACF', temperature_profile, 'ccf', p0_gaussian=p0_gaussian)

    #breakpoint()
    keep = Kp == np.round(unp.nominal_values(Kp_expected))
    ccf_1d = snr[keep,:]

    np.save('alias-drv.npy',drv)
    np.save(spec_two + '-alias-ccf.npy', ccf_1d)