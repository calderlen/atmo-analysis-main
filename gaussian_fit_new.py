
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
                
                popt, pcov = curve_fit(gaussian, drv, plotsnr[i,:], p0=[plotsnr[i, peak], drv[peak], 2.55035], sigma = sigma_shifted_ccfs[i,:], maxfev=100000)

                amps[i] = popt[0]
                rv[i] = popt[1]
                width[i] = popt[2]
                amps_error[i] = np.sqrt(pcov[0,0])
                rv_error[i] = np.sqrt(pcov[1,1])
                width_error[i] = np.sqrt(pcov[2,2])
                
                idx = np.where( (Kp == int(np.floor(Kp_true))) )[0][0] #Kp slice corresponding to expected Kp
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
    idx = np.argmax(slice_peak)
    
    Kp = np.arange(50, 350, 1)
    rv_chars_temp, rv_chars_error_temp = np.zeros((len(Kp), n_spectra)), np.zeros((len(Kp), n_spectra))
    rv_chars, rv_chars_error = np.zeros((len(Kp), n_spectra)), np.zeros((len(Kp), n_spectra))
    phase_array = np.linspace(np.min(orbital_phase), np.max(orbital_phase), num=n_spectra)   
    slice_peak_chars, slice_peak_chars_temp = np.zeros(len(Kp)), np.zeros(len(Kp))
    
    i = 0
    for Kp_i in Kp:
        RV = Kp_i*np.sin(2.*np.pi*orbital_phase)
        
        for j in range(n_spectra):
            #restrict to only in-transit spectra if doing transmission:
            #also want to leave out observations in 2ndary eclipse!
            phase_here = np.argmin(np.abs(phase_array - orbital_phase[j]))
            temp_ccf = np.interp(drv, drv-RV[j], cross_cor[j, :], left=0., right=0.0)
            temp_ccf *= ccf_weights[j]
            peak = np.argmax(temp_ccf)
            sigma_temp_ccf = np.interp(drv, drv-RV[j], sigma_cross_cor[j, :], left=0., right=0.0)
            sigma_temp_ccf = sigma_temp_ccf**2 * ccf_weights[j]**2
            popt, pcov = curve_fit(gaussian, drv, temp_ccf, p0=[temp_ccf[peak], drv[peak], 2.5], sigma = np.sqrt(sigma_temp_ccf), maxfev=100000)
            rv_chars_temp[i,phase_here] = popt[1]
            rv_chars_error_temp[i,phase_here] = np.sqrt(pcov[1,1])
            slice_peak_chars_temp[i] = temp_ccf[peak]
            
            if not (390 <= peak <= 411):
                rv_chars[i,phase_here] = np.nan
                rv_chars_error[i,phase_here] = np.nan
                slice_peak_chars[i] = np.nan
                                
        i+=1

    fig, ax1 = pl.subplots(figsize=(8,8))

    ax1.text(0.05, 0.99, species_label, transform=ax1.transAxes, verticalalignment='top', horizontalalignment='left', fontsize=12)
    ax1.plot(rv_chars[idx,:], phase_array, '-', label='Center', color='b')
    ax1.fill_betweenx(phase_array, rv_chars[idx,:] - rv_chars_error[idx, :], rv_chars[idx,:] + rv_chars_error[idx,:], color='blue', alpha=0.2, zorder=2)
    ax1.set_ylabel('Orbital Phase')
    ax1.set_xlabel('$v_{sys}$ (km/s)', color='b')
    ax1.tick_params(axis='x', labelcolor='b')
    
    line_profile = path_modifier_plots + 'plots/' + planet_name + '.' + observation_epoch + '.' + arm + '.' + species_name_ccf + model_tag + '.line-profile.pdf'
    fig.savefig(line_profile, dpi=300, bbox_inches='tight')

    return amps, amps_error, rv, rv_error, width, width_error, residual, do_molecfit, idx, line_profile, drv_restricted, plotsnr_restricted, residual_restricted