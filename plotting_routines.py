from radiant import *
from run_all_ccfs import *

path_modifier_plots = '/home/calder/Documents/atmo-analysis-main/'  #linux
path_modifier_data = '/home/calder/Documents/petitRADTRANS_data/'   #linux


# Make plot stacking SYSREM Resids and a single synthetic transmission spectra for the paper
species_dict = {'Cu I' : {'vmr' : 2.06e-8},
                'Cr I' : {'vmr' : 7.08e-07},
                'Cr II' : {'vmr' : 7.08e-07},
                'Ca I' : {'vmr' : 2.46e-06},
                'Zr II' : {'vmr' : 6.68e-10},
                'Zr I' : {'vmr' : 6.68e-10},
                'Zn I' : {'vmr' : 6.23e-8},
                'Y II' : {'vmr' : 2.78e-10},
                'Y I' : {'vmr' : 2.78e-10},
                'W I' : {'vmr' : 1.06e-11},
                'Tl I' : {'vmr' : 1.428e-11},
                'Ti I' : {'vmr' : 5.63e-9},
                'Ti II' : {'vmr' : 5.63e-9},
                'Sr I' : {'vmr' : 1.16e-9},
                'Sr II' : {'vmr' : 1.16e-9},
                'Sn I' : {'vmr' : 1.80e-10},
                'Sc I' : {'vmr' : 2.37e-9},
                'Sc II' : {'vmr' : 2.37e-9},
                'Ru I' : {'vmr' : 9.65e-11},
                'Rh I' : {'vmr' : 1.03e-11},
                'Pd I' : {'vmr' : 6.38e-11},
                'Pb I' : {'vmr' : 1.53e-10},
                'Os I' : {'vmr' : 3.84e-11},
                'Mo I' : {'vmr' : 1.30e-10},
                'Mn I' : {'vmr' : 2.31e-7},
                'Ir I' : {'vmr' : 3.59e-11},
                'In I' : {'vmr' : 1.08e-11},
                'Hf I' : {'vmr' : 1.22e-11},
                'Ge I' : {'vmr' : 7.15e-9},
                'Ga I' : {'vmr' : 1.80e-9},
                'Mg I' : {'vmr' : 6.08e-5},
                'Mg II' : {'vmr' : 6.08e-5}
                }

species_dict = dict(sorted(species_dict.items(), key=lambda item: item[1]['vmr']))
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
            parameters['Teq'] = 2262.
            
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

        if planet_name == 'WASP-189b' or planet_name == 'KELT-20b':
            axs[i].fill([4265,4265,4800,4800],[np.nanmin(template_flux),np.nanmax(template_flux),np.nanmax(template_flux),np.nanmin(template_flux)],color='cyan',alpha=0.25)
        if planet_name != 'WASP-189b':
            axs[i].fill([4800,4800,5441,5441],[np.nanmin(template_flux),np.nanmax(template_flux),np.nanmax(template_flux),np.nanmin(template_flux)],color='blue',alpha=0.25)
            
        axs[i].text(0.125, 0.85, species_name, transform=axs[i].transAxes)
        axs[i].plot(template_wave, template_flux, color='black')

        axs[i].fill([6278,6278,7419,7419],[np.nanmin(template_flux),np.nanmax(template_flux),np.nanmax(template_flux),np.nanmin(template_flux)],color='red',alpha=0.25)

        if i == n_spectra - 1:  # If this is the last subplot
            axs[i].tick_params(axis='x', which='both', labeltop=False, labelbottom=True)  # Only show x-axis label on the bottom
            axs[i].set_xlabel('Wavelength (Å)')
        else:
            axs[i].tick_params(axis='x', which='both', labeltop=False, labelbottom=False)  # Don't show x-axis label on other subplots
        axs[i].set_ylabel('normalized flux')

        plotout = path_modifier_plots+'plots/spectra.' + planet_name +  '.' + temperature_profile + '.pdf'
        pl.savefig(plotout,format='pdf')

# Make plot stacking RAW CCF, Doppler Shadow Model, and RAW CCF with Doppler Shadow Model removed





# Make plot stacking SNR map and 1D CCF with x axis RV-V_sys(kms-1) with different y-axes. For KSNR map the y-axis is Kp. For the 1D CCF the y-axis is the amplitude of the ccf standardized at abs(RV) greater than 40 km/s.

