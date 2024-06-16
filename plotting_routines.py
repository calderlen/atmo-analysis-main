from radiant import *
from run_all_ccfs import *
import pyfastchem
import numpy as np
import os
import matplotlib.pyplot as plt
from astropy import constants as const
from scipy.integrate import simps
from petitRADTRANS import Radtrans
from run_all_ccfs import *
from periodic_trends import plotter


abundance_species = ['Mg', 'Fe', 'Fe+', 'Na', 'Co', 'Cr','Cr+', 'Zn', 'Cu', 'Ca', 'Ti', 'Sc', 'Ru',]

# Make plot stacking SYSREM Resids and a single synthetic transmission spectra for the paper


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
            axs[i].set_xlabel('Wavelength (Å)')
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
# Make plot stacking RAW CCF, Doppler Shadow Model, and RAW CCF with Doppler Shadow Model removed





# Make plot stacking SNR map and 1D CCF with x axis RV-V_sys(kms-1) with different y-axes. For KSNR map the y-axis is Kp. For the 1D CCF the y-axis is the amplitude of the ccf standardized at abs(RV) greater than 40 km/s.

# Make plot overlaying phase-binned phase-resolved line profiles of Fe I and Fe II in blue arm.

# Make plot asssessing observability score of each element considered in this analysis
def calculate_integral(wavelengths, tau):
    ln_tau = np.log(tau)
    integral = simps(ln_tau, wavelengths)
    return integral

# Compute the integral for the combined opacities
integral_tau_all = calculate_integral(wavelengths, tau_all)


# Normalize the observability scores
max_observability_score = max(observability_scores.values())
normalized_observability_scores = {species: score / max_observability_score for species, score in observability_scores.items()}

# Print the normalized observability scores
for species, score in normalized_observability_scores.items():
    print(f"Observability score for {species}: {score:.3f}")

# Example setup for petitRADTRANS
def observability_table(species_dict, parameters, planet_name='KELT-20b', spectrum_type='transmission', temperature_profile='inverted-transmission-better', instrument='PEPSI', arm='combined'):

    R_host, R_pl, M_pl, Teff, gravity = get_planetary_parameters(planet_name)
    spectrum = {}

    for species_name_new, params in species_dict.items():
        vmr = params.get('vmr')

        template_wave, template_flux, pressures, atmosphere, parameters = make_new_model(instrument, species_name_new, vmr, spectrum_type, planet_name, temperature_profile, do_plot=False)

        temperature = parameters['Teq']
        abundances = parameters['abundances']
        P0 = parameters['P0']

        spectrum[species_name_new] = atmosphere.calc_transm(temperature, abundances, gravity, R_pl, P0)

    # Extract wavelength and combined opacity (tau_all)
    wavelengths = nc.c / atmosphere.freq / 1e-4  # Convert frequency to wavelength in Å


    # Example: Extract combined opacity from spectrum (replace with actual data extraction method)
    tau_all = spectrum['transm_radius']  # Example: Extract transmission radius

    # Example: Extract individual species opacities (replace with actual data extraction method)
    tau_species = {
        'Fe': spectrum['species_opacity']['Fe'],  # Example
        'Na': spectrum['species_opacity']['Na'],  # Example
        'Ca': spectrum['species_opacity']['Ca']   # Example
    }

    # Calculate the observability score for each species
    observability_scores = {}
    for species, tau in tau_species.items():
        tau_all_minus_x = tau_all - tau  # Combined opacities without the contribution of species x
        integral_tau_all_minus_x = calculate_integral(wavelengths, tau_all_minus_x)
        observability_score = integral_tau_all_minus_x - integral_tau_all
        observability_scores[species] = observability_score




        
