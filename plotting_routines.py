import csv
from radiant import *
from run_all_ccfs import *
import pyfastchem
import numpy as np
import os
import matplotlib.pyplot as plt
from astropy import constants as const
from scipy.integrate import simps
from run_all_ccfs import *

from petitRADTRANS import Radtrans
from petitRADTRANS import nat_cst as nc
from petitRADTRANS.physics import guillot_global

from bokeh.models import (
    ColumnDataSource,
    LinearColorMapper,
    LogColorMapper,
    ColorBar,
    BasicTicker,
)
from bokeh.plotting import figure, output_file
from bokeh.io import show as show_
from bokeh.sampledata.periodic_table import elements
from bokeh.transform import dodge

from bokeh.io import export_png
from bokeh.plotting import figure, output_file, show
from selenium import webdriver

from csv import reader
from matplotlib.colors import Normalize, LogNorm, to_hex
from matplotlib.cm import (
    plasma,
    inferno,
    magma,
    viridis,
    cividis,
    turbo,
    ScalarMappable,
)
from pandas import options
from typing import List
import warnings

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
def overlayArms(planet_name, temperature_profile, species_label, vmr, do_inject_model, do_run_all, do_make_new_model, method, phase_ranges):
    
    drv_restricted, plotsnr_restricted, residual_restricted = {}, {}, {}
    arms = ['blue', 'red']
    fit_params, ccf_parameters, observation_epochs,_ = run_all_ccfs(planet_name, temperature_profile, species_label, vmr, do_inject_model, do_run_all, do_make_new_model, method, phase_ranges)

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

    all_arms = ['blue', 'red', 'combined']
        
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

        for arm in all_arms:
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
    

def multiSpeciesCCF(planet_name, temperature_profile, species_dict, do_inject_model, do_run_all, do_make_new_model, method, phase_ranges):
    # This function only works with a single observation epoch! FIX THIS!

    ccf_params = {}
    if planet_name == 'KELT-20b': observation_epoch = '20190504'

    for species_label, params in species_dict.items():
        vmr = params.get('vmr')
        arm = str(params.get('arm'))

        fit_params, ccf_params, observation_epochs, plotsnr_restricted = run_all_ccfs(planet_name, temperature_profile, species_label, vmr, do_inject_model, do_run_all, do_make_new_model, method, phase_ranges)
        
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
    plotname = 'plots/' + planet_name + '.' + temperature_profile +'.'+ str(do_inject_model) +'.CombinedRVs.pdf'
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
        amps, amps_error, rv, rv_error, width, width_error, selected_idx, orbital_phase, fit_params, ccf_parameters, observation_epochs, plotsnr_restricted = run_all_ccfs(planet_name, temperature_profile, species_label, vmr, do_inject_model, do_run_all, do_make_new_model, method)

        # Initialize 'combined' key for each species
        if species_label not in line_profile:
            line_profile[species_label] = {'combined': {'combined': {}}}

        # Mask 'rv' and 'rv_error' values exceeding ±20 km/s
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
    plotname_combined = 'plots/' + planet_name + '.' + temperature_profile + 'CombinedWindCharacteristics_Combined.pdf'   
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
        template_wave, template_flux,_,_,_ = make_new_model(instrument, species_name, vmr, spectrum_type, planet_name, temperature_profile)

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


def calculate_observability_score(instrument_here, opacities_all, opacities_without_species, wavelengths):

    observability_scores = {}

    # Define the wavelength range in Å
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

    wavelengths = nc.c / atmosphere.freq / 1e-8  # Convert frequency to wavelength in Å

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




def phaseResolvedBinnedVelocities(planet_name, temperature_profile, species_dict, do_inject_model, do_run_all, do_make_new_model, method, num_binned_obs=3, phase_ranges='halves'):
  
    if do_inject_model:
        model_tag = '.injected-'+str(vmr)
    else:
        model_tag = ''

    Period, epoch, M_star, RV_abs, i, M_p, R_p, RA, Dec, Kp_expected, half_duration_phase, Ks_expected = get_planet_parameters(planet_name)

    Kp_here = unp.nominal_values(Kp_expected)

    arms = ['blue', 'red', 'combined']
        
    # Intializing first layer of dictionaries
    cross_cor_display, sigma_cross_cor, ccf_weights, sigma_shifted_ccfs = {}, {}, {}, {}
    orbital_phases = {}
    phase_bin, drv, binned_ccfs, var_shifted_ccfs = {}, {}, {}, {}
    RV, RVe, RVdiff = {}, {}, {}
    nphase, nv = {}, {}
    good = {}

    for species_label, params in species_dict.items():

        vmr = params.get('vmr')
    
        fit_params, ccf_parameters, observation_epochs, _ = run_all_ccfs(planet_name, temperature_profile, species_label, vmr, do_inject_model, do_run_all, do_make_new_model, method, phase_ranges)

        # Initializing the second layer of dictionaries
        if species_label not in binned_ccfs:
            cross_cor_display[species_label], sigma_cross_cor[species_label], ccf_weights[species_label], sigma_shifted_ccfs[species_label] = {}, {}, {}, {}
            binned_ccfs[species_label] = {}
            var_shifted_ccfs[species_label] = {}
            drv[species_label] = {}
            orbital_phases[species_label] = {}
            good[species_label] = {}
            RV[species_label] = {}


        for arm in arms:

            # Initializing the third layer of dictionaries
            if arm not in orbital_phases[species_label]:
                
                RV[species_label][arm], RVe[arm], RVdiff[arm] = {}, {}, {}
                phase_bin[arm] = {}

                orbital_phases[species_label][arm] = {}
                cross_cor_display[species_label][arm] = {}
                sigma_cross_cor[species_label][arm] = {}
                ccf_weights[species_label][arm] = {}
                sigma_shifted_ccfs[species_label][arm] = {}
                drv[species_label][arm] = {}

                binned_ccfs[species_label][arm] = {}
                var_shifted_ccfs[species_label][arm] = {}
                nphase[arm], nv[arm] = {}, {}
                good[species_label][arm] = {}

            for observation_epoch in observation_epochs:

                # Check to account for bad output dict setup in run_all_ccfs, notably arm and observation_epoch dict keys orders are switched

                if arm == 'combined':
                    observation_epoch = 'combined'

                if observation_epoch not in cross_cor_display[species_label][arm]:

                    # Initializing the fourth layer of dictionaries

                    cross_cor_display[species_label][arm][observation_epoch] = {}
                    sigma_cross_cor[species_label][arm][observation_epoch] = {}
                    ccf_weights[species_label][arm][observation_epoch] = {}
                    sigma_shifted_ccfs[species_label][arm][observation_epoch] = {}
                    drv[species_label][arm][observation_epoch] = {}
                    binned_ccfs[species_label][arm][observation_epoch] = {}
                    var_shifted_ccfs[species_label][arm][observation_epoch] = {}
                    good[species_label][arm][observation_epoch] = {}

                # Assigning run_all_ccfs outputs to the third- and fourth-layer dictionaries, again note that arm and observation_epoch dict keys orders are switched

                orbital_phases[species_label][arm] = fit_params[species_label][observation_epoch][arm]['orbital_phase']
                cross_cor_display[species_label][arm][observation_epoch] = ccf_parameters[species_label][observation_epoch][arm]['cross_cor_display']
                sigma_cross_cor[species_label][arm][observation_epoch] = ccf_parameters[species_label][observation_epoch][arm]['sigma_cross_cor']
                ccf_weights[species_label][arm][observation_epoch] = ccf_parameters[species_label][observation_epoch][arm]['ccf_weights']
                sigma_shifted_ccfs[species_label][arm][observation_epoch] = ccf_parameters[species_label][observation_epoch][arm]['sigma_shifted_ccfs']
                drv[species_label][arm][observation_epoch] = fit_params[species_label][observation_epoch][arm]['drv']
                binsize = np.max(orbital_phases[species_label][arm] - np.min(orbital_phases[species_label][arm])) / len(orbital_phases[species_label][arm])*num_binned_obs
                phase_bin[arm] = np.arange(np.min(orbital_phases[species_label][arm]), np.max(orbital_phases[species_label][arm]), binsize)
                nphase[arm], nv[arm] = len(phase_bin[arm]), len(drv[species_label][arm][observation_epoch])

                binned_ccfs[species_label][arm][observation_epoch] = np.zeros((nphase[arm], nv[arm]))
                var_shifted_ccfs[species_label][arm][observation_epoch] = np.zeros((nphase[arm], nv[arm]))

                i = 0

                Kp_here = unp.nominal_values(Kp_expected)
                RV[species_label][arm] = Kp_here*np.sin(2.*np.pi*orbital_phases[species_label][arm])

                order = np.argsort(orbital_phases[species_label][arm])
                good[species_label][arm][observation_epoch] = np.abs(drv[species_label][arm][observation_epoch]) < 10.

                RV[species_label][arm] = RV[species_label][arm][order]
                orbital_phases[species_label][arm] = orbital_phases[species_label][arm][order]

                for x in range(len(orbital_phases[species_label][arm])):
                    #restrict to only in-transit spectra if doing transmission:
                    #print(orbital_phase[x])
                    #if not 'transmission' in temperature_profile or np.abs(orbital_phase[x]) <= half_duration_phase:
                    if np.abs(orbital_phases[species_label][arm][x]) <= half_duration_phase:
                        phase_here = np.argmin(np.abs(phase_bin[arm] - orbital_phases[species_label][arm][x]))
                        temp_ccf = np.interp(drv[species_label][arm][observation_epoch], drv[species_label][arm][observation_epoch]-RV[species_label][arm][x], cross_cor_display[species_label][arm][observation_epoch][x, :], left=0., right=0.0)
                        sigma_temp_ccf = np.interp(drv[species_label][arm][observation_epoch], drv[species_label][arm][observation_epoch]-RV[species_label][arm][x], sigma_cross_cor[species_label][arm][observation_epoch][x, :], left=0., right=0.0)
                        binned_ccfs[species_label][arm][observation_epoch][phase_here,:] += temp_ccf * ccf_weights[species_label][arm][observation_epoch][x]
                        #use_for_sigma = (np.abs(drv) <= 100.) & (temp_ccf != 0.)
                        use_for_sigma = (np.abs(drv[species_label][arm][observation_epoch]) > 100.) & (temp_ccf != 0.)
                        #this next is b/c the uncertainties produced through the ccf routine are just wrong
                        var_shifted_ccfs[species_label][arm][observation_epoch][phase_here,:] += np.std(temp_ccf[use_for_sigma])**2 * ccf_weights[species_label][arm][observation_epoch][x]**2

                sigma_shifted_ccfs[species_label][arm][observation_epoch] = np.sqrt(var_shifted_ccfs[species_label][arm][observation_epoch])

    


    # Initializing the first layer of the dictionaries
    rvs, widths, rverrors, widtherrors, snr = {}, {}, {}, {}, {}
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'grey', 'olive', 'cyan']
    species_colors = {species: colors[i % len(colors)] for i, species in enumerate(species_dict.keys())} 

    for species_label in species_dict.keys():
        
        # Initialize the second layer of dictionaries
        if species_label not in rvs:
            rvs[species_label] = {}
            widths[species_label] = {}
            rverrors[species_label] = {}
            widtherrors[species_label] = {}
            snr[species_label] = {}

        for arm in arms:
     
            # Initialize the third layer of dictionaries
            if arm not in rvs[species_label]:
                rvs[species_label][arm] = {}
                widths[species_label][arm] = {}
                rverrors[species_label][arm] = {}
                widtherrors[species_label][arm] = {}

                snr[species_label][arm] = {}
                    
            for observation_epoch in observation_epochs:

                if arm == 'combined':
                    observation_epoch = 'combined'

                # Initialize the fourth layer of dictionaries
                if observation_epoch not in rvs[species_label][arm]:
                    rvs[species_label][arm][observation_epoch] = {}
                    widths[species_label][arm][observation_epoch] = {}
                    rverrors[species_label][arm][observation_epoch] = {}
                    widtherrors[species_label][arm][observation_epoch] = {}

                    snr[species_label][arm][observation_epoch] = {}

                rvs[species_label][arm][observation_epoch] = np.zeros(nphase[arm])
                widths[species_label][arm][observation_epoch] = np.zeros(nphase[arm])
                rverrors[species_label][arm][observation_epoch] = np.zeros(nphase[arm])

                for i in range(0,nphase[arm]):
                    peak = np.argmax(binned_ccfs[species_label][arm][observation_epoch][i,good[species_label][arm][observation_epoch]])
                    rv_array = drv[species_label][arm][observation_epoch][good[species_label][arm][observation_epoch]]
                    binned_ccf_array = binned_ccfs[species_label][arm][observation_epoch][i,good[species_label][arm][observation_epoch]]
                    fit_amplitude =  binned_ccfs[species_label][arm][observation_epoch][i,peak]
                    mask = good[species_label][arm][observation_epoch]
                    drv_masked = drv[species_label][arm][observation_epoch][mask]
                    sigma_ccf_array = sigma_shifted_ccfs[species_label][arm][observation_epoch][i,good[species_label][arm][observation_epoch]]

                    popt, pcov = curve_fit(gaussian, rv_array, binned_ccf_array, p0=[fit_amplitude, drv_masked[peak], 2.5], sigma=sigma_ccf_array, maxfev=1000000)

                    rvs[species_label][arm][observation_epoch][i] = popt[1]
                    widths[species_label][arm][observation_epoch][i] = popt[2]
                    rverrors[species_label][arm][observation_epoch][i] = np.sqrt(pcov[1,1])
                    widtherrors[species_label][arm][observation_epoch][i] = np.sqrt(pcov[2,2])

                use_for_snr = np.abs(drv[species_label][arm][observation_epoch] > 100.)
                snr[species_label][arm] = binned_ccfs[species_label][arm][observation_epoch] / np.std(binned_ccfs[species_label][arm][observation_epoch][:,use_for_snr])
        
    for arm in arms:
        fig, ax = pl.subplots(layout='constrained', figsize=(10,8))
        for species_label in species_dict.keys():
            for observation_epoch in observation_epochs:
                if arm == 'combined':
                    observation_epoch = 'combined'
                
                ax.plot([0.,0.],[np.min(phase_bin[arm]), np.max(phase_bin[arm])],':',color='white')
                
                # Create a mask for RVs within ±10 of zero
                mask = (rvs[species_label][arm][observation_epoch] >= -10) & (rvs[species_label][arm][observation_epoch] <= 10)
                
                # Apply the mask to RVs and phase_bin for plotting
                ax.plot(rvs[species_label][arm][observation_epoch][mask], phase_bin[arm][mask], 'o', color=species_colors[species_label], label=species_label)
                ax.errorbar(rvs[species_label][arm][observation_epoch][mask], phase_bin[arm][mask], xerr = rverrors[species_label][arm][observation_epoch][mask], color=species_colors[species_label], fmt='none') 

                pl.xlabel('$\Delta V$ (km/s)')
                pl.ylabel('Orbital Phase (fraction)')
                # add vertical line at 0
                pl.axvline(x=0, color='black', linestyle='--', linewidth=0.66)
                ax.set_xlim([-25.,25.])
                secax = ax.secondary_yaxis('right', functions=(phase2angle, angle2phase))
                secax.set_ylabel('Orbital Phase (degrees)')
                pl.legend()
                # add text showing number of binned observations
                pl.text(0.05, 0.95, f'{num_binned_obs} binned observations', transform=ax.transAxes, fontsize=12, verticalalignment='top')
                # add text showing arm and observation epoch
                pl.text(0.05, 0.90, f'{arm} - {observation_epoch}', transform=ax.transAxes, fontsize=12, verticalalignment='top')
                pl.savefig('plots/'+planet_name+'.' + '.' + str(num_binned_obs) + '.' +  observation_epoch + '.' + arm +'.phase-binned+RVs-overlaid.pdf', format='pdf')
        pl.clf()
            
            
def aliasPlots(species_dict, instrument="PEPSI", planet_name="KELT-20b", spectrum_type="transmission", temperature_profile="inverted-transmission-better", model_tag="alias", spec_one='Fe', vmr_one=4.95e-05):
    
    arms = ['blue', 'red']
    
    Period, epoch, M_star, RV_abs, i, M_p, R_p, RA, Dec, Kp_expected, half_duration_phase, Ks_expected = get_planet_parameters(planet_name)
    
    observation_epoch = 'mock-obs'
    
    template_wave, template_flux,_, _, _ = make_new_model(instrument, spec_one, vmr_one, spectrum_type, planet_name, temperature_profile, do_plot=True)
    
    goods = (template_wave >= 4800.) & (template_wave < 5441.)
    
    template_wave, template_flux = template_wave[goods], template_flux[goods]
 
    
    
    for species, params in species_dict.items():
        for arm in arms:
            spec_search = get_species_keys(species)[1]
            vmr_search = params.get('vmr')
            template_wave_search, template_flux_search,_, _, _ = make_new_model(instrument, spec_search, vmr_search, spectrum_type, planet_name, temperature_profile, do_plot=True)
            
            goods = (template_wave_search >= 4800.) & (template_wave_search < 5441.)
            template_wave_search, template_flux_search = template_wave_search[goods], template_flux_search[goods]
            
            orbital_phase = np.load('data_products/KELT-20b.20190504.' + arm + '.' + spec_search + '.CCFs-raw.npy.phase.npy')
            
            n_spectra = len(orbital_phase)
            
            mock_spectra = np.zeros((n_spectra, len(template_wave)))
            mock_wave = np.zeros((n_spectra, len(template_wave)))
            
            for i in range (n_spectra): mock_wave[i,:] = template_wave
            
            fluxin, Kp_true, V_sys_true = inject_model(Kp_expected, orbital_phase, mock_wave, mock_spectra, template_wave, template_flux, n_spectra)
            
            drv, cross_cor, sigma_cross_cor = get_ccfs(mock_wave, mock_spectra, np.ones_like(mock_spectra), template_wave_search, template_flux_search, n_spectra, mock_spectra, np.where(template_wave > 0.))

            for i in range (n_spectra):
                cross_cor[i,:]-=np.mean(cross_cor[i,:])
                sigma_cross_cor[i,:] = np.sqrt(sigma_cross_cor[i,:]**2 + np.sum(sigma_cross_cor[i,:]**2)/len(sigma_cross_cor[i,:])**2)
                #I guess the following is OK as long as there isn't a strong peak, which there shouldn't be in any of the individual CCFs
                cross_cor[i,:]/=np.std(cross_cor[i,:])

            snr, Kp, drv, cross_cor, sigma_shifted_ccfs, ccf_weights = combine_ccfs(drv, cross_cor, sigma_cross_cor, orbital_phase, n_spectra, np.ones_like(orbital_phase), half_duration_phase, temperature_profile)
            
            plotsnr, amps, amps_error, rv, rv_error, width, width_error, idx, drv_restricted, plotsnr_restricted, residual_restricted, pl = make_shifted_plot(snr, planet_name, observation_epoch, arm, spec_one + spec_search + '_Alias', model_tag, RV_abs, Kp_expected, V_sys_true, Kp_true, False, drv, Kp,  spec_one + spec_search + '.' + temperature_profile, sigma_shifted_ccfs, 'ccf', cross_cor, sigma_cross_cor, ccf_weights, plotformat = 'pdf')