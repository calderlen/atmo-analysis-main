from run_all_ccfs import *
from plotting_routines import *


species_dict = {
    'Mg I' : {'vmr' : 6.08e-5, 'arm':'blue'},
    'Fe I' : {'vmr' : 4.95e-5, 'arm':'combined'},
    'Fe II' : {'vmr' : 4.95e-5, 'arm':'combined'},
    'Na I' : {'vmr' : 2.82e-6,'arm':'combined'},
    'Co I' : {'vmr' : 1.49e-7, 'arm':'blue'},
    'Cr I' : {'vmr' : 7.08e-7, 'arm':'combined'},
    'Cr II' : {'vmr' : 7.08e-7, 'arm':'blue'},
    'Zn I' : {'vmr' : 6.23e-8, 'arm':'blue'},
    'Cu I' : {'vmr' : 2.06e-8, 'arm':'blue'},
    'Ca I' : {'vmr' : 2.10e-8, 'arm':'blue'},
    'Ti I' : {'vmr' : 5.63e-9, 'arm':'blue'},
    'Ti II' : {'vmr' : 5.63e-9, 'arm':'blue'},
    'Sc II' : {'vmr' : 5.63e-9, 'arm':'blue'},
    'Ru I' : {'vmr' : 9.65e-11, 'arm':'blue'}
}

species_dict = dict(sorted(species_dict.items(), key=lambda item: item[1]['vmr']))
                    
#Make plot stacking all of the synthetic transmission spectra for appendix
multiSpeciesCCF('KELT-20b', 'inverted-transmission-better', species_dict, False, True, True, 'ccf')

#for species_label, species_params in species_dict.items():
#    vmr = species_params['vmr']
#    overlayArms('KELT-20b', 'inverted-transmission-better', species_label, vmr, False, True, True, 'ccf')

    
#species_dict = dict(sorted(species_dict.items(), key=lambda item: item[1]['vmr'], reverse=True))
#make_spectrum_plots(species_dict)


