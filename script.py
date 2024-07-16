from plotting_routines import *
#from run_all_ccfs_old import *

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
    'Ti II' : {'vmr' : 5.63e-9, 'arm':'blue'}}

species_dict_temp = {
    'Fe I' : {'vmr' : 4.95e-5, 'arm':'combined'},
    'Fe II' : {'vmr' : 4.946571802411902e-05, 'arm':'combined'}}

species_dict_full = {
    'Al I' : {'vmr' : 6.284450120736337e-07, 'arm':'blue'},
    'B I' : {'vmr' : 8.601920308199416e-10, 'arm':'blue'},
    'Ba I' : {'vmr' : 3.195916432382493e-10, 'arm':'blue'},
    #'Ba II' : {'vmr' : 3.195916432382493e-10, 'arm':'blue'},
    'Be I' : {'vmr' : 4.117137911265462e-11, 'arm':'blue'},
    'Ca I' : {'vmr' : 2.4587155473819153e-06, 'arm':'blue'},
    'Ca II' : {'vmr' : 2.4587155473819153e-06, 'arm':'blue'},
    'Co I' : {'vmr' : 1.4948340961688448e-07, 'arm':'blue'},
    'Cr I' : {'vmr' : 7.079592959038503e-07, 'arm':'blue'},
    'Cr II' : {'vmr' : 7.079592959038503e-07, 'arm':'blue'},
    'Cs I' : {'vmr' : 2.0634569602091076e-11, 'arm':'blue'},
    'Cu I' : {'vmr' : 2.063093447370899e-08, 'arm':'blue'},
    'Fe I' : {'vmr' : 4.946571802411902e-05, 'arm':'combined'},
    'Fe II' : {'vmr' : 4.946571802411902e-05, 'arm':'combined'},
    'Ga I' : {'vmr' : 1.7971958817835836e-09, 'arm':'blue'},
    'Ge I' : {'vmr' : 7.154765674272624e-09, 'arm':'blue'},
    'Hf I' : {'vmr' : 1.2150535391184618e-11, 'arm':'blue'},
    'In I' : {'vmr' : 1.0829176066220596e-11, 'arm':'blue'},
    'Ir I' : {'vmr' : 3.585877215540056e-11, 'arm':'blue'},
    'K I' : {'vmr' : 1.9642279376167866e-07, 'arm':'blue'},
    'Li I' : {'vmr' : 1.565292177052519e-11, 'arm':'blue'},
    'Mg I' : {'vmr' : 6.081422083824978e-05, 'arm':'blue'},
    'Mg II' : {'vmr' : 6.081422083824978e-05, 'arm':'blue'},
    'Mn I' : {'vmr' : 2.3059703152266368e-07, 'arm':'blue'},
    'Mo I' : {'vmr' : 1.301953324062707e-10, 'arm':'blue'},
    'N I' : {'vmr' : 6.153003923345118e-13, 'arm':'blue'},
    'Na I' : {'vmr' : 2.8215497643996917e-06, 'arm':'combined'},
    'Nb I' : {'vmr' : 5.065186197357635e-11, 'arm':'blue'},
    'Ni I' : {'vmr' : 2.568167989682126e-06, 'arm':'blue'},
    'Os I' : {'vmr' : 3.8423366626628385e-11, 'arm':'blue'},
    'Pb I' : {'vmr' : 1.5296617770866685e-10, 'arm':'blue'},
    'Pd I' : {'vmr' : 6.376691619322586e-11, 'arm':'blue'},
    'Rb I' : {'vmr' : 3.5858772155400563e-10, 'arm':'blue'},
    'Rh I' : {'vmr' : 1.0341782855961415e-11, 'arm':'blue'},
    'Ru I' : {'vmr' : 9.651513328234593e-11, 'arm':'blue'},
    'Sc I' : {'vmr' : 2.369165581667039e-09, 'arm':'blue'},
    'Sc II' : {'vmr' : 2.369165581667039e-09, 'arm':'blue'},
    'Si I' : {'vmr' : 1.9784001872259786e-09, 'arm':'blue'},
    'Sn I' : {'vmr' : 1.7971958817835839e-10, 'arm':'blue'},
    'Sr I' : {'vmr' : 1.1603671214772389e-09, 'arm':'blue'},
    'Sr II' : {'vmr' : 1.1603671214772389e-09, 'arm':'blue'},
    'Ti I' : {'vmr' : 5.6280315001592464e-09, 'arm':'blue'},
    'Ti II' : {'vmr' : 5.6280315001592464e-09, 'arm':'blue'},
    'Tl I' : {'vmr' : 1.4275634322309038e-11, 'arm':'blue'},
    'V I' : {'vmr' : 5.632160208295746e-09, 'arm':'blue'},
    'W I' : {'vmr' : 1.0582673924194556e-11, 'arm':'blue'},
    'Y I' : {'vmr' : 2.7835268491474936e-10, 'arm':'blue'},
    'Y II' : {'vmr' : 2.7835268491474936e-10, 'arm':'blue'},
    'Zn I' : {'vmr' : 6.23154039405557e-08, 'arm':'blue'},
    'Zr I' : {'vmr' : 6.677215836709593e-10, 'arm':'blue'}
}

species_dict_final = {
    'Ba II' : {'vmr' : 2.4587155473819153e-06, 'arm':'blue'},
    'Cr I' : {'vmr' : 7.079592959038503e-07, 'arm':'blue'},
    'Cu I' : {'vmr' : 2.063093447370899e-08, 'arm':'blue'},
    'Fe I' : {'vmr' : 4.946571802411902e-05, 'arm':'combined'},
    'Fe II' : {'vmr' : 4.946571802411902e-05, 'arm':'combined'},
    'Mg I' : {'vmr' : 6.081422083824978e-05, 'arm':'blue'},
    'Mn I' : {'vmr' : 2.3059703152266368e-07, 'arm':'blue'},
    'Na I' : {'vmr' : 2.8215497643996917e-06, 'arm':'blue'}
}


species_dict_molecules = {
    #'AlO' : {'vmr' : 1.0e-5, 'arm':'blue'},
    'CaH' : {'vmr' : 1.0e-5, 'arm':'blue'},
    'CaO' : {'vmr' : 1.0e-5, 'arm':'blue'},
    'CO' : {'vmr' : 1.0e-5, 'arm':'blue'},
    'CO2' : {'vmr' : 1.0e-5, 'arm':'blue'},
    'FeH' : {'vmr' : 1.0e-5, 'arm':'blue'},
    'H2O' : {'vmr' : 1.0e-5, 'arm':'blue'},
    'MgH' : {'vmr' : 1.0e-5, 'arm':'blue'},
    'NaH' : {'vmr' : 1.0e-5, 'arm':'blue'},
    'TiO' : {'vmr' : 1.0e-5, 'arm':'blue'},
    'VO' : {'vmr' : 1.0e-5, 'arm':'blue'}
}


# metal oxides and hydrides that should be checked
# BaH, BaO, CrH, CrO, CuH, CuO, FeO, MgO, MnH, MnO, NaO

#species_dict_final = dict(sorted(species_dict_final.items(), key=lambda item: item[1]['vmr'], reverse=True))
                    
#Make plot stacking all of the synthetic transmission spectra for appendix
#multiSpeciesCCF('KELT-20b', 'inverted-transmission-better', species_dict_temp, False, True, True, 'ccf', 'halves')
multiSpeciesCCF('KELT-20b', 'inverted-transmission-better', species_dict_molecules, False, True, True, 'ccf', 'halves')


#for species_label, species_params in species_dict_final.items():
#    vmr = species_params['vmr']
#    #overlayArms('KELT-20b', 'inverted-transmission-better', species_label, vmr, False, True, True, 'ccf')
#    overlayArms('KELT-20b', 'inverted-transmission-better', species_label, vmr, False, True, True, 'ccf', 'ingress-egress')pythee

#phaseResolvedBinnedVelocities('KELT-20b', 'inverted-transmission-better', species_dict_temp, False, True, True, 'ccf', phase_ranges='halves')
    
#species_dict = dict(sorted(species_dict.items(), key=lambda item: item[1]['vmr'], reverse=True))
#make_spectrum_plots(species_dict_final)


#generate_observability_table('KELT-20b','inverted-transmission-better', 'PEPSI', species_dict_full, 'guillot')

#aliasPlots(species_dict_full, instrument="PEPSI", planet_name="KELT-20b", spectrum_type="transmission", temperature_profile="inverted-transmission-better", model_tag="alias", spec_one='Fe+', vmr_one=4.95e-05)