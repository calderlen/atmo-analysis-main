from run_all_ccfs import *

#species_dict = {'V I': {'vmr': 5.62e-9},  
#                'Ca I': {'vmr': 2.10e-8},
#                'Fe II' : {'vmr' : 5.39e-5},
#                'Fe_1_Kurucz' : {'vmr' : 5.39e-5},
#
#                'Fe I' : {'vmr' : 5.39e-5},
#                'Fe_0_Kurucz' : {'vmr' : 5.39e-5},
#                'Fe_0_Vald' : {'vmr' : 5.39e-5},
#                'Fe I' : {'vmr' : 5.39e-5},
#
#                'Co_Kurucz' : {'vmr' : 1.67e-7}
#                }

#                'Na I': {'vmr': 2.94e-6},

#                 'Na_Burrows': {'vmr': 2.94e-6},
#               'Na_Allard': {'vmr': 2.94e-6},
 #               'Na_lor_cut': {'vmr': 2.94e-6},
                
species_dict = {'V I': {'vmr': 5.62e-9},  
                'Ca I': {'vmr': 2.10e-8},
                'Co I' : {'vmr' : 1.67e-7},
                'Na I': {'vmr': 2.94e-6},
                'Fe II' : {'vmr' : 5.39e-5},
                'Fe I' : {'vmr' : 5.39e-5},
}                
    #if species_label == 'Na_Allard':
    #    species_names.add(('Na_allard_new', 'Na'))
    #if species_label == 'Na_Burrows':
    #    species_names.add(('Na_burrows', 'Na'))
    #if species_label == 'Na_lor_cut':
    #    species_names.add(('', 'Na'))
    #if species_label == 'Co_0_Kurucz':
    #    species_names.add(('Co_0_Kurucz', 'Co'))
    #if species_label == 'Cr_1_VALD':
    #    species_names.add(('Cr_1_VALD', 'Cr'))
    #if species_names == 'Fe_0_Kurucz':
    #    species_names.add(('Fe-Kurucz', 'Fe'))
    #if species_label == 'Fe_1_Kurucz':
    #    species_names.add(('Fe_1_Kurucz', 'Fe+'))
    #if species_label == 'Fe_0_Vald':
    #    species_names.add(('Fe_0_Vald', 'Fe'))
    #if species_label == 'Mg_0_Kurucz':
    #    species_names.add(('Mg_0_Kurucz', 'Mg'))
    #if species_label == 'Ni_0_Kurucz':
    #    species_names.add(('Ni_0_Kurucz', 'Ni'))
    #if species_label == 'Ti_0_Kurucz':
    #    species_names.add(('Ti_0_Kurucz', 'Ti'))
    #if species_label == 'Ti_1_Kurucz':
    #    species_names.add(('Ti_1_Kurucz', 'Ti+'))
    #if species_label == 'Ti_0_VALD':
    #    species_names.add(('Ti_0_VALD', 'Ti'))
    #if species_label == 'Ti1':
    #    species_names.add(('Ti_1_VALD', 'Ti+'))
    #if species_label == 'Ti_1_Kurucz':
    #    species_names.add(('Ti_1_Kurucz', 'Ti+'))
      
multiSpeciesCCF('KELT-20b', 'inverted-transmission-better', species_dict, False, True, True, 'ccf')