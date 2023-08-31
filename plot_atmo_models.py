import numpy as np
import matplotlib.pyplot as pl
from astropy.io import fits

planet_name = 'KELT-20b'
species_name = 'TiO_48_Exomol_McKemmish'
temperature_profile = 'inverted-test'

vmrs = np.array([1e-6, 1e-7, 1e-8, 1e-9, 1e-10])

pl.figure(figsize=(6,10))

n=0
for vmr in vmrs:
    filein = 'templates/' + planet_name + '.' + species_name + '.' + str(vmr) + '.' + temperature_profile + '.fits'
    hdu = fits.open(filein)

    template_wave = hdu[1].data['wave']
    template_flux = hdu[1].data['flux']

    hdu.close()

    pl.subplot(len(vmrs), 1, n+1)

    pl.plot(template_wave, template_flux, color='lightgray')

    iii = (template_wave >= 4800.) & (template_wave <= 5441.)
    v = (template_wave >= 6278.) & (template_wave <= 7419.)

    pl.plot(template_wave[iii], template_flux[iii], color='blue')
    pl.plot(template_wave[v], template_flux[v], color='red')

    if ('VO' in species_name or 'TiO' in species_name) and vmr > 5e-8:
        pl.text(4600.,5e-7, 'VMR='+str(vmr))
    else:
        pl.text(4600.,1e-3, 'VMR='+str(vmr))

    pl.ylabel('$F_P/F_*$')
    pl.yscale('log')
    if 'FeH' in species_name: pl.ylim([1e-7, 0.004])
    if 'VO' in species_name: pl.ylim([1e-7, 0.007])
    if 'VO' in species_name: pl.ylim([1e-7, 0.008])
    pl.xlim([4500,7500])

    n+=1

pl.xlabel('wavelength (Angstroms)')

pl.tight_layout()

pl.savefig('plots/'+planet_name + '.' + species_name + '.spectra-comparison.pdf', format='pdf')

pl.clf()
