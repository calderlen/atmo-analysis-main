#this script converts a PEPSI fits file to the format understandable by molecfit

from astropy.io import fits
import numpy as np
from glob import glob
import os
import sys
from atmo_utilities import ccf
import matplotlib.pyplot as pl

def sexagesimal_to_decimal(string_in, mode):

    temp = string_in.split(':')

    if mode == 'coordinates':
        sign = np.sign(np.float(temp[0]))
        temp[0] = np.abs(np.float(temp[0]))
        return sign * (np.float(temp[0]) + np.float(temp[1]) / 60. + np.float(temp[2]) / 3600.)
    if mode == 'time':
        return np.float(temp[0]) * 3600. + np.float(temp[1]) * 60. + np.float(temp[2])

def convert_fits(inname, velocity_offset):

    hdu=fits.open(inname)

    #add necessary header keywords

    try:
        hdu[0].header['ESO TEL AMBI RHUM'] = hdu[0].header['LBTH']
    except KeyError:
        hdu[0].header['ESO TEL AMBI RHUM'] = hdu[0].header['CHAHUMID'] #this isn't exactly right because this is the humidity in the spectrograph chamber, but AFAICT there is no exterior humidity measurement in old data
    hdu[0].header['ESO TEL AMBI PRES START'] = hdu[0].header['CHAPRESS'] #hPa = mbar
    try:
        hdu[0].header['ESO TEL AMBI TEMP'] = hdu[0].header['LBTT'] #ditto for temperature
        hdu[0].header['ESO TEL TH M1 TEMP'] = hdu[0].header['LBTT']
    except KeyError:
        hdu[0].header['ESO TEL AMBI TEMP'] = hdu[0].header['CHATEMPE']
        hdu[0].header['ESO TEL TH M1 TEMP'] = hdu[0].header['CHATEMPE']

    hdu[0].header['ESO TEL GEOLAT'] = sexagesimal_to_decimal(hdu[0].header['LATITUDE'], 'coordinates')
    hdu[0].header['ESO TEL GEOLON'] = sexagesimal_to_decimal(hdu[0].header['LONGITUD'], 'coordinates')*(-15.)
    hdu[0].header['ESO TEL GEOELEV'] = 3221.

    hdu[0].header['ESO INS SLIT1 WID'] = 1.5

    try:
        hdu[0].header['ESO TEL ALT'] = sexagesimal_to_decimal(hdu[0].header['TELAL'], 'coordinates')
        hdu[0].header['RA'] = 15. * sexagesimal_to_decimal(hdu[0].header['RA'], 'coordinates')
        hdu[0].header['DEC'] = sexagesimal_to_decimal(hdu[0].header['DEC'], 'coordinates')
        #second from midnight, for some godforsaken reason
        hdu[0].header['UTC'] = sexagesimal_to_decimal(hdu[0].header['UT-OBS'], 'time')
    except KeyError:
        hdu[0].header['ESO TEL ALT'] = sexagesimal_to_decimal(hdu[0].header['LBTAL'], 'coordinates')
        hdu[0].header['RA'] = 15. * sexagesimal_to_decimal(hdu[0].header['LBTRA'], 'coordinates')
        hdu[0].header['DEC'] = sexagesimal_to_decimal(hdu[0].header['LBTDE'], 'coordinates')
        #second from midnight, for some godforsaken reason
        hdu[0].header['UTC'] = sexagesimal_to_decimal(hdu[0].header['TIME-OBS'], 'time')

    
    
    hdu[0].header['LST'] = sexagesimal_to_decimal(hdu[0].header['LST'], 'time')

    hdu[0].header['MJD-OBS'] = hdu[0].header['JD-OBS'] - 2400000.5

    hdu[0].header['CCFC0'] = 0.0
    hdu[0].header['CCFC1'] = 0.0
    
    #rename data columns
    hdu[1].data.columns['Arg'].name = 'lambda'
    hdu[1].data.columns['Fun'].name = 'flux'
    hdu[1].data.columns['Var'].name = 'error'
    hdu[1].data['error'] = np.sqrt(hdu[1].data['error'])

    #convert wavelengths from Angstroms to microns
    hdu[1].data['lambda']/=10000.

    #get back in telluric rest frame
    ckms = 2.9979e5
    #if np.abs(hdu[0].header['RADVEL']) > 100.: hdu[0].header['RADVEL'] = 0.0
    try:
        doppler_shift = 1.0 / (1.0 - ((hdu[0].header['RADVEL'] + hdu[0].header['OBSVEL'])/1000. + velocity_offset) * (-1.0) / ckms)
    except KeyError:
        doppler_shift = 1.0 / (1.0 - ((hdu[0].header['RADVEL'] + hdu[0].header['SSBVEL'])/1000. + velocity_offset) * (-1.0) / ckms)
    hdu[1].data['lambda'] *= doppler_shift

    #write out the data

    hdu.writeto(inname+'.fits', overwrite = True)
    hdu.close()

def get_velocity_offset(inname):

    convert_fits(inname, 0.0)
    sof_file = 'testing.sof'
    f = open(sof_file, 'w')
    f.write(inname + '.fits SCIENCE \n')
    f.write('/home/johnson.7240/Documents/system/molecfit42/molecfit-kit-4.2/ATM_PARAMETERS.fits ATM_PARAMETERS \n')
    f.write('/home/johnson.7240/Documents/system/molecfit42/molecfit-kit-4.2/MODEL_MOLECULES.fits MODEL_MOLECULES \n')
    f.close()

    os.system('esorex --recipe-config=Default_model.rc molecfit_model ' + sof_file)

    f = open(sof_file, 'a')
    f.write('/home/johnson.7240/Documents/system/molecfit42/molecfit-kit-4.2/BEST_FIT_PARAMETERS.fits BEST_FIT_PARAMETERS \n')
    f.close()

    os.system('esorex --recipe-config=Default_model_2.rc molecfit_calctrans ' + sof_file)

    hdu1 = fits.open('BEST_FIT_MODEL.fits')
    hdu2 = fits.open('LBLRTM_RESULTS.fits')

    drv, cross_cor, sigma_cross_cor = ccf(hdu1[1].data['lambda'], hdu1[1].data['flux'], np.ones_like(hdu1[1].data['weight']), hdu2[1].data['lambda'], hdu2[1].data['flux'], -1000., 1000., 0.5)

    fit = np.polyfit(drv, cross_cor, 1)

    cross_cor_flat = cross_cor / np.polyval(fit, drv)

    v_best = drv[np.argmax(cross_cor_flat)]

    pl.plot(drv, cross_cor_flat)
    pl.plot([v_best, v_best], [np.min(cross_cor_flat), np.max(cross_cor_flat)], 'r')
    pl.show()

    #refine
    drv, cross_cor, sigma_cross_cor = ccf(hdu1[1].data['lambda'], hdu1[1].data['flux'], np.ones_like(hdu1[1].data['weight']), hdu2[1].data['lambda'], hdu2[1].data['flux'], v_best-20., v_best+20., 0.1)

    v_best = drv[np.argmax(cross_cor)]

    print('The best velocity offset is ',v_best,' km/s')

    pl.plot(drv, cross_cor)
    pl.plot([v_best, v_best], [np.min(cross_cor), np.max(cross_cor)], 'r')
    pl.show()

    return v_best


def run_molecfit(planet_name, observation_epoch):
    
    arm_file = 'pepsir' #assuming will never need to telluric correct the blue arm
    path = '/home/johnson.7240/Documents/astro/atmos/data/'
    data_location = path + observation_epoch + '_' + planet_name + '/*' + arm_file + '*.dxt.nor'
    spectra_files = glob(data_location)
    #pwd = os.getcwd()
    #path = pwd + '/data/'
    sof_file = planet_name + '.' + observation_epoch + '.sof'

    for spectrum_file in spectra_files:
        if spectrum_file == spectra_files[0]:
            velocity_offset = get_velocity_offset(spectrum_file)
        convert_fits(spectrum_file, velocity_offset)
        file_name = spectrum_file.split('/')[-1]+'.fits'
        
        f = open(sof_file, 'w')
        f.write(spectrum_file + '.fits SCIENCE \n')
        f.write('/home/johnson.7240/Documents/system/molecfit42/molecfit-kit-4.2/ATM_PARAMETERS.fits ATM_PARAMETERS \n')
        f.write('/home/johnson.7240/Documents/system/molecfit42/molecfit-kit-4.2/MODEL_MOLECULES.fits MODEL_MOLECULES \n')
        f.close()

        os.system('esorex --recipe-config=Default_model.rc molecfit_model ' + sof_file)

        

        f = open(sof_file, 'a')
        f.write('/home/johnson.7240/Documents/system/molecfit42/molecfit-kit-4.2/BEST_FIT_PARAMETERS.fits BEST_FIT_PARAMETERS \n')
        f.close()

        os.system('esorex --recipe-config=Default_model_2.rc molecfit_calctrans ' + sof_file)

        f = open(sof_file, 'a')
        f.write('/home/johnson.7240/Documents/system/molecfit42/molecfit-kit-4.2/TELLURIC_CORR.fits TELLURIC_CORR \n')
        f.close()

        os.system('esorex --recipe-config=Default_model_3.rc molecfit_correct '+ sof_file)

        os.system('cp SCIENCE_TELLURIC_CORR_' + file_name + ' /home/johnson.7240/Documents/astro/atmos/data/' + observation_epoch + '_' + planet_name + '/molecfit_weak/')

        print('The best velocity offset was ',velocity_offset,' km/s')

        
