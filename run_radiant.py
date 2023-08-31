from radiant import *

import numpy as np
import petitRADTRANS.nat_cst as nc
from petitRADTRANS import Radtrans
from astropy import units as u
from astropy.io import ascii

from create_model import create_model, instantiate_radtrans
from atmo_utilities import *
#from run_all_ccfs import process_data

import emcee
import argparse
import os


#import matplotlib.pyplot as pl

os.environ["OMP_NUM_THREADS"] = "1"

from multiprocessing import Pool





import time as timemod
thewholestart=timemod.time()

#the actual main part of the script will go here
parser=argparse.ArgumentParser()
parser.add_argument("infile", type=str, help="name of the input file")
parser.add_argument("--startnew", action="store_true", help="start a new chain regardless of whether the given output files already exist")
parser.add_argument("--finalonly", action="store_true", help="only produce final data products without any other data processing")
args=parser.parse_args()
infile=args.infile

struc1=inreader(infile)

index=np.array(struc1['index'])
invals=np.array(struc1['invals'])

if 'mockup' not in  index:
    struc1['mockup'] = False

#Get the parameters for the chains
nwalkers=np.int64(struc1['nwalkers'])
nsteps=np.int64(struc1['nsteps'])
nthreads=np.int64(struc1['nthreads'])
nburnin=np.int64(struc1['nburnin'])

#get the general input and output filenames
chainfile=struc1['chainfile']
probfile=struc1['probfile']
accpfile=struc1['accpfile']


#translate the input parameters into the structures required for the MCMC

allpardex = index[[i for i, s in enumerate(index) if 'par' in s]]

inpos, inposindex, perturbs = [], [], []
instruc, parstruc, priorstruc = {}, {}, {}
instruc['all_species'] = []

for par0 in allpardex:
    temp = par0.split('-')
    par = temp[1]
    if str2bool(struc1['fit-'+par]):
        inpos.append(struc1['par-'+par]), perturbs.append(struc1['scl-'+par])
        if not 'vmr' in par:
            tag = par
        else:
            tag = struc1['nam-'+par]
            instruc['all_species'].append(struc1['nam-'+par])
        inposindex.append(tag)
        if 'pri-'+par in struc1:
            priorstruc[tag], instruc['val-'+tag] = float(struc1['pri-'+par]), float(struc1['par-'+par])
            
    else:
        if not 'vmr' in par:
            instruc[par] = float(struc1['par-'+par])
        else:
            instruc[struc1['nam-'+par]] = float(struc1['par-'+par])
            instruc['all_species'].append(struc1['nam-'+par])






ndim=len(inpos)
inpos, perturbs = np.array(inpos,dtype=float), np.array(perturbs,dtype=float)



#get the data

instruc['instrument'], instruc['planet_name'] = struc1['instrument'], struc1['planet_name']
instruc['spectrum_type'] = struc1['spectrum_type']
observation_epochs = invals[[i for i, s in enumerate(index) if 'dataset' in s]]
if struc1['instrument'] == 'PEPSI':
    arms = ['red', 'blue']
else:
    arms = ['combined']

#instantiate the petitRADTRANS model
lambda_low, lambda_high = get_wavelength_range(instruc['instrument'])
instruc['pressures'] = np.logspace(-8, 2, 100)

if not args.finalonly: atmosphere = instantiate_radtrans(instruc['all_species'], lambda_low, lambda_high, instruc['pressures'], downsample_factor = 6)

if not args.finalonly:
    if not struc1['mockup']:
        datastruc = process_data(observation_epochs, arms, instruc['planet_name'])
    else:
        datastruc = make_mocked_data(struc1, index, invals, instruc, atmosphere, instruc['pressures'], lambda_low, lambda_high)
else:
    datastruc = {}

loaded=False

if not args.startnew:
    try:
        chainin=np.load(chainfile)
        chainshape=chainin.shape
        nin=chainshape[1]
        pos=chainin[:,nin-1,:]
        probin=np.load(probfile)
        loaded=True
    except FileNotFoundError:
        print('The input files were not found, starting a new chain.')

if not loaded:
    pos = np.array([inpos + perturbs*np.random.randn(ndim) for i in range(nwalkers)])
    #check for bad starting values
    for i in range (0,nwalkers):
        for j in range (0,ndim):
            while ('alpha' in  inposindex[j] and pos[i,j] <= 0.0) or ('P0' in  inposindex[j] and pos[i,j] <= 0.0) or ('kappa' in  inposindex[j] and pos[i,j] <= 0.0) or ('gamma' in  inposindex[j] and pos[i,j] <= 0.0):
                pos[i,j]=inpos[j]+perturbs[j]*np.random.randn(1)


datastruc['count']=0

themcmcstart = timemod.time()

if not args.finalonly:
    with Pool(processes=nthreads) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(instruc, datastruc, inpos, inposindex, priorstruc, atmosphere), pool=pool)

        sampler.run_mcmc(pos, nsteps, progress=True)

    chain=np.array(sampler.chain)
    if loaded == False:
        np.save(chainfile, np.array(sampler.chain))
        np.save(probfile, np.array(sampler.lnprobability))
        np.save(accpfile,np.array(sampler.acceptance_fraction))
        
    if loaded == True:
        prob=np.array(sampler.lnprobability)
        accp=np.array(sampler.acceptance_fraction)
        chain2=np.append(chainin,chain,axis=1)
        prob2=np.append(probin,prob,axis=1)
        accpin=np.load(accpfile)
        accpin*=nin #to get total number of acceptances
        accp*=nsteps #ditto
        accp2=accpin+accp
        accp2=accp2/(nin+nsteps) #to get back to fraction
        np.save(chainfile, chain2)
        np.save(probfile, prob2)
        np.save(accpfile, accp2)
        chain=chain2

else:
    chain = chainin



samples = chain[:,nburnin:,:].reshape((-1,ndim))


print_results(samples, inposindex, ndim, struc1, allpardex, index)
plot_pt_profile(samples, inposindex, struc1, instruc['pressures'], instruc['planet_name'])
make_corner(samples, inposindex, ndim)



temp=chain.shape                         
thewholeend=timemod.time()
print('Radiant is now done running! It took a total time of ',thewholeend-thewholestart,' seconds, for ',nsteps,' steps and a total length of ',temp[1],' steps')
print('The time to run the MCMC was ',thewholeend-themcmcstart,' seconds')
print('The order of the parameters was ',inposindex)
