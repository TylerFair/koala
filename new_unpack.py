import os,sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pandas as pd 
import jax 
import jax.numpy as jnp 

def get_whitelightlc(fluxcube,fluxcube_err):
    whitelightflux = np.nanmean(fluxcube,axis=1) 
    return whitelightflux

def get_rms_colour(fluxcube,fluxcube_err,wavelength):
    #meanflux = np.nanmedian(fluxcube,axis=0)
    #fluxcube /= meanflux

    wavebin_rms = []
    i = 0
    binsize = 10
    print(fluxcube.shape)
    while i < len(fluxcube[0]):
        #flux_i = np.nanmedian(fluxcube[:,i:i+binsize],axis=1)
        flux_i = get_whitelightlc(fluxcube[:,i:i+binsize],fluxcube_err[:,i:i+binsize])
        flux_i/=np.nanmedian(flux_i)

        #plt.plot(flux_i, '.')
        #plt.show()
        mad = np.nanmedian(abs(np.diff(flux_i)))
        wave_i = np.nanmedian(wavelength[i:i+binsize])
        wavebin_rms.append([wave_i,mad])
        i += binsize
        print(i, mad)
    wavebin_rms = np.array(wavebin_rms)

    return wavebin_rms

def join_whitelight(whitelightlist):

    whitelightnew = list(whitelightlist[0])
    for i in np.arange(1,len(whitelightlist)):
        whitelightnew += list(whitelightlist[i])

    return np.array(whitelightnew)/np.nanmedian(whitelightnew)
                       
def unpack_nirspec_eureka(file_pattern, nrs, planet_str, output_dir, bins_wanted, prior_t0, prior_duration):
    x1dfitslist = np.sort(glob.glob(file_pattern))
    
    fluxcubeall = []
    fluxcubeall_err = []
    wavelength = None
    t = []
    aperturekey = 'FLUX'
    for i in range(len(x1dfitslist)):
        x1dfits = x1dfitslist[i]
        print(x1dfits)
        x1dfits = fits.open(x1dfits)
        try:
            bjd = x1dfits[1].data['int_mid_BJD_TDB'][1:]
        except IndexError:
            bjd = np.arange(len(x1dfits)-3)
        wave = x1dfits[2].data['WAVELENGTH']
        wave_mask = wave > 2.86
        wave = wave[wave_mask]
        
        fluxcube = []
        for j in np.arange(3, len(x1dfits)):
            fluxcube.append(x1dfits[j].data[aperturekey])
        
        fluxcube = np.array(fluxcube)
        
        # Apply wavelength cut to flux data
        fluxcube = fluxcube[:, wave_mask]

        fluxcubeall += list(fluxcube)
        wavelength = wave
        t.append(bjd)

    t = np.concatenate(t)
    fluxcube = np.array(fluxcubeall)
    
    base = np.concatenate([np.arange(100), np.arange(100)-100]).astype(int) 
    median_flux = np.nanmedian(fluxcube[base], axis=0)
    fluxcube = fluxcube / median_flux

    wavelengths = jnp.array(wavelength)  
    indiv_y = jnp.array(fluxcube)  

    return wavelengths, t, indiv_y, None


def unpack_niriss_exoted(infile, order, trim_start, trim_end, wl_min_o1=None, wl_max_o1=None, wl_min_o2=None, wl_max_o2=None):    

    bjd = fits.getdata(infile, 9)
    wave = fits.getdata(infile, 1 + 4 * (order - 1))
    wave_err = fits.getdata(infile, 2 + 4 * (order - 1))
    fluxcube = fits.getdata(infile, 3 + 4 * (order - 1))
    fluxcube_err = fits.getdata(infile, 4 + 4 * (order -1))
    wave = wave[5:-5]
    wave_err = wave_err[5:-5]
    fluxcube = fluxcube[:, 5:-5]
    fluxcube_err = fluxcube_err[:, 5:-5]
    
    start = 0 if (trim_start is None) else int(trim_start)
    stop  = None if (trim_end in (None, 0)) else -int(trim_end)

    fluxcube     = fluxcube[start:stop, :]
    fluxcube_err = fluxcube_err[start:stop, :]
    bjd            = bjd[start:stop]   # keep time aligned!


    if order == 2:
        ii = np.where((wave >= 0.6) & (wave <= 0.85))[0]
        fluxcube, fluxcube_err = fluxcube[:, ii], fluxcube_err[:,ii]
        wave, wave_err = wave[ii], wave_err[ii]

    if wl_min_o1 is not None and wl_max_o1 is not None and order == 1:
        ii = np.where((wave >= wl_min_o1) & (wave <= wl_max_o1))[0]
        fluxcube, fluxcube_err = fluxcube[:, ii], fluxcube_err[:,ii]
        wave, wave_err = wave[ii], wave_err[ii]
    
    if wl_min_o2 is not None and wl_max_o2 is not None and order == 2:
        ii = np.where((wave >= wl_min_o2) & (wave <= wl_max_o2))[0]
        fluxcube, fluxcube_err = fluxcube[:, ii], fluxcube_err[:,ii]
        wave, wave_err = wave[ii], wave_err[ii]

    wavelength = wave
    wavelength_err = wave_err
    t = np.array(bjd)
    fluxcube = np.array(fluxcube)
    fluxcube_err = np.array(fluxcube_err)

    return wavelength,wavelength_err, t, fluxcube, fluxcube_err

def unpack_nirspec_exoted(infile, instrument, trim_start, trim_end, wl_min=None, wl_max=None):    
    print(wl_min, wl_max)
    bjd = fits.getdata(infile, 5)
    wave = fits.getdata(infile, 1)
    wave_err = fits.getdata(infile, 2)
    fluxcube = fits.getdata(infile, 3)
    fluxcube_err = fits.getdata(infile, 4)
    wave = wave[5:-5]
    wave_err = wave_err[5:-5]
    fluxcube = fluxcube[:, 5:-5]
    fluxcube_err = fluxcube_err[:, 5:-5]
    print(bjd)
    print(wave)
    
    start = 0 if (trim_start is None) else int(trim_start)
    stop  = None if (trim_end in (None, 0)) else -int(trim_end)

    fluxcube     = fluxcube[start:stop, :]
    fluxcube_err = fluxcube_err[start:stop, :]
    bjd            = bjd[start:stop]   # keep time aligned!


    if instrument == 'NIRSPEC/G395M' or instrument == 'NIRSPEC/G395H':
        ii = np.where((wave >= 2.9) & (wave <= 5.0))[0]
        fluxcube, fluxcube_err = fluxcube[:, ii], fluxcube_err[:,ii]
        wave, wave_err = wave[ii], wave_err[ii]
    if instrument == 'NIRSPEC/PRISM':
        ii = np.where((wave >= 0.6) & (wave <= 5.0))[0]
        fluxcube, fluxcube_err = fluxcube[:,ii], fluxcube_err[:,ii]
        wave, wave_err = wave[ii], wave_err[ii]
    if instrument == 'NIRSPEC/G140H':
        ii = np.where((wave >= 1.0) & (wave <= 1.8))[0]
        fluxcube, fluxcube_err = fluxcube[:,ii], fluxcube_err[:,ii]
        wave, wave_err = wave[ii], wave_err[ii]

    if wl_min is not None and wl_max is not None:
        ii = np.where((wave >= wl_min) & (wave <= wl_max))[0]
        fluxcube, fluxcube_err = fluxcube[:, ii], fluxcube_err[:,ii]
        wave, wave_err = wave[ii], wave_err[ii]

    wavelength = wave
    wavelength_err = wave_err
    t = np.array(bjd)
    fluxcube = np.array(fluxcube)
    fluxcube_err = np.array(fluxcube_err)
    return wavelength, wavelength_err,  t, fluxcube, fluxcube_err

def unpack_miri_exoted(infile, trim_start, trim_end, wl_min=None, wl_max=None):

    bjd = fits.getdata(infile, 5)
    wave = fits.getdata(infile, 1)
    wave_err = fits.getdata(infile, 2)
    fluxcube = fits.getdata(infile, 3)
    fluxcube_err = fits.getdata(infile, 4)
    wave = wave[5:-5]
    wave_err = wave_err[5:-5]
    fluxcube = fluxcube[:, 5:-5]
    fluxcube_err = fluxcube_err[:, 5:-5]


    start = 0 if (trim_start is None) else int(trim_start)
    stop  = None if (trim_end in (None, 0)) else -int(trim_end)

    fluxcube     = fluxcube[start:stop, :]
    fluxcube_err = fluxcube_err[start:stop, :]
    bjd            = bjd[start:stop]   # keep time aligned!


    ii = np.where((wave > 5) & (wave <= 12))[0]
    fluxcube, fluxcube_err = fluxcube[:, ii], fluxcube_err[:,ii]
    wave, wave_err = wave[ii], wave_err[ii]
    ii = np.argsort(wave)
    wave, wave_err = wave[ii], wave_err[ii]
    fluxcube, fluxcube_err = fluxcube[:,ii], fluxcube_err[:,ii]

    if wl_min is not None and wl_max is not None:
        ii = np.where((wave >= wl_min) & (wave <= wl_max))[0]
        fluxcube, fluxcube_err = fluxcube[:, ii], fluxcube_err[:,ii]
        wave, wave_err = wave[ii], wave_err[ii]
    
    wavelength = wave
    wavelength_err = wave_err
    t = np.array(bjd)
    fluxcube = np.array(fluxcube)
    fluxcube_err = np.array(fluxcube_err)
    return wavelength, wavelength_err, t, fluxcube, fluxcube_err

