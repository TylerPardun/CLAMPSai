#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Tyler Pardun
"""

import numpy as np
from scipy import signal,stats
import scipy.integrate as si

def todectime(h,m,s):
    '''Returns the decimal time from the input hour,minute second integers'''
    roundt=np.round(int(h)+(m/60.) + (s/3600.),decimals=4)
    return(roundt)
    
def integral_ts_compute(temp,time):
    '''This function will compute the integral time scale and e-folding time for a 1D variable which is placed inside the parent function integral_ts.
    INPUTS:
        temp -> 1d array contatining the data of a variable set by the user
        time -> 1d array of time values [decimal hour] that will be converted to seconds
    RETURNS:
        lag_h[zc0] -> int. the integral time scale
        efold -> int. the e-folding time
    '''

    #Replace the nan value with the previous data point
    val=0
    for counter,n in enumerate(temp):
        if ~np.isnan(n):
            val = n
        else:
            temp[counter] = val

    #time series data function call
    data,time = temp,time
    data_orig = data #holder since later data gets detrended

    #grab time information
    time_s=time*3600. #time needed in seconds for this block
    nt = time_s.shape[0] #number of time records
    dt = time_s[1]-time_s[0] #time interval of data
    ft = int(np.ceil(nt/2)*2)
    mt = int(ft/2.)

    # detrend linear
    detr1 = signal.detrend(data)
    # detrend mean
    detr2 = signal.detrend(data,type='constant')

    # compute power spectra
    #on linear detrend data
    Pyy    = np.abs(np.fft.fft(detr1))**2/(2*np.pi)
    freq   = np.fft.fftfreq(nt,d=dt)[:mt]
    power1 = 2.*Pyy[:mt]
    #on mean detrend data
    Pyy2   = np.abs(np.fft.fft(detr2))**2/(2*np.pi)
    power2 = 2.*Pyy2[:mt]

    # normalized auto-correlation
    a = (data - np.mean(data)) / (np.std(data) * len(data))
    b = (data - np.mean(data)) / (np.std(data))
    c = np.correlate(a, b, 'full')
    c = c[nt-1::]

    # lags
    lag = np.arange(0,nt*dt,dt)
    lag_m = lag / 60 #minutes
    lag_h = lag_m / 60 #hours

    # find zero-crossing in auto correlation then compute integaral time scale
    # the integral time scale is an estimation of the characteristic time scale
    # it is estimated by integrating the autocorrelation from 0-the first zero-crossing
    # it is an estimate because the 'true' calculation intergrates from 0 to infinity
    # The characteristic time scale or intergral time scale gives a rough measure of
    # the longest connection in the turbulent (flucating) behaviour of the field. It
    # can also be thought of as the dureation for which flucations of various frequencies
    # last before getting destroyed (Swamy et al. 1979: Applied Scientific Research 35 (1979) 265-316)
    zc0 = list(map(lambda i: i<0 , c)).index(True)
    its = si.trapz(c[0:zc0],dx=dt)
    its_m = its / 60 #mins
    its_h = its_m / 60 #hours
    f_its = 1 / its

    #Compute the e-folding lag
    eidx = np.where(c<=np.exp(-1))[0][0]
    efold = c[eidx]
    
    return lag_h[zc0],efold
    
def integral_ts(data_arr,time,hght):
    '''This function will compute the integral time scale as a function of height for a specified variable of the user's choice.
    INPUTS:
        data_arr -> 2d or 1d array consisting of the variable specified by the user [time,height]
        time -> 1d time array in hours [will be converted to seconds]
        hght -> 1d height array in meters or kilometers [choice set by user]
    RETURNS:
        lag_data -> 1d array of integral time scale as a function of height
        efolding_time -> 1d array of the e-folding time as a function of height
    '''
    
    #Get the number of dimensions of the data
    dim = len(data_arr.shape)
    
    #For 1-dimensional variables, this is all we need
    if dim == 1:
        lag_time,efolding = integral_ts_compute(data_arr,time)
        return lag_time,efolding
    
    #For two dimensional variables, loop through each height level
    elif dim == 2:
        #Compute and store the lag times with height according to the bin
        lag_data,efolding_time = np.zeros((hght.size))*np.nan,np.zeros((hght.size))*np.nan
        for count,z_loop in enumerate(hght):
            lag_data[count],efolding_time[count] = integral_ts_compute(data_arr[:,count],time)
        
        return lag_data,efolding_time
    
    else:
        print('Higher-order dimensions are not supported. Please make the data 1 or 2-dimensions to continue.')
        return
    

        
