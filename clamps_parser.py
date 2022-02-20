#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 07:30:18 2019

@author: tylerpardun
"""
#Supress Warnings
import warnings
warnings.filterwarnings("ignore")

from utils import todectime

import numpy as np
from netCDF4 import Dataset
from siphon.catalog import TDSCatalog
from datetime import datetime
from glob import glob
import os
from tqdm import tqdm
import pickle

import sharppy.sharptab.profile as profile
import sharppy.sharptab.params as params

import metpy.calc as mpcalc
from metpy.units import units




def read_aeri(aeri_file):
    '''This function will read AERI files and return the variables through the height maximum set
    INPUTS:
        aeri_file -> file location
    RETURNS:
        variables needed to be extracted
    '''
    d0 = Dataset(aeri_file)

    time = np.array([datetime.utcfromtimestamp(d) for d in d0['base_time'][0]+d0['time_offset'][:]])
    height = d0['height'][:]*1000 #meters
    lat,lon = float(d0['lat'][0]),float(d0['lon'][0])

    #hPa, C, C, g/kg
    pres,temp,dew,qv = d0['pressure'][:],d0['temperature'][:],d0['dewpt'][:],d0['waterVapor'][:]
    #K, K, m
    theta,thetae,cbh = d0['theta'][:,],d0['thetae'][:],d0['cbh'][:]*1000 #meters
    
    d0.close()
    
    return time,height,lat,lon,pres,temp,dew,qv,theta,thetae,cbh

def read_lidar(lidar_file,dt,height_max):
    '''This function will read LiDAR files and return the variables through the height maximum set
    INPUTS:
        lidar_file -> file location
        dt -> datetime object, the time of passage
        height_max -> int. of the maximum height in meters
    RETURNS:
        variables needed to be extracted
    '''

    d = Dataset(lidar_file)
    try:
        hidx = np.where(d['height'][:]>=(height_max/1000))[0][0]
    except IndexError:
        hidx = d['height'][:].size #This means the max height is less than the specified height

    #Get the SNR threshold
    if dt.year == 2016:
        snr = d['intensity'][:]
        filt = -29
    elif dt.year == 2017:
        snr = d['s2n'][:]
        filt = -23
    elif dt.year == 2019:
        if int(dt.strftime('%Y%m%d')) == 20190309:
            snr= d['intensity'][:]
        else:
            snr = d['intensity'][:]
        filt = 1.005
    else:
        print('No SNR threshold has been set for CLAMPS LiDAR files in the year {}. Please find it before continuing.'.format(dt.year))


    time = np.array([datetime.utcfromtimestamp(x) for x in d['base_time'][0]+d['time_offset'][:]])
    height = d['height'][:hidx]*1000 #meters

    #m/s, deg, m/s
    wspd,wdir,w = d['wspd'][:,:hidx],d['wdir'][:,:hidx],d['w'][:,:hidx]

    #Filter by SNR
    ifilt = np.where(snr[:,:hidx]<=filt)
    wspd[ifilt],wdir[ifilt],w[ifilt] = np.nan,np.nan,np.nan

    d.close()

    return time,height,wspd,wdir,w

def read_stare(stare_file,dt,height_max):
    '''This function will read vertical stare files and return the variables through the height maximum set
    INPUTS:
        stare_file -> file location
        dt -> datetime object, the time of passage
        height_max -> int. of the maximum height in meters
    RETURNS:
        variables needed to be extracted
    '''

    d = Dataset(stare_file)

    hidx = np.where(d['height'][:]>=(height_max/1000))[0][0]

    #Get the SNR threshold
    if dt.year == 2016:
        snr = d['cnr'][:]
        filt = -29
    elif dt.year == 2017:
        snr = d['intensity'][:]
        filt = 1.02
    elif dt.year == 2019:
        snr = d['snr'][:]
        filt = 0.01
    else:
        print('No SNR threshold has been set for vertical stare files in the year {}. Please find it before continuing.'.format(dt.year))

    time = np.array([datetime.utcfromtimestamp(x) for x in d['base_time'][0]+d['time_offset'][:]])
    height = d['height'][:hidx]*1000 #meters

    #m/s, deg, m/s
    w,cbh = d['velocity'][:,:hidx],d['cbh'][:]

    #Filter by SNR
    ifilt = np.where(snr[:,:hidx]<=filt)
    w[ifilt],cbh[ifilt[0]] = np.nan,np.nan

    d.close()

    return time,height,w,cbh

def read_surface(file):
    '''This function will read in the surface data and return variables of interest.
    INPUTS:
        file -> location of the surface data file
    RETURNS:
        1d arrays of the variables wanted
    '''
    #Read in the data file
    d = Dataset(file)
    time = np.array([datetime.utcfromtimestamp(d) for d in d['base_time'][0]+d['time_offset'][:]])
    #C, hPa, %
    temp,pres,rh = d['sfc_temp'][:],d['sfc_pres'][:],d['sfc_rh'][:]
    #deg #m/s
    wdir,wspd = d['sfc_wspd'][:],d['sfc_wdir'][:]

    #Compute the other variables
    dew = dew_calc(rh,temp)
    thetae = thetaecalc(pres,temp,dew)
    theta = theta_calc(pres,temp)
    qv = qv_calc(pres,temp,rh)
    
    return time,pres,temp,dew,qv,theta,thetae,wspd,wdir


#Thermodynamic functions----------------------------------------------------------
def dew_calc(rh,temp):
    '''This function will compute the dewpoint.
    INPUTS:
        rh -> nd array of relative humidity (automatically put into correct units [decimal])
        temp -> nd array of temperature (automatically put into correct units [K])
    RETURNS:
        Dewpoint temperature [C]
    '''
    #Make sure temperature is in Kelvin
    temp[temp<0] = np.nan
    if temp[~np.isnan(temp)][0]<100:
        dtemp = temp+273.15
    else:
        dtemp = temp

    #Make sure RH is in decimal
    rh[rh<0] = np.nan
    if rh[~np.isnan(rh)][0]<1:
        drh = rh
    else:
        drh = rh/100
    
    Lv = 2501000 #J/kg
    Rv = 461.5 #J/kg K
    e0 = 6.112 #hPa
    T0 = 273.15 #K
    
    es = e0*(np.e**((Lv/Rv)*((1/T0)-(1/dtemp)))) #saturation vapor pressure
    e = drh*es #vapor pressure
    return (((1/T0) - ((Rv/Lv)*np.log(e/e0)))**-1)-T0 #C

def qv_calc(pres,temp,rh):
    '''This function will compute the water vapor mixing ratio.
    INPUTS:
        pres -> nd array of pressure (automatically put into correct units [hPa])
        temp -> nd array of temperature (automatically put into correct units [K])
        rh -> nd array of relative humidity (automatically put into the correct untis [decimal])
    RETURNS:
        Water vapor mixing ratio [g/kg]
    '''
    
    #Make sure pressure is in hPa (most likely from Pa)
    pres[pres<0] = np.nan
    if pres[~np.isnan(pres)][0]>1500:
        dpres = pres/100
    else:
        dpres = pres
        
    #Make sure temperature is in Kelvin
    temp[temp<0] = np.nan
    if temp[~np.isnan(temp)][0]<100:
        dtemp = temp+273.15
    else:
        dtemp = temp
        
    #Make sure RH is in decimal
    rh[rh<0] = np.nan
    if rh[~np.isnan(rh)][0]<1:
        drh = rh
    else:
        drh = rh/100
        
    Lv = 2501000 #J/kg
    Rd = 287 #J/kg K
    Rv = 461.5 #J/kg K
    e0 = 6.112 #hPa
    T0 = 273.15 #K
    
    epsilon = Rd/Rv
    es = e0*(np.e**((Lv/Rv)*((1/T0)-(1/dtemp)))) #saturation vapor pressure
    e = drh*es #vapor pressure
    
    return  ((epsilon*e) / (dpres-e))*1000 #g/kg


def theta_calc(pres,temp):
    '''This function will compute the dewpoint.
    INPUTS: )
        temp -> nd array of temperature (automatically put into correct units [K])
    RETURNS:
        Potential temperature [C]
    '''
    
    #Make sure pressure is in hPa (most likely from Pa)
    pres[pres<0] = np.nan
    if pres[~np.isnan(pres)][0]>1500:
        dpres = pres/100
    else:
        dpres = pres
        
    #Make sure temperature is in Kelvin
    temp[temp<0] = np.nan
    if temp[~np.isnan(temp)][0]<100:
        dtemp = temp+273.15
    else:
        dtemp = temp
    
    Rd = 287 #J/kg K
    cp = 1004 #J/kg
    return  dtemp*((1000/dpres)**(Rd/cp)) #K

def thetaecalc(pres,temp,dew):
    '''This function will compute theta-e using MetPy
    INPUTS:
        pres -> nd array of pressure (automatically put into correct units)
        temp -> nd array of temperature (automatically put into correct units)
        dew -> nd array of dewpoint temperature (automatically put into correct untis)
    RETURNS:
        thetae [K]
    '''
    #Make sure pressure is in hPa (most likely from Pa)
    pres[pres<0] = np.nan
    if pres[~np.isnan(pres)][0]>1500:
        dpres = pres/100
    else:
        dpres = pres
    
    #Make sure temperature is in C
    temp[temp<0] = np.nan
    if temp[~np.isnan(temp)][0]<100:
        dtemp = temp
    else:
        dtemp = temp-273.15

    #Make sure dew is in C
    dew[dew<0] = np.nan
    if dew[~np.isnan(dew)][0]<100:
        drh = dew
    else:
        drh = dew-273.15
    
    #Put into arrays
    pres = np.array(dpres)*units.hPa
    temp = np.array(dtemp)*units.degC
    dew = np.array(dew)*units.degC
    
    return np.array(mpcalc.equivalent_potential_temperature(pres,temp,dew)/units.kelvin) #K

def clamps_parser(file_dir,t0,p0,height_max=3000,cape_cin=True):
    '''This function will parse through CLAMPS data and return a list of dictionaries filled with variables worth looking at.
    INPUTS:
    file_dir -> str. location of where the CLAMPS data is located. This should lead to the directory where all CLAMPS
                data is located. Within this directory, there should be sub-directories labeled by year and then another by CLAMPS.
                The code is easy to follow to make sure the sub-directories are lined up correctly.
    t0 -> datetime object corresponding to the beginning of the observational period.
    p0 -> datetime object corresponding to the time of passage (essentially the end of the period).
    height_max -> int. corresponding to the maximum height [meters] of the data to be returned (default is 3km).
    cape_cin -> bool. users choice whether to compute CAPE and CIN at all levels up to height_max for analysis (takes some time)
    
    RETURNS:
        data_dict -> list of dictionaries corresponding to each case that has been parsed through.
    '''
    
    #Loop through all the cases
    data_dict = []
    for k,dt in enumerate(p0):

        print('Parsing through {}'.format(dt.strftime('%d %B %Y')))
        ####################################################################
        #Import AERI retrievals
        ####################################################################
        aeri0 = sorted(glob(file_dir+'{}/thermo/*{}*'.format(dt.year,t0[k].strftime('%Y%m%d'))))[0]
        aeri1 = sorted(glob(file_dir+'{}/thermo/*{}*'.format(dt.year,dt.strftime('%Y%m%d'))))[0]

        #Get the contents of each file -> lat,lon should not change
        time0,height0,lat0,lon0,pres0,temp0,dew0,qv0,theta0,thetae0,cbh0 = read_aeri(aeri0)
        time1,height1,lat1,lon1,pres1,temp1,dew1,qv1,theta1,thetae1,cbh1 = read_aeri(aeri1)

        if (lat0 != lat1) or (lon0!=lon1):
            print('ERROR: Files from {} to {} are not at the same location!'.format(t0[k].strftime('%Y%m%d'),dt.strftime('%Y%m%d')))
        elif np.sum((height0-height1)) != 0:
            print('ERROR: Height levels from {} to {} do not match.'.format(t0[k].strftime('%Y%m%d'),dt.strftime('%Y%m%d')))

        #Concatenate them together
        time,lat,lon = np.concatenate((time0,time1)),lat0,lon0
        theight = height0
        pres,temp,dew,qv = np.concatenate((pres0,pres1)),np.concatenate((temp0,temp1)),np.concatenate((dew0,dew1)),np.concatenate((qv0,qv1))
        theta,thetae,cbh = np.concatenate((theta0,theta1)),np.concatenate((thetae0,thetae1)),np.concatenate((cbh0,cbh1))

        #Compute the relative time scale
        rel_hours = np.array([val.total_seconds()/3600 for val in (time - dt)])
        tidx = np.where((rel_hours>=-8)&(rel_hours<=2))[0]

        #Index the data here
        t_utc,t_rel = time[tidx],rel_hours[tidx]
        t_dec = np.array([todectime(val.hour,val.minute,val.second) for val in t_utc])
        #Make the decimal time continous
        for i,val in enumerate(t_dec):
            if val<10:
                t_dec[i] = val+24
                
        pres,temp,dew,qv = pres[tidx,:],temp[tidx,:],dew[tidx,:],qv[tidx,:]
        theta,thetae,cbh = theta[tidx,:],thetae[tidx,:],cbh[tidx]

        #Replace the surface data with the MetTower data
        sfc_file0 = sorted(glob(file_dir+'{}/surface/*{}*'.format(dt.year,t0[k].strftime('%Y%m%d'))))[0]
        sfc_file1 = sorted(glob(file_dir+'{}/surface/*{}*'.format(dt.year,dt.strftime('%Y%m%d'))))[0]

        #Read in the surface files
        stime0,spres0,stemp0,sdew0,sqv0,stheta0,sthetae0,wspd0,wdir0 = read_surface(sfc_file0)
        stime1,spres1,stemp1,sdew1,sqv1,stheta1,sthetae1,wspd1,wdir1 = read_surface(sfc_file1)

        #Concatenate all the data together
        stime = np.concatenate((stime0,stime1))
        spres,stemp,sdew = np.concatenate((spres0,spres1)),np.concatenate((stemp0,stemp1)),np.concatenate((sdew0,sdew1))
        sqv,stheta,sthetae = np.concatenate((sqv0,sqv1)),np.concatenate((stheta0,stheta1)),np.concatenate((sthetae0,sthetae1))
        swspd,swdir = np.concatenate((wspd0,wspd1)),np.concatenate((wdir0,wdir1))

        #Remove the nans
        nan_idx = np.setdiff1d(np.arange(stime.size),np.where(np.isnan(spres)))
        stime,spres,stemp,sdew = stime[nan_idx], spres[nan_idx], stemp[nan_idx], sdew[nan_idx]
        sqv,stheta,sthetae = sqv[nan_idx],stheta[nan_idx], sthetae[nan_idx]

        #Index the time to match CLAMPS data (between 20-30 seconds off of CLAMPS time)
        stidx = np.array([np.where(stime>=val)[0][0] for val in time[tidx]])

        #Replace the surface values in the CLAMPS data
        pres[:,0],temp[:,0],dew[:,0],qv[:,0] = spres[stidx],stemp[stidx],sdew[stidx],sqv[stidx]
        theta[:,0],thetae[:,0] = stheta[stidx],sthetae[stidx]

        ####################################################################
        #Import the kinematic data
        ####################################################################

        lid0 = sorted(glob(file_dir+'{}/lidar/vad/*{}*'.format(dt.year,t0[k].strftime('%Y%m%d'))))[0]
        lid1 = sorted(glob(file_dir+'{}/lidar/vad/*{}*'.format(dt.year,dt.strftime('%Y%m%d'))))[0]

        ltime0,lheight0,wspd0,wdir0,w0 = read_lidar(lid0,dt,height_max)
        ltime1,lheigh1t,wspd1,wdir1,w1 = read_lidar(lid1,dt,height_max)

        if np.sum((lheight0-lheigh1t)) != 0:
            print('ERROR: LiDAR height levels from {} to {} do not match.'.format(t0[k].strftime('%Y%m%d'),dt.strftime('%Y%m%d')))

        #Concatenate all together
        ltime,lheight = np.concatenate((ltime0,ltime1)),lheight0
        wspd,wdir,w = np.concatenate((wspd0,wspd1)),np.concatenate((wdir0,wdir1)),np.concatenate((w0,w1))

        #Compute the relative time scale
        rel_hours = np.array([val.total_seconds()/3600 for val in (ltime - dt)])
        tidx = np.where((rel_hours>=-8)&(rel_hours<=2))[0]

        #Index the data here
        l_utc,l_rel = ltime[tidx],rel_hours[tidx]
        l_dec = np.array([todectime(val.hour,val.minute,val.second) for val in l_utc])
        #Make the decimal time continous
        for i,val in enumerate(l_dec):
            if val<10:
                l_dec[i] = val+24
        wspd,wdir,w = wspd[tidx,:],wdir[tidx,:],w[tidx,:]

        #Replace with surface MetTower data
        stidx = np.array([np.where(stime>=val)[0][0] for val in l_utc])

        #Replace the surface values in the LiDAR data (w=0 at the surface)
        wspd[:,0],wdir[:,0],w[:,0] = swspd[stidx],swdir[stidx],np.linspace(0,0,stidx.size)

        #Compute u and v components
        u = wspd * np.cos(np.radians(270-wdir))
        v = wspd * np.cos(np.radians(270-wdir))

        ####################################################################
        #Import the vertical stares
        ####################################################################
        stare0 = sorted(glob(file_dir+'{}/lidar/vs/*{}*'.format(dt.year,t0[k].strftime('%Y%m%d'))))[0]
        try:
            stare1 = sorted(glob(file_dir+'{}/lidar/vs/*{}*'.format(dt.year,dt.strftime('%Y%m%d'))))[0]
            no_vs = False
        except IndexError:
            no_vs = True

        #Parse through each vertical stare file
        vs_time0,vs_height0,vs_vel0,vs_cbh0 = read_stare(stare0,dt,height_max)

        if no_vs == False:
            vs_time1,vs_height1,vs_vel1,vs_cbh1 = read_stare(stare1,dt,height_max)

            if np.sum((vs_height0-vs_height1)) != 0:
                print('ERROR: Vertical stare height levels from {} to {} do not match.'.format(t0[k].strftime('%Y%m%d'),dt.strftime('%Y%m%d')))

            #Concatenate all together
            vs_time,vs_height = np.concatenate((vs_time0,vs_time1)),vs_height0
            vs_vel,vs_cbh = np.concatenate((vs_vel0,vs_vel1)),np.concatenate((vs_cbh0,vs_cbh1))
        else:
            vs_time,vs_height = vs_time0, vs_height0
            vs_vel,vs_cbh = vs_vel0, vs_cbh0

        #Compute the relative time scale
        rel_hours = np.array([val.total_seconds()/3600 for val in (vs_time - dt)])
        tidx = np.where((rel_hours>=-8)&(rel_hours<=2))[0]

        #Index the data here
        vs_utc,vs_rel = vs_time[tidx],rel_hours[tidx]
        vs_dec = np.array([todectime(val.hour,val.minute,val.second) for val in vs_utc])
        #Make the decimal time continous
        for i,val in enumerate(vs_dec):
            if val<10:
                vs_dec[i] = val+24
        vs_vel,vs_cbh = vs_vel[tidx,:],vs_cbh[tidx]

        
        ####################################################################
        #Compute wind shear and Integrate CAPE and CIN
        ####################################################################

        top = [0,1,0]
        bot = [1,2,2]
        mag_shear01,mag_shear03,mag_shear13 = np.zeros((l_rel.size)),np.zeros((l_rel.size)),np.zeros((l_rel.size))
        dir_shear01,dir_shear03,dir_shear13 = np.zeros((l_rel.size)),np.zeros((l_rel.size)),np.zeros((l_rel.size))

        mag = [mag_shear01,mag_shear03,mag_shear13]
        dirc = [dir_shear01,dir_shear03,dir_shear13]
        for i in range(len(top)):
            hidx0,hidx1 = np.where(lheight>=bot[i]*1000)[0][0],np.where(lheight>=top[i]*1000)[0][0]

            #Subtract the u and v components
            udiff = u[:,hidx1] - u[:,hidx0]
            vdiff = v[:,hidx1] - v[:,hidx0]

            mag[i][:] = np.sqrt((udiff**2) + (vdiff**2))
            dirc[i][:] = mpcalc.wind_direction(np.array(udiff) * (units.meter/units.second),np.array(vdiff) * (units.meter/units.second),convention='from') / units.degrees

        #Interate CAPE at all levels
        hidx = np.where(theight>=height_max)[0][0]
        missing = np.zeros((theight.size))*np.nan

        #Compute CAPE at all levels
        sbcape2d,sbcin2d = np.zeros((t_utc.size,theight[:hidx].size))*np.nan,np.zeros((t_utc.size,theight[:hidx].size))*np.nan
        
        if cape_cin:
            for j in tqdm(range(t_utc.size)):
                cape = np.array([params.parcelx(profile.create_profile(pres=pres[j,i:], hght=theight[i:], tmpc=temp[j,i:], dwpc=dew[j,i:], wspd=missing[i:],wdir=missing[i:],strictQC=False, profile='default',missing=-9999),flag=5).bplus for i in range(theight[:hidx].size)])
                cin = np.array([params.parcelx(profile.create_profile(pres=pres[j,i:], hght=theight[i:], tmpc=temp[j,i:], dwpc=dew[j,i:], wspd=missing[i:],wdir=missing[i:],strictQC=False, profile='default',missing=-9999),flag=5).bminus for i in range(theight[:hidx].size)])

                #Replace the nans with zeros and append
                cape[np.isnan(cape)],cin[np.isnan(cin)] = 0,0
                sbcape2d[j,:],sbcin2d[j,:] = cape,cin
        
        #Put all the data together
        data = {'t_utc':t_utc,'t_rel':t_rel,'t_dec':t_dec,'passage_time':dt,'t_height':theight,'pres':pres,'temp':temp,
                'dew':dew,'qv':qv,'theta':theta,'thetae':thetae,'sbcape2d':sbcape2d,'sbcin':sbcin2d,'l_utc':l_utc,
                'l_rel':l_rel,'l_dec':l_dec,'l_height':lheight,'wspd':wspd,'wdir':wdir,'u':u,'v':v,'w':w,'cbh':cbh,
                'vs_utc':vs_utc,'vs_rel':vs_rel,'vs_dec':vs_dec,'vs_height':vs_height,'vs_vel':vs_vel,'vs_cbh':vs_cbh,'mag01':mag_shear01,
                'mag03':mag_shear03,'mag13':mag_shear13,'dir01':dir_shear01,'dir03':dir_shear03,'dir13':dir_shear13,
                'date':dt,'lon':lon,'lat':lat}
        data_dict.append(data)
    
    return data_dict


if __name__ == '__main__':
    #Arbitrary start and end times as the code will index -8 to +2 hours relative to p0
    start_h,end_h = 9,5

    #List of linear mode cases
    lt0 = [datetime(2017,3,9,start_h,0),datetime(2017,3,27,start_h,0)]
    lt1 = [datetime(2017,3,10,end_h,0),datetime(2017,3,28,end_h,0)]
    lp0 = [datetime(2017,3,10,6,45),datetime(2017,3,28,3,35)]

    #List of discrete mode cases
    dt0 = [datetime(2017,3,27,start_h,0), datetime(2017,4,5,start_h,0), datetime(2019,4,7,start_h,0), datetime(2016,3,31,start_h,0), datetime(2016,4,30,start_h,0)]
    dt1 = [datetime(2017,3,28,end_h,0), datetime(2017,4,6,end_h,0), datetime(2019,4,8,end_h,0), datetime(2016,4,1,end_h,0), datetime(2016,5,1,end_h,0)]
    dp0 = [datetime(2017,3,27,20,16), datetime(2017,4,5,23,39), datetime(2019,4,7,22,5), datetime(2016,4,1,0,45), datetime(2016,4,30,21,15)]

    #List of mixed mode cases
    mt0 = [datetime(2017,3,1,start_h,0), datetime(2017,3,21,start_h,0), datetime(2017,3,27,start_h,0), datetime(2017,4,22,start_h,0), datetime(2019,3,9,start_h,0)]
    mt1 = [datetime(2017,3,2,end_h,0), datetime(2017,3,22,end_h,0), datetime(2017,3,28,end_h,0), datetime(2017,4,23,end_h,0), datetime(2019,3,10,end_h,0)]
    mp0 = [datetime(2017,3,1,19,55), datetime(2017,3,21,22,15), datetime(2017,3,27,22,35), datetime(2017,4,22,22,25), datetime(2019,3,9,20,25)]


    #Process the data files for a specific case
    file_dir = '/Users/admin/Desktop/VSE/ProfilerData/'
    out_dir = '/Users/admin/Desktop/'

    print('----------------------------- Linear Cases -----------------------------')
    lin = clamps_parser(file_dir,lt0,lp0,height_max=3000)
    print('\n----------------------------- Discrete Cases -----------------------------')
    dis = clamps_parser(file_dir,dt0,dp0,height_max=3000)
    print('\n----------------------------- Mixed Cases -----------------------------')
    mix = clamps_parser(file_dir,mt0,mp0,height_max=3000)

    #Add these to a pickle file
    f = open(out_dir+'clamps_cases_test.pckl', 'wb')
    pickle.dump([lin,dis,mix], f)
    f.close()

    print('Done!')
