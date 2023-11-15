#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 18:00:53 2023

@author: liu3315
"""


#%%
###### This code is to track the wave events ######
from math import pi
import numpy as np
import matplotlib.pyplot as plt
%matplotlib qt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import datetime as dt
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import pandas as pd
import cv2
import copy
import matplotlib.path as mpath
import pickle
import glob
from netCDF4 import Dataset



### A function to calculate distance between two grid points on earth ###
from math import radians, cos, sin, asin, sqrt
 
def haversine(lon1, lat1, lon2, lat2): # 经度1，纬度1，经度2，纬度2 （十进制度数）
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
 
    # haversine公式
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # 地球平均半径，单位为公里
    return c * r * 1000
#%%
### Read the daily Z500-based LWA of SH DJF ##
LWA_DJF_total= np.load("/Volumes/depot/wanglei/data/Reanalysis/MERRA2/LWA_Z/LWA_Z_DJF.npy")
Z_DJF_total = np.load("/Volumes/depot/wanglei/data/Reanalysis/MERRA2/Z500/Z500_DJF.npy")

## Read lon and lat ##
path = glob.glob(r"/Volumes/depot/wanglei/data/Reanalysis/MERRA2/tdt_moist/*.nc4")
path.sort()
N=len(path)   #The number of nc files

### Read basic variables ###
file0 = Dataset(path[0],'r')
lon = file0.variables['lon'][:]
lat = file0.variables['lat'][:]
lev = file0.variables['lev'][:]
lat_SH = lat[0:180]
file0.close()

nlon = len(lon)
nlat = len(lat)
nlat_SH = len(lat_SH)

n_season = 42
nday = len(LWA_DJF_total) ### all season days in the selected years ###
ns = 90                   ### season day ###


#%%
### Extract the DJF date ###
Start_date = dt.datetime(1980, 1, 1)
delta_t = dt.timedelta(days=1)
Datestamp = [Start_date + delta_t*tt for tt in np.arange(15706)]
Date = pd.DataFrame({'date': pd.to_datetime(Datestamp)})
Month = Date['date'].dt.month 
Year = Date['date'].dt.year
Day = Date['date'].dt.day
Date = list(Date['date'])

DJF = []
for t in np.arange(15706):
    if (Month[t] ==1 or Month[t] ==2 or Month[t] ==12):
        if (Month[t]==1 and Year[t]==1980) or (Month[t]==2 and Year[t]==1980) or (Month[t]==12 and Year[t]==2022):
            continue
        elif (Month[t]==2 and Day[t]==29):
            continue
        else:
            DJF.append(Date[t])
Season = DJF


LWA_DJF = []
for i in np.arange(n_season):
    LWA_DJF.append(LWA_DJF_total[0+i*90:90+i*90,:,:])

LWA_DJF_clim = LWA_DJF_total.mean(axis=0)   ### SH DJF LWA climatology ###
LWA_DJF = LWA_DJF - LWA_DJF_clim[np.newaxis,np.newaxis,:,:]  ### anomaly ###

#%%
###### Detect Blocking ######
sd_tresh = 1.5
Duration = 5
keynumber = 15
LWA_zm = np.mean(np.mean(np.mean(LWA_DJF,axis=0),axis=0),axis=1)   ### This is the time mean zoanl mean LWA or FAWA

Blocking_season_total_lat = []
Blocking_season_total_lon = []
Blocking_season_total_date = []

B_freq = np.zeros((nlat_SH, nlon))         ### The final blocking frequency/total number 
B_freq2 = np.zeros((nday ,nlat_SH, nlon))  ### Record whether a grid point is under a blocking for a certain day (only 1 and 0 in this 3D array)
for y in np.arange(n_season):
    LWA = LWA_DJF[y,:,:,:]   
    sd = np.max(np.mean(np.std(LWA,axis=0),axis=1)) # The largest zonal mean std as the threshold
    
    LWA_a = LWA - LWA_zm[np.newaxis,:,np.newaxis]
    
    WE = np.zeros((ns,nlat_SH,nlon),dtype='int8')  
    WE[LWA_a>=sd_tresh*sd] = 255                    # Wave event
    
    ################### connected component-labeling algorithm ################
    num_labels = np.zeros(ns)
    labels = np.zeros((ns,nlat_SH,nlon))
    for d in np.arange(ns):
        num_labels[d], labels[d,:,:], stats, centroids  = cv2.connectedComponentsWithStats(WE[d,:,:], connectivity=4)
        
    ####### connect the label around 180, since they are labeled separately ########
    ####### but actually they should belong to the same label  ########
    labels_new = copy.copy(labels)
    for d in np.arange(ns):
        if np.any(labels_new[d,:,0]) == 0 or np.any(labels_new[d,:,-1]) == 0:   ## If there are no events at either -180 or 179.375, then we don't need to do any connection
            continue
                
        column_0 = np.zeros((nlat_SH,3))       ## We assume there are at most three wave events at column 0 (-180) (actuaaly most of the time there is just one)
        column_end = np.zeros((nlat_SH,3))
        label_0 = np.zeros(3)
        label_end = np.zeros(3)
        
        ## Get the wave event at column 0 (-180) ##
        start_lat0 = 0
        for i in np.arange(3):
            for la in np.arange(start_lat0, nlat_SH):
                if labels_new[d,la,0]==0:
                    continue
                if labels_new[d,la,0]!=0:
                    label_0[i]=labels_new[d,la,0]
                    column_0[la,i]=labels_new[d,la,0]
                    if labels_new[d,la+1,0]!=0:
                        continue
                    if labels_new[d,la+1,0]==0:
                        start_lat0 = la+1
                        break 

            ## Get the wave event at column -1 (179.375) ## 
            start_lat1 = 0
            for j in np.arange(3):
                for la in np.arange(start_lat1, nlat_SH):
                    if labels_new[d,la,-1]==0:
                        continue
                    if labels_new[d,la,-1]!=0:
                        label_end[j]=labels_new[d,la,-1]
                        column_end[la,j]=labels_new[d,la,-1]
                        if labels_new[d,la+1,-1]!=0:
                            continue
                        if labels_new[d,la+1,-1]==0:
                            start_lat1 = la+1
                            break                       
                ## Compare the two cloumns at -180 and 179.375, and connect the label if the two are indeed connected
                if (column_end[:,i]*column_0[:,j]).mean() == 0:
                    continue                
                if (column_end*column_0).mean() != 0:
                    num_labels[d]-=1
                    if label_0[i] < label_end[j]:
                        labels_new[d][labels_new[d]==label_end[j]] = label_0[i]
                        labels_new[d][labels_new[d]>label_end[j]] = (labels_new[d]-1)[labels_new[d]>label_end[j]]            
                    if label_0[i] > label_end[j]:
                        labels_new[d][labels_new[d]==label_0[i]] = label_end[j]
                        labels_new[d][labels_new[d]>label_0[i]] = (labels_new[d]-1)[labels_new[d]>label_0[i]]
                        
                        
    ############ Now we get the maximum LWA location of each individule event ###########
    ############ Also we get the area or width of each individual event #########
    lat_d = []; lon_d = []; max_d = []
    lon_w = []; lon_e = []; area = []
    lon_wide = []
    for d in np.arange(ns):
        if int(num_labels[d]-1)==0:
            lat_list=np.zeros((1));lon_list=np.zeros((1))
            lat_list[0] = np.nan; lon_list[0] = np.nan
            lat_d.append(lat_list)
            lon_d.append(lon_list)
            lon_w.append(lon_list)
            lon_e.append(lon_list)
            area.append(lon_list)
            lon_wide.append(lon_list)
            continue
        
        lat_list=np.zeros(( int(num_labels[d]-1) ));    lon_list=np.zeros(( int(num_labels[d]-1) ));   max_list=np.zeros(( int(num_labels[d]-1) ))
        lon_w_list = np.zeros(( int(num_labels[d]-1) ));lon_e_list=np.zeros(( int(num_labels[d]-1) )); area_list = np.zeros((int(num_labels[d]-1))) 
        lon_wide_list = np.zeros(( int(num_labels[d]-1) ))
        for n in np.arange(0,int(num_labels[d]-1)):
            LWA_d = np.zeros((nlat_SH, nlon))
            LWA_d[labels_new[d]==n+1]=LWA_a[d][labels_new[d]==n+1]   ### isolate that wave event ###
            ### Get the maximum location ###
            lat_list[n] = lat_SH[np.squeeze(np.array(np.where( LWA_d==LWA_d.max() )))[0]]
            lon_list[n] = lon[np.squeeze(np.array(np.where( LWA_d==LWA_d.max() )))[1]]
            max_list[n] = LWA_d.max()
            ### Get the west east boundary longitude and area ###
            for lo in np.arange(nlon):
                if (np.any(LWA_d[:,lo])) and (not np.any(LWA_d[:,lo-1])):
                    lon_w_list[n] = lon[lo]
                if (not np.any(LWA_d[:,lo])) and (np.any(LWA_d[:,lo-1])):
                    lon_e_list[n] = lon[lo-1]
            ### Get the width from west to east ###
            if lon_e_list[n]-lon_w_list[n] > 0:
                lon_wide_list[n] = lon_e_list[n]-lon_w_list[n]
            else:
                lon_wide_list[n] = 360+(lon_e_list[n]-lon_w_list[n])
            ### Get the total area ###
            area_count = np.zeros((nlat_SH, nlon))
            area_count[labels_new[d]==n+1]=1
            area_list[n] = np.sum(area_count)
                        
        lat_d.append(lat_list);   lon_d.append(lon_list);    max_d.append(max_list)
        lon_w.append(lon_w_list); lon_e.append(lon_e_list);  area.append(area_list)
        lon_wide.append(lon_wide_list)

        
                
    ########### Now we begin to track wave events #########
    ########### A blocking event requires at least 5 consecutive wave events ########
    ######## The distance between two wave events should be less than 13.5 of latitudes and 18 of longitudes #######
    ######## Also, the blocking event should be large enough, the width of such event is at least 15 degress width #######
    Blocking_lat = []; Blocking_lon = []; Blocking_date = []
    Blocking_lon_wide = []; Blocking_area = []
    for d in np.arange(ns-1):        
        for i in np.arange(len(lon_d[d])):
            day = 0
            track_lon = []; track_lat = []; track_date = []
            track_lon_index = []; track_lat_index = []
            track_lon_wide = []; track_area = []
            B_count = np.zeros((nlat_SH, nlon))
            B = np.zeros((nlat_SH, nlon))
            B_count2 = []
            
            B_count[labels_new[d+day]==i+1]+=1
            B[labels_new[d+day]==i+1]=1            
            B_count2.append(B)
            
            track_lon.append(lon_d[d+day][i])
            track_lon_index.append(i)
            lon_shift = lon_d[d+day+1][:]-track_lon[-1]
            for n in np.arange(len(lon_shift)):        ### -180 and 179.375 are connnected.        
                if lon_shift[n] > 180:
                    lon_shift[n] = lon_shift[n]-360
                if lon_shift[n] < -180:
                    lon_shift[n] = lon_shift[n] + 360
                
            
            track_lat.append(lat_d[d+day][i])
            track_lat_index.append(i)
            lat_shift = lat_d[d+day+1][:]-track_lat[-1]
            
            total_shift = abs(lon_shift) + abs(lat_shift)
            total_shift_min = np.nanmin(total_shift)
            if np.isnan(total_shift_min):
                lon_shift_min = np.nan
                lat_shift_min = np.nan
            else:
                for n in np.arange(len(total_shift)):
                    if total_shift[n] == total_shift_min:
                        lon_shift_min = lon_shift[n]
                        lat_shift_min = lat_shift[n]
                        break
                       
            track_date.append(Season[y*ns+d+day])
            track_lon_wide.append(lon_wide[d+day][i])
            track_area.append(area[d+day][i])
                    
            while abs(lon_shift_min) < 18 and abs(lat_shift_min) < 13.5 and total_shift_min < 31.5 and track_lat[-1]<-30:
                track_date.append(Season[y*ns+d+day+1])
                for n in np.arange(len(total_shift)):
                    if total_shift[n] == total_shift_min:
                        B_count[labels_new[d+day+1]==n+1]+=1
                        B = np.zeros((nlat_SH, nlon))
                        B[labels_new[d+day+1]==n+1]=1            
                        B_count2.append(B)
                           
                        track_lon.append(lon_d[d+day+1][n])
                        track_lon_index.append(n)
                        track_lat.append(lat_d[d+day+1][n])
                        track_lat_index.append(n)
                        track_lon_wide.append(lon_wide[d+day+1][n])
                        track_area.append(area[d+day+1][n])
                        break
                        
                day+=1
                
                if d+day+1>ns-1:
                    break
                
                lon_shift = lon_d[d+day+1][:]-track_lon[-1]
                for n in np.arange(len(lon_shift)):                
                    if lon_shift[n] > 180:
                        lon_shift[n] = lon_shift[n]-360
                    if lon_shift[n] < -180:
                        lon_shift[n] = lon_shift[n] + 360
                lat_shift = lat_d[d+day+1][:]-track_lat[-1]
                total_shift = abs(lon_shift) + abs(lat_shift)
                total_shift_min = np.nanmin(total_shift)
                for n in np.arange(len(total_shift)):
                    if total_shift[n] == total_shift_min:
                        lon_shift_min = lon_shift[n]
                        lat_shift_min = lat_shift[n]
                        break
                
            ### For this tracked wave event evolution, we want to know how many days the longitude width is longer than 15 degree l###
            wide_threshold = 5
            n_large_wave = 0
            for j in np.arange(len(track_lon_wide) - wide_threshold + 1 ):
                if track_lon_wide[j]>15 and track_lon_wide[j+1]>15 and track_lon_wide[j+2]>15:
                    n_large_wave+=1
                    
            ### Decide whetehr it is a blocking event: it should be last longer than at least 5 days ###
            ### Also, the area/width should be large enough ### 
            if day+1 >= Duration and n_large_wave>0:
                Blocking_lon.append(track_lon)
                Blocking_lat.append(track_lat)
                Blocking_date.append(track_date)
                Blocking_lon_wide.append(track_lon_wide)
                Blocking_area.append(track_area)
                B_freq += B_count
                for dd in np.arange(len(B_count2)):
                    B_freq2[y*ns+d+dd,:,:]+= B_count2[dd]
                
                ### Once we successfully detected a blocking, we mark that with nan to avoid repeating ###
                for dd in np.arange(day+1):
                    lon_d[d+dd][track_lon_index[dd]] = np.nan
                    lat_d[d+dd][track_lat_index[dd]] = np.nan
                    
    Blocking_season_total_lat.append(Blocking_lat)
    Blocking_season_total_lon.append(Blocking_lon)
    Blocking_season_total_date.append(Blocking_date)
    
    print(y)
                    

