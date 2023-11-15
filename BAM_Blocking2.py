#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 17:51:44 2023

@author: liu3315
"""

#%%
###### This code is to explore the conidtional probability of blocking events based on MERRA2 ######
###### P(blocking|BAM) vs P(blocking)
from scipy.io import netcdf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import datetime as dt
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import pandas as pd
import glob
import copy
import pickle
import matplotlib.path as mpath
from netCDF4 import Dataset

import multiprocessing ###  for parallel Coding ###

#%%
### Read basic variables ###
path = glob.glob(r"/depot/wanglei/data/Reanalysis/MERRA2/LWA_Z/*.nc")
path.sort()
N=len(path)   #The number of nc files

file0 = Dataset(path[0],'r')
lon0 = file0.variables['lon'][:]
lon = np.zeros((len(lon0)+1))
lon[0:len(lon0)]=lon0
lon[-1] = 180
lat = file0.variables['lat'][:]
lev = file0.variables['lev'][:]
lat_SH = lat[0:180]

file0.close()

nlon = len(lon)
nlat = len(lat)
nlat_SH = len(lat_SH)

n_season = 42                        ### number of (DJF) season ###
season_day= 90                       ### how many days in one season ###
season_all = n_season*season_day     ### entire days in all such season ###

### Read key index and variables ###
BI = np.load('/depot/wanglei/data/Reanalysis/MERRA2/EKE/BI.npy')                             ### This is full BAM index from 1980 to 2022, 15706 in total ###

LWA0 = np.load("/depot/wanglei/data/Reanalysis/MERRA2/LWA_Z/LWA_Z_DJF.npy")                          ### This is SH DJF daily LWA data, 3870 in total ###

LWA0 = LWA0[0:season_all,:,:]
LWA = np.zeros((season_all,nlat_SH,nlon))
LWA[:,:,0:len(lon0)]=LWA0
LWA[:,:,-1] = LWA0[:,:,0]
del LWA0

with open("/depot/wanglei/data/Reanalysis/MERRA2/LWA_Z/Blocking_season_total_date", "rb") as fp:    ### This is the blocking events dates ### 
    Blocking_season_total_date2 = pickle.load(fp)
    
with open("/depot/wanglei/data/Reanalysis/MERRA2/LWA_Z/Blocking_season_total_lon", "rb") as fp:     ### This is the blocking events center longitudes ###
    Blocking_season_total_lon2 = pickle.load(fp)
    
with open("/depot/wanglei/data/Reanalysis/MERRA2/LWA_Z/Blocking_season_total_lat", "rb") as fp:     ### This is the blocking events center latitudes ###
    Blocking_season_total_lat2 = pickle.load(fp)

#%%
### Time management ###
### For the MERRA2 data, the time range we will analyze is from 1980/01/01 to 2022/12/31, 15706 days in total ###
Start_date = dt.datetime(1980, 1, 1)
delta_t = dt.timedelta(days=1)
Datestamp = [Start_date + delta_t*tt for tt in np.arange(15706)]
Date = pd.DataFrame({'date': pd.to_datetime(Datestamp)})
Month = Date['date'].dt.month 
Year = Date['date'].dt.year
Day = Date['date'].dt.day

### Since 2.29 is removed in BAM index, we need to remove that in the above imported dataset as well###
Date = list(Date['date'])
Date2 = []
Month2 = []
Year2 = []

ti = -1
for t in np.arange(15706):
    if str(Date[t])[4:19] != '-02-29 00:00:00':
        ti +=1
        Date2.append(Date[t])
        Month2.append(Month[t])
        Year2.append(Year[t])


Date_DJF = []
for t in np.arange(len(Date2)):
    if (Month2[t]==12 or Month2[t]==1 or Month2[t]==2) and ~(Month2[t]==1 and Year2[t]==1980) and ~(Month2[t]==2 and Year2[t]==1980) and ~(Month2[t]==12 and Year2[t]==2022):
        Date_DJF.append(Date2[t])

Date_DJF2 = []
for t in np.arange(37):
    Date_DJF2.append(Date2[334+t*365:424+t*365])



BI_DJF = []
for i in np.arange(n_season):
    BI_DJF.append(BI[334+i*365:424+i*365])


#%%
### Plot one year BAM index ###
t=0
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.plot(np.arange(90), BI_DJF[t][:,0], '-k', linewidth=2)
# plt.plot(np.arange(90), BI_DJF2[t], '-r', linewidth=2)
plt.title("PC time series ("+str(1980+t)+"-"+str(1980+t+1)+")", pad=5, fontdict={'family':'Times New Roman', 'size':16})
plt.xlabel('Days',fontsize=12)
plt.ylabel('PC',fontsize=12)
plt.xticks([0,29,59,89])
ax.set_xticklabels([1,30,60,90])
ax.grid()
plt.show()


#%%
###### Blocking probability without any conditions for each grid points ######
###### This is just the blocking frequency we commonly plot ######
B_freq_old = np.load('/depot/wanglei/data/Reanalysis/MERRA2/LWA_Z/B_freq.npy')
B_freq = np.zeros((nlat_SH,nlon))
B_freq[:,0:len(lon0)]=B_freq_old
B_freq[:,-1] = B_freq_old[:,0]
B_freq = B_freq/season_all
del B_freq_old

### Make a quick plot ###
minlev =0
maxlev = B_freq[:,:].max()
levs = np.linspace(minlev,maxlev,15)

proj=ccrs.PlateCarree(central_longitude=180)
fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection=proj)
h1 = plt.contourf(lon,lat_SH[:], B_freq[:,:], levs, transform=ccrs.PlateCarree(), cmap='RdBu_r')  #_r is reverse
# b = plt.contour(lon,lat_SH[20:], B_freq[20:,:], levs, transform=ccrs.PlateCarree())  #_r is reverse
plt.xlabel('longitude',fontsize=12)
plt.ylabel('latitude',fontsize=12)
plt.title("Natural Blocking Probability (without conditions)", pad=5, fontdict={'family':'Times New Roman', 'size':14})

ax.coastlines()
ax.gridlines(linestyle="--", alpha=0.7)
# ax.set_extent([-180,180,0,-90],crs=ccrs.PlateCarree())
ax.set_xticks([0,60,120,180,240,300,358.5], crs=ccrs.PlateCarree())
ax.set_yticks([-70,-60,-50,-40,-30], crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label='FALSE')
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter) 

cbar = fig.add_axes([0.92,0.35,0.015,0.3])
cb = plt.colorbar(h1, cax=cbar, ticks=[0,0.01,0.02,0.03])  ## For LWA
cb.set_label('blocking frequency',fontsize=10)


#%%
### Define BAM events ###

BI_copy = copy.copy(BI)
BAM_event_all = []      ## To store the date or index with large pc1 value ##
BAM_event_BI_all = []   ## The value of the BI at that peaking date ##
number = 0
while 1:
    index = np.squeeze(np.array(np.where( BI_copy==np.nanmax(BI_copy) )))[0]
    if BI_copy[index]>=BI[index+1] and BI[index]>=BI[index-1] and BI_copy[index]>=BI[index+2] and BI[index]>=BI[index-2]:        
        BAM_event_all.append(index)
        BAM_event_BI_all.append(BI[index])
        number+=1
        BI_copy[index-12:index+13,0] = np.nan
        if number > 550:
            break
    else:
        BI_copy[index]=np.nan
        
  
### We extract the BAM event in DJF ###
BAM_event_DJF = []
BAM_event_date_DJF = [] 
for i in np.arange(len(BAM_event_all)):
    if (Month2[BAM_event_all[i]] == 12) | (Month2[BAM_event_all[i]] == 1) | (Month2[BAM_event_all[i]] == 2):
        if (Month2[BAM_event_all[i]]==1 and Year2[BAM_event_all[i]]==1980) or (Month2[BAM_event_all[i]]==2 and Year2[BAM_event_all[i]]==1980) or (Month2[BAM_event_all[i]]==12 and Year2[BAM_event_all[i]]==2022):
            continue
        else:
            BAM_event_DJF.append(BAM_event_all[i])
            BAM_event_date_DJF.append(Date2[BAM_event_all[i]])


BAM_event_DJF[54] = 1128
BAM_event_DJF[31] = np.nan
BAM_event_DJF[123] = np.nan
BAM_event_DJF[114] = np.nan
BAM_event_DJF[28] = 5829
BAM_event_DJF[33] = np.nan
BAM_event_DJF = np.array(BAM_event_DJF)
BAM_event_DJF = BAM_event_DJF[np.logical_not(np.isnan(BAM_event_DJF))]
BAM_event_DJF = BAM_event_DJF.astype(int)
BAM_event_DJF =list(BAM_event_DJF )

BAM_event_date_DJF = [] 
for i in np.arange(len(BAM_event_DJF)):
    BAM_event_date_DJF.append(Date2[BAM_event_DJF[i]])
    

### correct the index in the Date_DJF time series ###
BAM_event = []
BAM_event_BI = []
for i in np.arange(len(BAM_event_DJF)):
    for t in np.arange(len(Date_DJF)):
        if Date_DJF[t] == BAM_event_date_DJF[i]:
            BAM_event.append(t)
            BAM_event_BI.append(BAM_event_BI_all[i])
            break



BAM_event_BI = np.array(BAM_event_BI)



#%%
###### Blocking probability when BAM index is positive ######
### Read the 3D array which records the blocking area for each day  ###
B_freq2_old = np.load('/depot/wanglei/data/Reanalysis/MERRA2/LWA_Z/B_freq2.npy') 
B_freq2_old =B_freq2_old[0:season_all,:,:]
B_freq2 = np.zeros((season_all,nlat_SH,nlon))
B_freq2[:,:,0:len(lon0)]=B_freq2_old
B_freq2[:,:,-1] = B_freq2_old[:,:,0]
del B_freq2_old 

B_B_P = np.zeros((season_all,nlat_SH,nlon))   ## This array is to store the blocking which falls into positive BAM ##
n_B_P = 0                                     ## How many days are in positive BAM phase
B_B_N = np.zeros((season_all,nlat_SH,nlon))   ## This array is to store the blocking which falls into negative BAM ##
n_B_N = 0                                     ## How many days are in negative BAM phase
## We only count blocking events on positive BAM phase ##
for y in np.arange(n_season):
    for d in np.arange(season_day):
        if BI_DJF[y][d]>0:
            n_B_P+=1
            B_B_P[y*season_day+d,:,:] = B_freq2[y*season_day+d,:,:] 
        else:
            n_B_N +=1
            B_B_N[y*season_day+d,:,:] = B_freq2[y*season_day+d,:,:] 
            
            
B_B_P_freq = B_B_P.sum(axis=0)/n_B_P            
B_B_N_freq = B_B_N.sum(axis=0)/n_B_N

### Make a plot to compare the probability ###
minlev =0
maxlev = B_B_P_freq[:,:].max()
levs = np.linspace(minlev,maxlev,15)

proj=ccrs.PlateCarree(central_longitude=180)
fig = plt.figure()

ax = fig.add_subplot(2,1,1, projection=proj)
h1 = plt.contourf(lon,lat_SH[:], B_B_P_freq[:,:], levs, transform=ccrs.PlateCarree(), cmap='RdBu_r')  #_r is reverse
plt.xlabel('longitude',fontsize=12)
plt.ylabel('latitude',fontsize=12)
plt.title("Blocking Probability on Positive BAM Phase ", pad=5, fontdict={'family':'Times New Roman', 'size':12})

ax.coastlines()
ax.gridlines(linestyle="--", alpha=0.7)
# ax.set_extent([-180,180,0,-90],crs=ccrs.PlateCarree())
ax.set_xticks([0,60,120,180,240,300,358.5], crs=ccrs.PlateCarree())
ax.set_yticks([-80,-60,-40,-20,0], crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label='FALSE')
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter) 

bx = fig.add_subplot(2,1,2, projection=proj)
h2 = plt.contourf(lon,lat_SH[:], B_B_N_freq[:,:], levs, transform=ccrs.PlateCarree(), cmap='RdBu_r')  #_r is reverse
plt.xlabel('longitude',fontsize=12)
plt.ylabel('latitude',fontsize=12)
plt.title("Blocking Probability on Negative BAM Phase ", pad=5, fontdict={'family':'Times New Roman', 'size':12})

bx.coastlines()
bx.gridlines(linestyle="--", alpha=0.7)
# ax.set_extent([-180,180,0,-90],crs=ccrs.PlateCarree())
bx.set_xticks([0,60,120,180,240,300,358.5], crs=ccrs.PlateCarree())
bx.set_yticks([-80,-60,-40,-20,0], crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label='FALSE')
lat_formatter = LatitudeFormatter()
bx.xaxis.set_major_formatter(lon_formatter)
bx.yaxis.set_major_formatter(lat_formatter) 

cbar = fig.add_axes([0.95,0.15,0.015,0.7])
cb = plt.colorbar(h1, cax=cbar, ticks=[0.02,0.04,0.06,0.08,0.1,0.12,0.14,0.16])  ## For LWA
cb.set_label('blocking probability',fontsize=10)





#%%
###### Above are blocking probability under positive/negative phase of BAM ######
###### But a positive/negative phases are not acccurate enough in terms of 'predict', we want to see whether the exact peaking date (+- several days around) can be a good indicator ######
###### So, now we calculate blocking probability +- T days around the BMA peaking ######
### Now we extract the blocking events +- T day around the BAM peaking ###
T=1
n_BAM = len(BAM_event_DJF)
B_B = np.zeros((season_all,nlat_SH,nlon))  ## This array is to store the blocking which falls into +-T day of BAM peaking date
B_LB = np.zeros((season_all,nlat_SH,nlon)) ## This array is to store the blocking whihc falls into +-T day of low BAM date
for i in np.arange(n_BAM):
    B_B[BAM_event[i]-T:BAM_event[i]+T+1,:,:] = B_freq2[BAM_event[i]-T:BAM_event[i]+T+1,:,:]
    B_LB[BAM_event[i]-12-T:BAM_event[i]-12+T+1,:,:] = B_freq2[BAM_event[i]-12-T:BAM_event[i]-12+T+1,:,:]
    
B_B_day = n_BAM*(2*T+1)
B_B_num = B_B.sum(axis=0)
B_LB_num = B_LB.sum(axis=0)
B_B_freq = (B_B_num/B_B_day)
B_LB_freq = (B_LB_num/B_B_day)
        
B_freq_diff = B_B_freq/B_freq
    
minlev =0
maxlev = B_B_freq[20:140,:].max()
levs = np.linspace(0.02,0.16,20)

proj=ccrs.PlateCarree(central_longitude=180)
fig = plt.figure(figsize=[10,7])

ax = fig.add_subplot(3,1,1, projection=proj)
h1 = plt.contourf(lon,lat_SH[20:140], B_freq[20:140,:], levs, transform=ccrs.PlateCarree(), cmap='hot_r')  #_r is reverse
# plt.xlabel('longitude',fontsize=10)
plt.ylabel('latitude',fontsize=10)
plt.title("(a) Blocking Frequency (Climatology)", pad=5, fontdict={'family':'Times New Roman', 'size':10})
ax.coastlines()
ax.gridlines(linestyle="--", alpha=0.7)
ax.set_xticks([0,60,120,180,240,300,358.5], crs=ccrs.PlateCarree())
ax.set_yticks([-80,-60,-40,-20], crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label='FALSE')
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter) 

bx= fig.add_subplot(3,1,2, projection=proj)
h1 = plt.contourf(lon,lat_SH[20:140], B_B_freq[20:140,:], levs, transform=ccrs.PlateCarree(), cmap='hot_r')  #_r is reverse
bx.coastlines()
bx.gridlines(linestyle="--", alpha=0.7)
bx.set_xticks([0,60,120,180,240,300,358.5], crs=ccrs.PlateCarree())
bx.set_yticks([-80,-60,-40,-20], crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label='FALSE')
lat_formatter = LatitudeFormatter()
bx.xaxis.set_major_formatter(lon_formatter)
bx.yaxis.set_major_formatter(lat_formatter) 
# plt.xlabel('longitude',fontsize=10)
plt.ylabel('latitude',fontsize=10)
plt.title("(b) Conditional Blocking Frequency (High BAM State)", pad=5, fontdict={'family':'Times New Roman', 'size':10})


cx = fig.add_subplot(3,1,3, projection=proj)
h2 = plt.contourf(lon,lat_SH[20:140], B_B_N_freq[20:140,:], levs, transform=ccrs.PlateCarree(), cmap='hot_r')  #_r is reverse
plt.xlabel('longitude',fontsize=10)
plt.ylabel('latitude',fontsize=10)
plt.title("(c) Conditional Blocking Frequency (Low BAM State) ", pad=5, fontdict={'family':'Times New Roman', 'size':10})
cx.coastlines()
cx.gridlines(linestyle="--", alpha=0.7)
cx.set_xticks([0,60,120,180,240,300,358.5], crs=ccrs.PlateCarree())
cx.set_yticks([-80,-60,-40,-20], crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label='FALSE')
lat_formatter = LatitudeFormatter()
cx.xaxis.set_major_formatter(lon_formatter)
cx.yaxis.set_major_formatter(lat_formatter) 

plt.subplots_adjust(hspace=0.5, right=0.8)

cbar = fig.add_axes([0.85,0.1,0.015,0.8])
cb = plt.colorbar(h1, cax=cbar, ticks=[0.02,0.04,0.06,0.08,0.1,0.12,0.14,0.16])
cb.set_label('frequency',fontsize=10)

plt.savefig("/home/liu3315/Research/BAM/BAM and Blocking/Figure3.png",dpi=600,layout='tight')



