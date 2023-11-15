#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 16:04:37 2023

@author: liu3315
"""

###### BAM_Blocking Manuscript Figure 1 and Figure 4 ######
#%%
###### This code is to explroe the connection between BAM and Blocking ######
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


### Read basic variables ###
path_LWA = glob.glob(r"/depot/wanglei/data/Reanalysis/MERRA2/LWA_Z/*.nc")
path_LWA.sort()
N=len(path_LWA)  

path_Z = glob.glob(r"/depot/wanglei/data/Reanalysis/MERRA2/Z500/*.nc")
path_Z.sort()
N=len(path_Z)  

file0 = Dataset(path_LWA[0],'r')
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
BI_DJF = []
BI_DJF2 = np.zeros((season_all))
for i in np.arange(n_season):
    BI_DJF.append(BI[334+i*365:424+i*365,0])
    BI_DJF2[0+i*90:90+i*90] = BI[334+i*365:424+i*365,0]



LWA0 = np.load("/depot/wanglei/data/Reanalysis/MERRA2/LWA_Z/LWA_Z_DJF.npy")                          ### This is SH DJF daily LWA data, 3870 in total ###
LWA0 = LWA0[0:season_all,:,:]
LWA = np.zeros((season_all,nlat_SH,nlon))
LWA[:,:,0:len(lon0)]=LWA0
LWA[:,:,-1] = LWA0[:,:,0]
del LWA0

Z0 = np.load("/depot/wanglei/data/Reanalysis/MERRA2/Z500/Z500_DJF.npy")                              ### This is SH DJF daily LWA data, 3870 in total ###
Z0 = Z0[0:season_all,:,:]
Z = np.zeros((season_all,nlat_SH,nlon))
Z[:,:,0:len(lon0)]=Z0
Z[:,:,-1] = Z0[:,:,0]
del Z0

with open("/depot/wanglei/data/Reanalysis/MERRA2/LWA_Z/Blocking_season_total_date", "rb") as fp:    ### This is the blocking events dates ### 
    Blocking_season_total_date2 = pickle.load(fp)
    
with open("/depot/wanglei/data/Reanalysis/MERRA2/LWA_Z/Blocking_season_total_lon", "rb") as fp:     ### This is the blocking events center longitudes ###
    Blocking_season_total_lon2 = pickle.load(fp)
    
with open("/depot/wanglei/data/Reanalysis/MERRA2/LWA_Z/Blocking_season_total_lat", "rb") as fp:     ### This is the blocking events center latitudes ###
    Blocking_season_total_lat2 = pickle.load(fp)


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

### Extract the DJF calendar only ###
Date_DJF = []
for t in np.arange(len(Date2)):
    if (Month2[t]==12 or Month2[t]==1 or Month2[t]==2) and ~(Month2[t]==1 and Year2[t]==1980) and ~(Month2[t]==2 and Year2[t]==1980) and ~(Month2[t]==12 and Year2[t]==2022):
        Date_DJF.append(Date2[t])

Date_DJF2 = []
for t in np.arange(37):
    Date_DJF2.append(Date2[334+t*365:424+t*365])



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


### This process is to further filter the BAM events, since some of them are just synoptc peaking ###
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
    

### correct the index in the original time seires Date with 02/29 ###
BAM_event_DJF2 = []
for i in np.arange(len(BAM_event_DJF)):
    for t in np.arange(len(Date)):
        if Date[t] == BAM_event_date_DJF[i]:
            BAM_event_DJF2.append(t)
            break
        
### correct the index in the DJF time seires  ###
BAM_event_DJF3 = []
for i in np.arange(len(BAM_event_DJF)):
    for t in np.arange(len(Date_DJF)):
        if Date_DJF[t] == BAM_event_date_DJF[i]:
            BAM_event_DJF3.append(t)
            break
        
        
### Now we get the location of each BAM event (the maximum LWA region) ###
lon_BAM = []; lat_BAM = []
lon_index_BAM = []; lat_index_BAM = []
for i in np.arange(len(BAM_event_DJF2)):

    file0 = Dataset(path_LWA[BAM_event_DJF2[i]],'r')
    LWA_d = file0.variables['LWA_Z500'][0,0,0:180,:]
    file0.close()

    lon_BAM.append(lon[np.squeeze(np.array(np.where( LWA_d == LWA_d[40:120,:].max())))[1]])
    lat_BAM.append(lat[np.squeeze(np.array(np.where( LWA_d == LWA_d[40:120,:].max())))[0]])
    
    lon_index_BAM.append(np.squeeze(np.array(np.where( LWA_d == LWA_d[40:120,:].max())))[1])
    lat_index_BAM.append(np.squeeze(np.array(np.where( LWA_d == LWA_d[40:120,:].max())))[0])
        
### South Pacific and Storm Track BAM event ###
n_BAM_SP = 0
n_BAM_ST = 0
for t in np.arange(len(BAM_event_DJF)):
    if lon_BAM[t] < -90 or lon_BAM[t] > 160:
        n_BAM_SP+=1
    else:
        n_BAM_ST+=1
# %%
### You can plot the Hovmller plot for each BAM events with blocking dates makred on it ###
n_BAM = len(BAM_event_DJF)

duration_half = 12
duration = 2* duration_half+1

LWA_BAM_all = np.zeros((n_BAM,duration,nlon))
Blocking_BAM_all = np.zeros((duration,nlon))

n=-1
for t in np.arange(len(BAM_event_DJF)):
    # if lon_BAM[t] < -90 or lon_BAM[t] > 160:
    n+=1
    ### In total it is a 25 days array ###
    LWA_event = np.zeros((duration,nlat_SH,nlon))
    
    ### Read the LWA data, note, use the original time series ###
    for d in np.arange(duration):
        index = BAM_event_DJF2[t]-duration_half+d
                
        file_LWA = Dataset(path_LWA[index],'r')
        LWA_d = file_LWA.variables['LWA_Z500'][0,0,0:180,:]
        LWA_event[d,:,0:nlon-1] = LWA_d
        LWA_event[d,:,-1] = LWA_d[:,0]
        file_LWA.close()    
        
    lon_c = lon_BAM[t]; lat_c = lat_BAM[t]

    ## Based on this latitude, we do a meridional average +- 5 latitudes to represyent the value for each longitude ##
    LWA_lon = LWA_event[:,lat_index_BAM[t]-5:lat_index_BAM[t]+6,:].mean(axis=1)

    ## For composite visulization, we will put the center lontitude to the center, so we need to reorganize the array ##
    LWA_lon_plot = np.roll(LWA_lon, 288-lon_index_BAM[t], axis=1)
    lon_plot = np.roll(lon,  288-lon_index_BAM[t])

    LWA_BAM_all[n,:,:] = LWA_lon_plot

    ## We build a calendar for this 25 day BAM event ##
    Date_BAM = Date2[BAM_event_DJF[t]-duration_half:BAM_event_DJF[t]+duration_half+1]
    year = Year2[BAM_event_DJF[t]]

    
    ## To search all blocking events in this time period ##
    Blocking_year_index = []
    Blocking_label_index = []
    for i in np.arange(year-1980-1,year-1980+1):
        if i>(n_season-1):
            break
        for j in np.arange(len(Blocking_season_total_date2[i])):            
            BB = np.isin(Blocking_season_total_date2[i][j], Date_BAM)
            if BB.sum() == len(Blocking_season_total_date2[i][j]):
                Blocking_year_index.append(i)
                Blocking_label_index.append(j)
                
    n_blocking = len(Blocking_year_index)
                
    
    ### Now we plot ###
    maxlevel = LWA_lon[:,:].max()
    minlevel = LWA_lon[:,:].min()  
    levs = np.linspace(0, maxlevel, 11)
            
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    for i in np.arange(n_blocking):
        for j in np.arange(len(Date_BAM)):
            if Blocking_season_total_date2[Blocking_year_index[i]][Blocking_label_index[i]][0] ==  Date_BAM[j]:
                date_index = j
                break
        
        for j in np.arange(len(Blocking_season_total_date2[Blocking_year_index[i]][Blocking_label_index[i]])):
            if abs(Blocking_season_total_lon2[Blocking_year_index[i]][Blocking_label_index[i]][j])<0.001:
                Blocking_season_total_lon2[Blocking_year_index[i]][Blocking_label_index[i]][j]=0
            lon_label = np.squeeze(np.array(np.where( Blocking_season_total_lon2[Blocking_year_index[i]][Blocking_label_index[i]][j] == lon_plot )))
                            
            # ax.plot(lon[lon_label], date_index+j, marker='+', color='blue', markersize=7,
            #     alpha=0.7)
            Blocking_BAM_all[date_index+j,lon_label] = Blocking_BAM_all[date_index+j,lon_label]+1
            


    plt.contourf(lon, np.arange(duration), LWA_lon_plot, levs,cmap='YlOrBr',extend ='max')
    cb=plt.colorbar(aspect=35)
    cb.set_label('LWA',fontsize=10)
    # plt.vlines(0,0,duration-1,colors='b')
    plt.contour(lon, np.arange(duration), LWA_lon_plot, levs, colors="k", linewidths=0.5)
    
    plt.xlabel('longitude',fontsize=10, labelpad=-1)
    plt.ylabel('time (days)',fontsize=10)
    ax.set_yticks([0,3,6,9,12,15,18,21,24])
    ax.set_yticklabels([-12,-9,-6,-3,0,3,6,9,12])
    ax.set_xticks([0])
    ax.set_xticklabels([lon_c])
    plt.title("BAM event (lat"+str(lat_c)+', lon'+str(lon_c)+') '+Date[BAM_event_DJF2[t]].strftime("%Y/%m/%d"), pad=10, fontdict={'family':'Times New Roman', 'size':12})




# %%
#### Plot the compoiste blocking events on top of the composite BAM ####
#### Figure 4 of the BAM Blocking manuscript ####
LWA_com = LWA_BAM_all.mean(axis=0)
maxlevel = LWA_com[:,:].max()
minlevel = LWA_com[:,:].min()
levs = np.linspace(4e7, 1.8e8, 8)

#### Corase Graining ####
Corase_y = 25
Corase_x = 36
Blocking_BAM_all_corase = np.zeros((Corase_y,Corase_x))
Corase_y_space = int(duration/Corase_y)
Corase_x_space = int((nlon-1)/Corase_x)
for y in np.arange(Corase_y):
    for x in np.arange(Corase_x):
        Blocking_BAM_all_corase[y,x] = np.sum(Blocking_BAM_all[y*Corase_y_space:(y+1)*Corase_y_space, x*Corase_x_space:(x+1)*Corase_x_space])

fig = plt.figure()
ax = fig.add_subplot(1,1,1)    
plt.pcolormesh(Blocking_BAM_all_corase, cmap='hot_r',vmin=0, vmax=50)
cb=plt.colorbar(aspect=35)
cb.set_label('number of blocking events',fontsize=10) 
plt.title("Composite BAM and Blocking",pad=5)
plt.ylabel('time (days)',fontsize=10)
plt.xlabel('relative longitudes')
ax.set_yticks([0,3,6,9,12,15,18,21,24])
ax.set_yticklabels([-12, -9,-6,-3,0,3,6,9,12])
plt.vlines(18,0,25,colors='k', linestyles='dashed')
plt.hlines(12,0,36,colors='k', linestyles='dashed')
ax.set_xticks([0,6,12,18,24,30,36])
ax.set_xticklabels(['-180','-120','-60','lon_c','+60','+120','+180'])

bx = ax.twiny()  # instantiate a second axes that shares the same y-axis
plt.contour(lon, np.arange(duration), LWA_com, levs, colors="k", linewidths=0.2)
plt.axis( 'off' )

plt.savefig("/home/liu3315/Research/BAM/BAM and Blocking/Figure4.png",dpi=600)





# %%
#### We take 2011-2012 as an example to illustrate how BMA and blokcing is connected ####
#### This is the Figure 1 in the BAM and Blocking manuscript ####

t=31        ## Year 2011-2012 ##
n_blocking = len(Blocking_season_total_date2[t])

## Find th blocking starting date ##
blocking_start = []
blocking_start_index = []
for i in np.arange(n_blocking):
    blocking_start.append(Blocking_season_total_date2[t][i][0])        
    for j in np.arange(len(Date_DJF)):
        if Blocking_season_total_date2[t][i][0] ==  Date_DJF[j]:
            blocking_start_index.append(j)
            break

## Find the blocking peaking date ##
blocking_peaking = []
blocking_peaking_index = []
blocking_peaking_SP = []
blocking_peaking_index_SP = []
    
for i in np.arange(n_blocking):
    LWA_B = np.zeros((len(Blocking_season_total_date2[t][i])))
    for j in np.arange(len(Blocking_season_total_date2[t][i])):
        LWA_B[j] = LWA[blocking_start_index[i]+j, int(180-Blocking_season_total_lat2[t][i][j]/(-0.5)), int( ( -180-Blocking_season_total_lon2[t][i][j])/(-0.625) ) ]
        
    for j in np.arange(len(Blocking_season_total_date2[t][i])):
        if LWA_B[j]==max(LWA_B):
            blocking_peaking.append(Blocking_season_total_date2[t][i][j])
            for k in np.arange(len(Date_DJF)):
                if Blocking_season_total_date2[t][i][j] ==  Date_DJF[k]:
                    blocking_peaking_index.append(k)
                    break
        
        if LWA_B[j]==max(LWA_B) and (Blocking_season_total_lon2[t][i][j]>160 or Blocking_season_total_lon2[t][i][j]<-90):
            blocking_peaking_SP.append(Blocking_season_total_date2[t][i][j])
            for k in np.arange(len(Date_DJF)):
                if Blocking_season_total_date2[t][i][j] ==  Date_DJF[k]:
                    blocking_peaking_index_SP.append(k)
                    break
n_blocking_SP = len(blocking_peaking_SP)
 

### Begin to Plot ###
duration = 3
startinglat = 140
fig = plt.figure(figsize=[12,3])
ax = fig.add_subplot(2,1,1)
plt.plot(np.arange(90), BI_DJF[t][:], '-k', linewidth=2)
for i in np.arange(len(BAM_event_DJF)):
    if BAM_event_DJF3[i]//90 ==t:
        BAM_date_index = BAM_event_DJF3[i]%90
        if BAM_date_index-duration<0:
            x = np.arange(0, BAM_date_index+duration+1)
            y = BI_DJF[t][0: BAM_date_index+duration+1]
        elif BAM_date_index+duration>=90:
            x = np.arange(BAM_date_index-duration, 90)
            y = BI_DJF[t][BAM_date_index-duration: 90]
        else:                                
            x = np.arange(BAM_date_index-duration, BAM_date_index+duration+1)
            y = BI_DJF[t][BAM_date_index-duration: BAM_date_index+duration+1]
                
        x = np.arange(90); y = BI_DJF[t][:]
        plt.fill_between(x, y, -2,
                 where = (x >= BAM_date_index-duration) & (x <= BAM_date_index+duration),
                 color = 'r',
                 alpha= 0.2)


for i in np.arange(n_blocking):        
    if np.isin(blocking_peaking_index[i], blocking_peaking_index_SP):            
        y = BI_DJF[t][blocking_peaking_index[i]%90]+0.6
        plt.arrow(blocking_peaking_index[i]%90, y, 0, -0.3, width=0.001, color='r', head_width=0.5,head_length=0.2)
        
    else:
        y = BI_DJF[t][blocking_peaking_index[i]%90]+0.6
        plt.arrow(blocking_peaking_index[i]%90, y, 0, -0.3, width=0.001, color='r', head_width=0.5,head_length=0.2)
        
plt.arrow(89,BI_DJF[t][-1]+1 ,0, -0.5, width=0.001, color='r', head_width=0.5,head_length=0.2)

# plt.plot(16,BI_DJF[t][16], color = "red", marker="o", markersize=3)
plt.plot(28,BI_DJF[t][28], color = "blue", marker="o", markersize=3)

plt.title("BAM cycle and Blocking Events ("+str(1980+t)+"-"+str(1980+t+1)+")", pad=10, fontsize=12)
plt.ylim(-3,3)
plt.ylabel('BAM index',fontsize=12)
plt.xticks([0,29,59,89])
ax.set_xticklabels(['12/01','12/30','1/29','2/28'])
ax.grid()
plt.text(13, 2, "Blocking A", fontdict={'family':'Times New Roman', 'size':6},color='red')
plt.text(25, -2, "Low BAM State", fontdict={'family':'Times New Roman', 'size':6},color='blue')
plt.text(40.5, 2.4, "Blocking B", fontdict={'family':'Times New Roman', 'size':6},color='red')




index1 = 11673
Blocking1_Z500 = np.zeros((nlat_SH,nlon))
Blocking1_LWA = np.zeros((nlat_SH,nlon))
file_Z = Dataset(path_Z[index1],'r')
file_LWA = Dataset(path_LWA[index1],'r')
Blocking1_Z500[:,0:nlon-1] = file_Z.variables['H'][0,0,0:180,:]
Blocking1_Z500[:,-1] = file_Z.variables['H'][0,0,0:180,0]
Blocking1_LWA[:,0:nlon-1] = file_LWA.variables['LWA_Z500'][0,0,0:180,:]
Blocking1_LWA[:,-1] = file_LWA.variables['LWA_Z500'][0,0,0:180,0]
file_Z.close()
file_LWA.close()

index2=11685
BAM_valley_Z500 = np.zeros((nlat_SH,nlon))
BAM_valley_LWA = np.zeros((nlat_SH,nlon))
file_Z = Dataset(path_Z[index2],'r')
file_LWA = Dataset(path_LWA[index2],'r')
BAM_valley_Z500[:,0:nlon-1] = file_Z.variables['H'][0,0,0:180,:]
BAM_valley_Z500[:,-1] = file_Z.variables['H'][0,0,0:180,0]
BAM_valley_LWA[:,0:nlon-1] = file_LWA.variables['LWA_Z500'][0,0,0:180,:]
BAM_valley_LWA[:,-1] = file_LWA.variables['LWA_Z500'][0,0,0:180,0]
file_Z.close()
file_LWA.close()

index3=11697
Blocking2_Z500 = np.zeros((nlat_SH,nlon))
Blocking2_LWA = np.zeros((nlat_SH,nlon))
file_Z = Dataset(path_Z[index3],'r')
file_LWA = Dataset(path_LWA[index3],'r')
Blocking2_Z500[:,0:nlon-1] = file_Z.variables['H'][0,0,0:180,:]
Blocking2_Z500[:,-1] = file_Z.variables['H'][0,0,0:180,0]
Blocking2_LWA[:,0:nlon-1] = file_LWA.variables['LWA_Z500'][0,0,0:180,:]
Blocking2_LWA[:,-1] = file_LWA.variables['LWA_Z500'][0,0,0:180,0]
file_Z.close()
file_LWA.close()


### Plot the Z500 and LWA at that peaking date ###
maxlev = Blocking2_Z500[0:startinglat,:].max()
minlev = Blocking2_Z500[0:startinglat,:].min()
levs_Z1 = np.linspace(minlev,maxlev, 15)

maxlev = Blocking2_LWA[0:startinglat,:].max()
minlev = Blocking2_LWA[0:startinglat,:].min()
levs_W1 = np.linspace(minlev,maxlev, 11)

bx = fig.add_subplot(2,3,4, projection=ccrs.PlateCarree(central_longitude=180))
h1 = plt.contour(lon,lat[0:startinglat], Blocking1_Z500[0:startinglat,:], levs_Z1, transform=ccrs.PlateCarree(), colors='k',linewidths=0.4) 
h2 = plt.contourf(lon,lat[0:startinglat], Blocking1_LWA[0:startinglat,:], levs_W1, transform=ccrs.PlateCarree(), cmap='YlOrRd' ,extend ='max') 
bx.set_extent([-180, -100, -90, -20], ccrs.PlateCarree())
bx.coastlines()
bx.set_xticks([0,90,180,270,358.5], crs=ccrs.PlateCarree())
bx.set_yticks([-80,-60,-40,], crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label='FALSE')
lat_formatter = LatitudeFormatter()
bx.xaxis.set_major_formatter(lon_formatter)
bx.yaxis.set_major_formatter(lat_formatter)  
plt.title("Blocking A",fontdict={'family':'Times New Roman', 'size':10}, color='red')
plt.plot([10, 10,80,80,10], [-80, -50,-50,-80,-80],
         'r--', linewidth=1, 
         transform=ccrs.PlateCarree())


cx = fig.add_subplot(2,3,5, projection=ccrs.PlateCarree(central_longitude=180))
h1 = plt.contour(lon,lat[0:startinglat], BAM_valley_Z500[0:startinglat:], levs_Z1, transform=ccrs.PlateCarree(), colors='k',linewidths=0.4) 
h2 = plt.contourf(lon,lat[0:startinglat], BAM_valley_LWA[0:startinglat,:], levs_W1, transform=ccrs.PlateCarree(), cmap='YlOrRd' ,extend ='max') 
cx.set_extent([-180, 180, -90, -20], ccrs.PlateCarree())
cx.coastlines()
cx.set_xticks([0,90,180,270,358.5], crs=ccrs.PlateCarree())
cx.set_yticks([-80,-60,-40], crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label='FALSE')
lat_formatter = LatitudeFormatter()
cx.xaxis.set_major_formatter(lon_formatter)
cx.yaxis.set_major_formatter(lat_formatter)  
plt.title("Low BAM State",fontdict={'family':'Times New Roman', 'size':10}, color='blue')


dx = fig.add_subplot(2,3,6, projection=ccrs.PlateCarree(central_longitude=180))
h1 = plt.contour(lon,lat[0:startinglat], Blocking2_Z500[0:startinglat,:], levs_Z1, transform=ccrs.PlateCarree(), colors='k',linewidths=0.4) 
h2 = plt.contourf(lon,lat[0:startinglat], Blocking2_LWA[0:startinglat,:], levs_W1, transform=ccrs.PlateCarree(), cmap='YlOrRd' ,extend ='max') 
dx.set_extent([-180, 180, -90, -20], ccrs.PlateCarree())
dx.coastlines()
dx.set_xticks([0,90,180,270,358.5], crs=ccrs.PlateCarree())
dx.set_yticks([-80,-60,-40], crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label='FALSE')
lat_formatter = LatitudeFormatter()
dx.xaxis.set_major_formatter(lon_formatter)
dx.yaxis.set_major_formatter(lat_formatter)  
plt.title("Blocking B",fontdict={'family':'Times New Roman', 'size':10}, color='red')
plt.plot([170, 170,210,210,170], [-75, -35,-35,-75,-65],
         'r--', linewidth=1.0, 
         transform=ccrs.PlateCarree())

plt.savefig("/home/liu3315/Research/BAM/BAM and Blocking/Figure1.png",dpi=600)





