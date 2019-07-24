"""
This script correlates the SPI3 and SPI12 with SDII, Rx1day and Rx5day globally.
All indices are based on REGEN v1 dataset.

Created on Wed Feb 8 2019
@author: david

"""
import numpy as np
import time
import os
import xarray as xr
import pandas as pd

start_time   = time.time()
spiTH = -1.0
#==============================================================================
def covariance(x, y, dim=None):
     valid_values = x.notnull() & y.notnull()
     valid_count = valid_values.sum(dim)
     demeaned_x = (x - x.mean(dim)).fillna(0)
     demeaned_y = (y - y.mean(dim)).fillna(0)
     return xr.dot(demeaned_x, demeaned_y, dims=dim) / valid_count

def correlation(x, y, dim=None):
     # dim should default to the intersection of x.dims and y.dims
     return covariance(x, y, dim) / (x.std(dim) * y.std(dim))

def correlate(file1, file2, var1, var2, droughtTH,season=None):
    print("Correlate {} and {}".format(var1,var2))

    name = "corr_{}_{}_droughtTH_{}_pval.nc".format(var1,var2,droughtTH)

    file1['time']=file2['time']
    varname_drought = "r_drought_{}".format(droughtTH)
    n = sum((~(np.ma.masked_invalid(file1[var1]).mask) & ~(np.ma.masked_invalid(file2[var2]).mask)))
    n_dr = file1[var1].where(n>=0.5*len(file1['time'])).where(file1[var1]<-1.5).count(dim='time')
    q_mask = np.where(n>=0.5*len(file1['time']),1,-1)

    corr        = correlation(file1[var1],file2[var2],dim='time')
    corr_drought = correlation(file1[var1].where(file1[var1]<spiTH),file2[var2].where(file1[var1]<spiTH),dim='time')

    # save whole correlations
    ds = xr.Dataset({'r': (['lat', 'lon'],  corr.where(q_mask)),varname_drought: (['lat', 'lon'], corr_drought.where(q_mask)),'q_mask': (['lat', 'lon'],  q_mask),'n': (['lat', 'lon'],  n),'n_dr': (['lat', 'lon'],  n_dr)},coords={'lon': file1['lon'],'lat': file1['lat']})

    return ds

def ttest(correl):
    from scipy.stats import t
    n = correl['n'].where(correl['n']>0)
    n_dr = correl['n_dr'].where(correl['n_dr']>0)

    r = correl['r'].where(correl['n']>0)
    r_dr = correl['r_drought_{}'.format(spiTH)].where(correl['n_dr']>0)

    tt = r*np.sqrt((n-2)/(1-r*r))
    tt_dr = r_dr*np.sqrt((n-2)/(1-r_dr*r_dr))

    p = r.copy()
    p_dr = r.copy()
    # p = t.sf(tt, n)
    for i in range(n.shape[0]):
        for j in range(n.shape[1]):
            p[i,j] = t.sf(tt[i,j], n[i,j])#*2  # two-sided pvalue?
            p_dr[i,j] = t.sf(tt_dr[i,j], n_dr[i,j])#*2  # two-sided pvalue?
    return p,p_dr


#==============================================================================
# correlations with REGEN_LongTermStns_v1
## directories
idir = "/short/w35/dh4185/RA_Ailie/data/REGEN_LongTermStns_v1/"
odir  = "/short/w35/dh4185/RA_Ailie/data/REGEN_LongTermStns_v1/corr/"
window = ("3","12")

for w in window:
    spifile = idir+"SPI"+w+"_prcptot_MON_REGENv1.1LongTerm_historical_NA_1950-2013.NaNmod.nc"

    var2 = ['SDII','W','Rx1day_mean','Rx1day_max','Rx5day_mean','Rx5day_max']
    var2file = ['sdii','sdii','rx1day','rx1day','rx5day','rx5day']

    for i,index in enumerate(var2):

        if w == '12':
            # correlate SPI and indices from var2
            name = 'corr_SPI{}_{}_LongTermStns_TH{}_pval.nc'.format(w,index,spiTH)

            if os.path.isfile(odir+name):
                print(name+" exists already")

            else:
                print("Correlation SPI{} and {} (REGEN_LongTermStns_v1)".format(w,index))

                file1 = xr.open_dataset(spifile).sel(time=slice('1951-01','2013-12'))
                file2 = xr.open_dataset("{}{}_{}MON_REGENv1.1LongTerm_historical_NA_1950-2013.nc".format(idir,var2file[i],w)).sel(time=slice('1951-01','2013-12'))

                correl = correlate(file1,file2,"SPI"+w,index,spiTH)
                pv,pv_dr = ttest(correl)

                correl['p_val']=pv
                correl['p_val_dr']=pv_dr

                correl.attrs['units']         = "correlation coefficient (r)"
                correl.attrs['description']   = "Pearson correlation of SPI{} and {} REGEN_LongTermStns_v1 ({}mon roll) 1950-2013. SPI{} drought threshold = {}".format(w,index,w,w,spiTH)
                ncfile                        = correl.to_netcdf(path=odir+name,mode='w')

            print("--- %s seconds ---" % (time.time() - start_time))

        elif w =='3':
            season  = ['Apr-Sep','Oct-Mar']

            for s, seas in enumerate(season):

                name = 'corr_SPI{}_{}_{}_LongTermStns_TH{}_pval.nc'.format(w,index,seas,spiTH)
                if os.path.isfile(odir+name):
                    print(name+" exists already")

                else:
                    tmp1 = xr.open_dataset(spifile).sel(time=slice('1950-03','2013-12'))
                    tmp2 = xr.open_dataset("{}{}_{}MON_REGENv1.1LongTerm_historical_NA_1950-2013.nc".format(idir,var2file[i],w)).sel(time=slice('1950-03','2013-12'))

                    if seas == 'Apr-Sep':
                        print("Correlation SPI{} and {} for {} (REGEN_LongTermStns_v1)".format(w,index,seas))
                        mask_seas = np.ma.masked_inside(tmp1['time.month'],4,9).mask
                    elif seas == 'Oct-Mar':
                        print("Correlation SPI{} and {} for {} (REGEN_LongTermStns_v1)".format(w,index,seas))
                        mask_seas = np.ma.masked_outside(tmp1['time.month'],4,9).mask

                    file1 = tmp1.where(tmp1['time'][mask_seas])
                    file2 = tmp2.where(tmp2['time'][mask_seas])

                    correl = correlate(file1,file2,"SPI"+w,index,spiTH)
                    pv,pv_dr = ttest(correl)

                    correl['p_val']=pv
                    correl['p_val_dr']=pv_dr

                    correl.attrs['units']         = "correlation coefficient (r)"
                    correl.attrs['description']   = "Pearson correlation of SPI{} and {} REGEN_LongTermStns_v1 ({}mon roll) 1950-2013 for {}. SPI{} drought threshold = {}".format(w,index,w,seas,w,spiTH)
                    ncfile                        = correl.to_netcdf(path=odir+name,mode='w')

                print("--- %s seconds ---" % (time.time() - start_time))

# #==============================================================================
# # correlations with REGEN_AllStns_v1
# ## directories
# idir = "/short/w35/dh4185/RA_Ailie/data/REGEN_AllStns_v1/"
# odir  = "/short/w35/dh4185/RA_Ailie/data/REGEN_AllStns_v1/corr/"
# window = ("3","12")
#
# for w in window:
#     spifile = idir+"SPI"+w+"_prcptot_MON_REGENv1.1ALL_historical_NA_1950-2013.NaNmod.nc"
#
#     var2 = ['SDII','W','Rx1day_mean','Rx1day_max','Rx5day_mean','Rx5day_max']
#     var2file = ['sdii','sdii','rx1day','rx1day','rx5day','rx5day']
#
#     for i,index in enumerate(var2):
#
#         if w == '12':
#             # correlate SPI and indices from var2
#             name = 'corr_SPI{}_{}_AllStns_TH{}.nc'.format(w,index,spiTH)
#
#             if os.path.isfile(odir+name):
#                 print(name+" exists already")
#
#             else:
#                 print("Correlation SPI{} and {} (REGEN_AllStns_v1)".format(w,index))
#
#                 file1 = xr.open_dataset(spifile).sel(time=slice('1951-01','2013-12'))
#                 file2 = xr.open_dataset("{}{}_{}MON_REGENv1.1ALL_historical_NA_1950-2013.nc".format(idir,var2file[i],w)).sel(time=slice('1951-01','2013-12'))
#
#                 correl = correlate(file1,file2,"SPI"+w,index,spiTH)
#                 pv,pv_dr = ttest(correl)
#
#                 correl['p_val']=pv
#                 correl['p_val_dr']=pv_dr
#
#                 correl.attrs['units']         = "correlation coefficient (r)"
#                 correl.attrs['description']   = "Pearson correlation of SPI{} and {} REGEN_AllStns_v1 ({}mon roll) 1950-2013. SPI{} drought threshold = {}".format(w,index,w,w,spiTH)
#                 ncfile                        = correl.to_netcdf(path=odir+name,mode='w')
#
#             print("--- %s seconds ---" % (time.time() - start_time))
#
#         elif w =='3':
#             season  = ['Apr-Sep','Oct-Mar']
#
#             for s, seas in enumerate(season):
#
#                 name = 'corr_SPI{}_{}_{}_AllStns_TH{}.nc'.format(w,index,seas,spiTH)
#                 if os.path.isfile(odir+name):
#                     print(name+" exists already")
#
#                 else:
#                     tmp1 = xr.open_dataset(spifile).sel(time=slice('1950-03','2013-12'))
#                     tmp2 = xr.open_dataset("{}{}_{}MON_REGENv1.1ALL_historical_NA_1950-2013.nc".format(idir,var2file[i],w)).sel(time=slice('1950-03','2013-12'))
#
#                     if seas == 'Apr-Sep':
#                         print("Correlation SPI{} and {} for {} (REGEN_AllStns_v1)".format(w,index,seas))
#                         mask_seas = np.ma.masked_inside(tmp1['time.month'],4,9).mask
#                     elif seas == 'Oct-Mar':
#                         print("Correlation SPI{} and {} for {} (REGEN_AllStns_v1)".format(w,index,seas))
#                         mask_seas = np.ma.masked_outside(tmp1['time.month'],4,9).mask
#
#                     file1 = tmp1.where(tmp1['time'][mask_seas])
#                     file2 = tmp2.where(tmp2['time'][mask_seas])
#
#                     correl = correlate(file1,file2,"SPI"+w,index,spiTH)
#                     pv,pv_dr = ttest(correl)
#
#                     correl['p_val']=pv
#                     correl['p_val_dr']=pv_dr
#
#                     correl.attrs['units']         = "correlation coefficient (r)"
#                     correl.attrs['description']   = "Pearson correlation of SPI{} and {} REGEN_AllStns_v1 ({}mon roll) 1950-2013 for {}. SPI{} drought threshold = {}".format(w,index,w,seas,w,spiTH)
#                     ncfile                        = correl.to_netcdf(path=odir+name,mode='w')
#
#                 print("--- %s seconds ---" % (time.time() - start_time))
