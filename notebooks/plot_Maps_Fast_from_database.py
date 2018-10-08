
# coding: utf-8

# In[ ]:


'''

This code is part of the SIPN2 project focused on improving sub-seasonal to seasonal predictions of Arctic Sea Ice. 
If you use this code for a publication or presentation, please cite the reference in the README.md on the
main page (https://github.com/NicWayand/ESIO). 

Questions or comments should be addressed to nicway@uw.edu

Copyright (c) 2018 Nic Wayand

GNU General Public License v3.0


'''

'''
Plot forecast maps with all available models.
'''




import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict
import itertools
import numpy as np
import numpy.ma as ma
import pandas as pd
import struct
import os
import xarray as xr
import glob
import datetime
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import seaborn as sns
np.seterr(divide='ignore', invalid='ignore')
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import json
from esio import EsioData as ed
from esio import ice_plot
from esio import import_data
import subprocess
import dask
from dask.distributed import Client
import timeit

# General plotting settings
sns.set_style('whitegrid')
sns.set_context("talk", font_scale=.8, rc={"lines.linewidth": 2.5})


# In[ ]:


# # Set up local cluster for testing
# client = Client()
# client


# In[ ]:


def get_figure_init_times(fig_dir):
    # Get list of all figures
    fig_files = glob.glob(os.path.join(fig_dir,'*.png'))
    init_times = list(reversed(sorted(list(set([os.path.basename(x).split('_')[3] for x in fig_files])))))
    return init_times


# In[ ]:


def update_status(ds_status=None, fig_dir=None, int_2_days_dict=None, NweeksUpdate=3):
    # Get list of all figures
    fig_files = glob.glob(os.path.join(fig_dir,'*.png'))
    # For each figure
    for fig_f in fig_files:
        # Get the init_time from file name
        cit = os.path.basename(fig_f).split('_')[3]
        # Get the forecast int from file name
        cft = int(os.path.basename(fig_f).split('_')[4].split('.')[0])
        # Check if current it and ft were requested, otherwise skip
        if (np.datetime64(cit) in ds_status.init_time.values) & (np.timedelta64(int_2_days_dict[cft]) in ds_status.fore_time.values):
            # Always update the last 3 weeks (some models have lagg before we get them)
            # Check if cit is one of the last NweeksUpdate init times in init_time
            if (np.datetime64(cit) not in ds_status.init_time.values[-NweeksUpdate:]):
                ds_status.status.loc[dict(init_time=cit, fore_time=int_2_days_dict[cft])] = 1
        
    return ds_status


# In[ ]:


def Update_PanArctic_Maps():
    # Plotting Info
    runType = 'forecast'
    variables = ['sic']
    metrics_all = {'sic':['anomaly','mean','SIP'], 'hi':['mean']}
    updateAll = False
    # Exclude some models
    MME_NO = ['hcmr']

    # Define Init Periods here, spaced by 7 days (aprox a week)
    # Now
    cd = datetime.datetime.now()
    cd = datetime.datetime(cd.year, cd.month, cd.day) # Set hour min sec to 0. 
    # Hardcoded start date (makes incremental weeks always the same)
    start_t = datetime.datetime(1950, 1, 1) # datetime.datetime(1950, 1, 1)
    # Params for this plot
    Ndays = 7 # time period to aggregate maps to (default is 7)
    Npers = 5 # number of periods to plot (from current date) (default is 14)
    NweeksUpdate = 3 # Always update the most recent NweeksUpdate periods
    init_slice = np.arange(start_t, cd, datetime.timedelta(days=Ndays)).astype('datetime64[ns]')
    init_slice = init_slice[-Npers:] # Select only the last Npers of periods (weeks) since current date

    # Forecast times to plot
    weeks = pd.to_timedelta(np.arange(0,5,1), unit='W')
    months = pd.to_timedelta(np.arange(2,12,1), unit='M')
    years = pd.to_timedelta(np.arange(1,2), unit='Y') - np.timedelta64(1, 'D') # need 364 not 365
    slices = weeks.union(months).union(years).round('1d')
    da_slices = xr.DataArray(slices, dims=('fore_time'))
    da_slices.fore_time.values.astype('timedelta64[D]')
    print(da_slices)

    # Help conversion between "week/month" period used for figure naming and the actual forecast time delta value
    int_2_days_dict = dict(zip(np.arange(0,da_slices.size), da_slices.values))
    days_2_int_dict = {v: k for k, v in int_2_days_dict.items()}

    #############################################################
    # Load in Data
    #############################################################

    E = ed.EsioData.load()

    # add missing info for climatology
    E.model_color['climatology'] = (0,0,0)
    E.model_linestyle['climatology'] = '--'
    E.model_marker['climatology'] = '*'
    E.model['climatology'] = {'model_label':'Clim. Trend'}
    E.icePredicted['climatology'] = True

    mod_dir = E.model_dir

    # Get median ice edge by DOY
    median_ice_fill = xr.open_mfdataset(os.path.join(E.obs_dir, 'NSIDC_0051', 'agg_nc', 'ice_edge.nc')).sic
    # Get mean sic by DOY
    mean_1980_2010_sic = xr.open_dataset(os.path.join(E.obs_dir, 'NSIDC_0051', 'agg_nc', 'mean_1980_2010_sic.nc')).sic
    # Get average sip by DOY
    mean_1980_2010_SIP = xr.open_dataset(os.path.join(E.obs_dir, 'NSIDC_0051', 'agg_nc', 'hist_SIP_1980_2010.nc')).sip    

    # Get recent observations
    ds_81 = xr.open_mfdataset(E.obs['NSIDC_0081']['sipn_nc']+'_yearly/*.nc', concat_dim='time', autoclose=True, parallel=True)#,


    # Define models to plot
    models_2_plot = list(E.model.keys())
    models_2_plot = [x for x in models_2_plot if x not in ['piomas','MME','MME_NEW','uclsipn']] # remove some models
    models_2_plot = [x for x in models_2_plot if E.icePredicted[x]] # Only predictive models
    models_2_plot = ['MME']+models_2_plot # Add models to always plot at top
    models_2_plot.insert(1, models_2_plot.pop(-1)) # Move climatology from last to second

    # Get # of models and setup subplot dims
    Nmod = len(models_2_plot) + 1#(+3 for obs, MME, and clim)
    Nc = int(np.floor(np.sqrt(Nmod)))
    # Max number of columns == 5 (plots get too small otherwise)
    Nc = 5 #np.min([Nc,5])
    Nr = int(np.ceil(Nmod/Nc))
    print(Nr, Nc, Nmod)
    assert Nc*Nr>=Nmod, 'Need more subplots'


    for cvar in variables:

#         # Load in all metrics for given variable
#         ds_m = import_data.load_MME_by_init_end(E=E, runType=runType, variable=cvar, 
#                                     metrics=metrics_all[cvar], 
#                                     init_range=[init_slice[0],init_slice[-1]])

#         # Drop models that we don't evaluate (i.e. monthly means)
#         models_keep = [x for x in ds_m.model.values if x not in ['noaasipn','modcansipns_3','modcansipns_4']]
#         ds_m = ds_m.sel(model=models_keep)
#         # Get list of dynamical models that are not observations
#         dynamical_Models = [x for x in ds_m.model.values if x not in ['Observed','climatology','dampedAnomaly','dampedAnomalyTrend']]
#         # Get list of all models
#         all_Models = [x for x in ds_m.model.values if x not in ['Observed']]
#         # Add MME
#         MME_avg = ds_m.sel(model=dynamical_Models).mean(dim='model') # only take mean over dynamical models
#         MME_avg.coords['model'] = 'MME'
#         ds_ALL = xr.concat([ds_m, MME_avg], dim='model')
        
#         # Save to Zarr
#         ds_ALL.to_zarr('/home/disk/sipn/nicway/data/model/zarr/sic.zarr', mode='w')
#         xr.exit()

        ds_All = xr.open_zarr('/home/disk/sipn/nicway/data/model/zarr/sic.zarr')

        # Define fig dir and make if doesn't exist
        fig_dir = os.path.join(E.fig_dir, 'model', 'all_model', cvar, 'maps_weekly_NEW')
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

        # Make requested dataArray as specified above
        ds_status = xr.DataArray(np.ones((init_slice.size, da_slices.size))*np.NaN, 
                                 dims=('init_time','fore_time'), 
                                 coords={'init_time':init_slice,'fore_time':da_slices}) 
        ds_status.name = 'status'
        ds_status = ds_status.to_dataset()


        # Check what plots we already have
        if not updateAll:
            print("Removing figures we have already made")
            ds_status = update_status(ds_status=ds_status, fig_dir=fig_dir, int_2_days_dict=int_2_days_dict, NweeksUpdate=NweeksUpdate)

        
        print(ds_status.status.values)
        # Drop IC/FT we have already plotted (orthoginal only)
        ds_status = ds_status.where(ds_status.status.sum(dim='fore_time')<ds_status.fore_time.size, drop=True)

        print("Starting plots...")
        # For each init_time we haven't plotted yet
        
        for it in ds_status.init_time.values: 
            start_time_cmod = timeit.default_timer()
            print(it)
            it_start = it-np.timedelta64(Ndays,'D') + np.timedelta64(1,'D') # Start period for init period (it is end of period). Add 1 day because when
            # we select using slice(start,stop) it is inclusive of end points. So here we are defining the start of the init AND the start of the valid time.
            # So we need to add one day, so we don't double count. 

            # For each forecast time we haven't plotted yet
            ft_to_plot = ds_status.sel(init_time=it)
            ft_to_plot = ft_to_plot.where(ft_to_plot.isnull(), drop=True).fore_time

            for ft in ft_to_plot.values: 

                print(ft.astype('timedelta64[D]'))
                cs_str = format(days_2_int_dict[ft], '02') # Get index of current forcast week
                week_str = format(int(ft.astype('timedelta64[D]').astype('int')/Ndays) , '02') # Get string of current week
                cdoy_end = pd.to_datetime(it + ft).timetuple().tm_yday # Get current day of year end for valid time
                cdoy_start = pd.to_datetime(it_start + ft).timetuple().tm_yday  # Get current day of year end for valid time
                it_yr = str(pd.to_datetime(it).year) 
                it_m = str(pd.to_datetime(it).month)

                # Get datetime64 of valid time start and end
                valid_start = it_start + ft
                valid_end = it + ft

                # Loop through variable of interest + any metrics (i.e. SIP) based on that
                for metric in metrics_all[cvar]:

                    # Set up plotting info
                    if cvar=='sic':
                        if metric=='mean':
                            cmap_c = matplotlib.colors.ListedColormap(sns.color_palette("Blues_r", 10))
                            cmap_c.set_bad(color = 'lightgrey')
                            c_label = 'Sea Ice Concentration (-)'
                            c_vmin = 0
                            c_vmax = 1
                        elif metric=='SIP':
                            cmap_c = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white","orange","red","#990000"])
                            cmap_c.set_bad(color = 'lightgrey')
                            c_label = 'Sea Ice Probability (-)'
                            c_vmin = 0
                            c_vmax = 1
                        elif metric=='anomaly':
    #                         cmap_c = matplotlib.colors.ListedColormap(sns.color_palette("coolwarm", 9))
                            cmap_c = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","white","blue"])
                            cmap_c.set_bad(color = 'lightgrey')
                            c_label = 'SIC Anomaly to 1980-2010 Mean'
                            c_vmin = -1
                            c_vmax = 1

                    elif cvar=='hi':
                        if metric=='mean':
                            cmap_c = matplotlib.colors.ListedColormap(sns.color_palette("Reds_r", 10))
                            cmap_c.set_bad(color = 'lightgrey')
                            c_label = 'Sea Ice Thickness (m)'
                            c_vmin = 0
                            c_vmax = None
                    else:
                        raise ValueError("cvar not found.") 


                    # New Plot
                    #start_time_plot = timeit.default_timer()
                    (f, axes) = ice_plot.multi_polar_axis(ncols=Nc, nrows=Nr, Nplots=Nmod)


                    ############################################################################
                    #                               OBSERVATIONS                               #
                    ############################################################################

                    # Plot Obs (if available)
                    ax_num = 0
                    axes[ax_num].set_title('Observed')

                    try:
                        da_obs_c = ds_ALL[metric].sel(model='Observed',init_end=it, fore_time=ft)
                        haveObs = True
                    except:
                        haveObs = False

                    # If obs then plot
                    if haveObs:

                        da_obs_c.plot.pcolormesh(ax=axes[ax_num], x='lon', y='lat', 
                                              transform=ccrs.PlateCarree(),
                                              add_colorbar=False,
                                              cmap=cmap_c,
                                              vmin=c_vmin, vmax=c_vmax)
                        axes[ax_num].set_title('Observed')     
                    else: # When were in the future (or obs are missing)
                        if metric=='SIP': # Plot this historical mean SIP 
                            print("plotting hist obs SIP")
                            da_obs_c = mean_1980_2010_SIP.isel(time=slice(cdoy_start,cdoy_end)).mean(dim='time')
                            da_obs_c.plot.pcolormesh(ax=axes[ax_num], x='lon', y='lat', 
                              transform=ccrs.PlateCarree(),
                              add_colorbar=False,
                              cmap=cmap_c,
                              vmin=c_vmin, vmax=c_vmax)
                            axes[ax_num].set_title('Hist. Obs.')

                    ############################################################################
                    #                    Plot all models                                       #
                    ############################################################################

                    for (i, cmod) in enumerate(models_2_plot):
                        #print(cmod)
                        i = i+1 # shift for obs
                        axes[i].set_title(E.model[cmod]['model_label'])

                        # Select current model to plot
                        try:
                            ds_model = ds_ALL[metric].sel(model=cmod,init_end=it, fore_time=ft)
                            haveMod = True
                        except:
                            haveMod = False

                        # Plot
                        if haveMod:
                            p = ds_model.plot.pcolormesh(ax=axes[i], x='lon', y='lat', 
                                              transform=ccrs.PlateCarree(),
                                              add_colorbar=False,
                                              cmap=cmap_c,
                                              vmin=c_vmin, vmax=c_vmax)

                            axes[i].set_title(E.model[cmod]['model_label'])

                            # Clean up for current model
                            ds_model = None

                    # Make pretty
                    f.subplots_adjust(right=0.8)
                    cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
                    if p:
                        cbar = f.colorbar(p, cax=cbar_ax, label=c_label)
                        if metric=='anomaly':
                            cbar.set_ticks(np.arange(-1,1.1,0.2))
                        else:
                            cbar.set_ticks(np.arange(0,1.1,0.1))

                    # Set title of all plots
                    init_time_2 =  pd.to_datetime(it).strftime('%Y-%m-%d')
                    init_time_1 =  pd.to_datetime(it_start).strftime('%Y-%m-%d')
                    valid_time_2 = pd.to_datetime(it+ft).strftime('%Y-%m-%d')
                    valid_time_1 = pd.to_datetime(it_start+ft).strftime('%Y-%m-%d')
                    plt.suptitle('Initialization Time: '+init_time_1+' to '+init_time_2+'\n Valid Time: '+valid_time_1+' to '+valid_time_2,
                                 fontsize=15) # +'\n Week '+week_str
                    plt.subplots_adjust(top=0.85)

                    # Save to file
                    f_out = os.path.join(fig_dir,'panArctic_'+metric+'_'+runType+'_'+init_time_2+'_'+cs_str+'.png')
                    f.savefig(f_out,bbox_inches='tight', dpi=200)
                    print("saved ", f_out)
                    #print("Figure took  ", (timeit.default_timer() - start_time_plot)/60, " minutes.")
                    
                    # Mem clean up
                    plt.close(f)
                    da_obs_c = None

            # Done with current it
            print("Took ", (timeit.default_timer() - start_time_cmod)/60, " minutes.")


    # Update json file
    json_format = get_figure_init_times(fig_dir)
    json_dict = [{"date":cd,"label":cd} for cd in json_format]

    json_f = os.path.join(fig_dir, 'plotdates_current.json')
    with open(json_f, 'w') as outfile:
        json.dump(json_dict, outfile)

    # Make into Gifs
    # TODO: make parallel, add &
    for cit in json_format:
        subprocess.call(str("/home/disk/sipn/nicway/python/ESIO/scripts/makeGif.sh " + fig_dir + " " + cit), shell=True)

    print("Finished plotting panArctic Maps.")


# In[ ]:


if __name__ == '__main__':
    # Start up Client
    client = Client(n_workers=8)
#     dask.config.set(scheduler='threads')  # overwrite default with threaded scheduler
    
    # Call function
    Update_PanArctic_Maps()


# In[ ]:


# # Run below in case we need to just update the json file and gifs


# fig_dir = '/home/disk/sipn/nicway/public_html/sipn/figures/model/all_model/sic/maps_weekly'
# json_format = get_figure_init_times(fig_dir)
# json_dict = [{"date":cd,"label":cd} for cd in json_format]

# json_f = os.path.join(fig_dir, 'plotdates_current.json')
# with open(json_f, 'w') as outfile:
#     json.dump(json_dict, outfile)

# # Make into Gifs
# # TODO fig_dir hardcoded to current variable
# for cit in json_format:
#     subprocess.call(str("/home/disk/sipn/nicway/python/ESIO/scripts/makeGif.sh " + fig_dir + " " + cit), shell=True)

# print("Finished plotting panArctic Maps.")


