import itertools

import numpy as np
import seaborn as sns
from cartopy import crs as ccrs
from matplotlib import pyplot as plt
from . import import_data


def plot_reforecast(ds=None, axin=None, labelin=None, color='cycle_init_time',
                    marker=None, init_dot=True, init_dot_label=True, linestyle='-',
                    no_init_label=False, linewidth=1.5, alpha=1, fade_out=False):
    labeled = False
    if init_dot:
        init_label = 'Initialization'
    else:
        init_label = '_nolegend_'
    if no_init_label:
        init_label = '_nolegend_'

    if color=='cycle_init_time':
        cmap_c = itertools.cycle(sns.color_palette("GnBu_d", ds.init_time.size))
    elif color=='cycle_ensemble':
        cmap_c = itertools.cycle(sns.color_palette("GnBu_d", ds.ensemble.size))
    else:
        ccolor = color # Plot all lines with one color

    # Calc valid time if not already present
    if 'valid_time' not in ds.coords:
        ds = import_data.get_valid_time(ds) # ds.init_time + ds.fore_time

    for e in ds.ensemble:
            cds = ds.sel(ensemble=e)

            if color=='cycle_ensemble':
                ccolor = next(cmap_c)

            if fade_out:
                c_alpha = 0.1 # Starting alpha value (low becasue we plot forward in time)
                dt_alpha = 1/ds.init_time.size # increment to increase by
            else:
                c_alpha = alpha # Plot all with 1

            for it in ds.init_time:

                if labeled:
                    labelin = '_nolegend_'
                    init_label = '_nolegend_'

                if color=='cycle_init':
                    ccolor = next(cmap_c)

                x = cds.valid_time.sel(init_time=it).values
                y = cds.sel(init_time=it).values

                # Check we have data (issue with some models that change number of ensembles, when xarray merges, missing data is filled in (i.e. ukmo)
                # So check y is greater than 0 before plotting)
                if np.sum(y)==0:
                    continue

                # (optional) plot dot at initialization time
                if init_dot:
                    axin.plot(x[0], y[0], marker='*', linestyle='None', color='red', label=init_label)

                # Plot line
                axin.plot(x, y, label=labelin, color=ccolor, marker=marker, linestyle=linestyle,
                             linewidth=linewidth, alpha=c_alpha)
                if fade_out:
                    c_alpha = np.min([c_alpha + dt_alpha, 1])

                labeled = True
            cds = None


def plot_reforecast_bokeh(ds=None, plot_h=None, labelin=None, color='cycle_init_time',
                    marker=None, init_dot=True, init_dot_label=True, linestyle='-',
                    no_init_label=False, linewidth=1.5):
    labeled = False
    if init_dot:
        init_label = 'Initialization'
    else:
        init_label = '_nolegend_'
    if no_init_label:
        init_label = '_nolegend_'

    if color=='cycle_init_time':
        cmap_c = itertools.cycle(sns.color_palette("GnBu_d", ds.init_time.size))
    elif color=='cycle_ensemble':
        cmap_c = itertools.cycle(sns.color_palette("GnBu_d", ds.ensemble.size))
    else:
        ccolor = color # Plot all lines with one color

    for e in ds.ensemble:
            cds = ds.sel(ensemble=e)

            if color=='cycle_ensemble':
                ccolor = next(cmap_c)

            for it in ds.init_time:

                if labeled:
                    labelin = '_nolegend_'
                    init_label = '_nolegend_'

                if color=='cycle_init':
                    ccolor = next(cmap_c)

                x = (cds.fore_time + cds.init_time.sel(init_time=it)).values
                y = cds.sel(init_time=it).values

                # Check we have data (issue with some models that change number of ensembles, when xarray merges, missing data is filled in (i.e. ukmo)
                # So check y is greater than 0 before plotting)
                if np.sum(y)==0:
                    continue

                # (optional) plot dot at initialization time
                if init_dot:
                    plot_h.asterisk(x[0], y[0], color='red', legend=init_label)

                # Plot line
                plot_h.line(x, y, legend=labelin, color=ccolor,
                             line_width=linewidth)

                labeled = True
            cds = None


def polar_axis(extent=None, central_longitude=-45):
    '''cartopy geoaxes centered at north pole'''
    f = plt.figure(figsize=(6, 5))
    ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=central_longitude))
    ax.coastlines(linewidth=0.75, color='black', resolution='50m')
    ax.gridlines(crs=ccrs.PlateCarree(), linestyle='-')
    #ax.set_extent([0, 359.9, 57, 90], crs=ccrs.PlateCarree())
    # Full NSIDC extent
    if not extent:
        ax.set_extent([-3850000*0.9, 3725000*0.8, -5325000*0.7, 5850000*0.9], crs=ccrs.NorthPolarStereo(central_longitude=central_longitude))
    else: # Set Regional extent
        ax.set_extent(extent, crs=ccrs.NorthPolarStereo(central_longitude=central_longitude))
    return (f, ax)


def multi_polar_axis(ncols=4, nrows=4,
                     Nplots=None, sizefcter=1,
                     extent=None, central_longitude=-45):
    if not Nplots:
        Nplots = ncols*nrows
    # Create a grid of plots
    f, (axes) = plt.subplots(ncols=ncols, nrows=nrows, subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=central_longitude)})
    f.set_size_inches(ncols*1.5*sizefcter, nrows*2*sizefcter)
    axes = axes.reshape(-1)
    for (i, ax) in enumerate(axes):
        axes[i].coastlines(linewidth=0.2, color='black', resolution='50m')
        axes[i].gridlines(crs=ccrs.PlateCarree(), linestyle='--', linewidth=0.20, color='grey')
        #ax.set_extent([0, 359.9, 57, 90], crs=ccrs.PlateCarree())
        # Full NSIDC extent
        if not extent:
            axes[i].set_extent([-3850000*0.9, 3725000*0.8, -5325000*0.7, 5850000*0.9], crs=ccrs.NorthPolarStereo(central_longitude=central_longitude))
        else: # Set Regional extent
             axes[i].set_extent(extent, crs=ccrs.NorthPolarStereo(central_longitude=central_longitude))

        if i>=Nplots-1:
            f.delaxes(axes[i])
    return (f, axes)

def remove_small_contours(p, thres=10):
    ''' Removes small contours to clean up single contour plots'''
    for level in p.collections:
        for kp,path in reversed(list(enumerate(level.get_paths()))):
            # go in reversed order due to deletions!

            # include test for "smallness" of your choice here:
            # I'm using a simple estimation for the diameter based on the
            #    x and y diameter...
            verts = path.vertices # (N,2)-shape array of contour line coordinates
            diameter = np.max(verts.max(axis=0) - verts.min(axis=0))

            if diameter<thres: # threshold to be refined for your actual dimensions!
                del(level.get_paths()[kp])  # no remove() for Path objects:(
