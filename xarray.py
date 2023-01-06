import xarray as xr
import numpy as np
import catopy.crs as ccrs
ds = xr.open_dataset('pr_wtr.nc')
ds #Will show all details

ds1 = ds.sel(time='2022-01-15') #A particular data point
ds2 = ds.sel(time=slice("2001-01-15", "2001-12-15"), lon=slice(70, 90), lat=slice(0, 30)) #All values of 2001, over 10-30N and 70-90E


#Plotting 
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
ax.gridlines(draw_labels=True)
ds1.toa_net_all_mon.plot() #toa_net_all_mon is a variable
