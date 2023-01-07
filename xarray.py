import xarray as xr
import numpy as np
import catopy.crs as ccrs
ds = xr.open_dataset('pr_wtr.nc')
ds #Will show all details

ds1 = ds.sel(time='2022-01-15') #A particular data point
ds2 = ds.sel(time=slice("2001", "2001"), lon=slice(70, 90), lat=slice(10, 30)) #All values of 2001, over 10-30N and 70-90E
ds3 = ds.sel(time=slice("2001-01-01", "2001-03-01"), lon=slice(70, 90), lat=slice(10, 30)) #First three months


#Plotting 
deca = ds.pr_wtr.sel(time='1979', lat=slice(30,0), lon=slice(70,90)) #pr_wtr is the variable
ax = plt.axes(projection = ccrs.PlateCarree())
gls = ax.gridlines(draw_labels=True, color="none") #color='none' makes it invisible
gls.top_labels=False   # suppress top labels
gls.right_labels=False # suppress right labels
ax.coastlines()
deca2=deca[10,:,:].where(mask.topo>0)
deca2.plot(cmap='viridis', levels=20)


#Monthly plot for a year
ds1 = ds.sel(time=slice("2001", "2001"), lon=slice(70, 90), lat=slice(10, 30)) #All month values from 2001
ds3 = ds1.mean(dim=['lat','lon']).squeeze() #mean over lat and lon (spatial mean)
ds3.toa_net_all_mon.plot()
plt.ylabel('Qnet (W/m**2)')


#Selecting specific months
def func(month):
    return (month >= 4) & (month <= 6)

season = ds.sel(time=func(ds['time.month'])) #Works always

da_jja_only = ds.sel(time=ds.time.dt.month.isin([6, 7, 8])) #Works only if time is a datetime object
