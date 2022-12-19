import xarray as xr
ds = xr.open_dataset('specific_humidity.nc')

print(ds.dims)
print(ds.coords)

print(ds['time'].values)

q = ds['q'].sel(time=f'{year}').values
