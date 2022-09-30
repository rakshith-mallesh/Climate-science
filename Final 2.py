import pandas as pd
import numpy as np
import netCDF4
import matplotlib.pyplot as plt
from scipy import signal

#NCEP lat and lon bounds for 10-30 N and 70-90 E
#Lat = [24:33]
#Lon = [28:37]

#ERA lat and lon bounds for 10-30 N and 70-90 E
#Lat = [240:320]
#Lon = [280:360]

#print(f1.variables.keys()) Gives the dictionary keys of the variables in the file
#print(f1['sst']) Gives details about the variable
#print(f1['longitude'][280:360]) Prints the array

## Conversions
# kg/m2/s to W/m2: *86400*28.96
# kg/m2/s to mm/month: *86400
# kg/m2/s to mm/day: *86400*number of days in month



#Latitude and Longitude bounds 
latbounds = [15, 25]
lonbounds = [75, 85]
lats = f1.variables['latitude'][:]
lons = f1.variables['longitude'][:]

# latitude lower and upper index
latli = np.argmin( np.abs( lats - latbounds[0] ) )
latui = np.argmin( np.abs( lats - latbounds[1] ) ) 

# longitude lower and upper index
lonli = np.argmin( np.abs( lons - lonbounds[0] ) )
lonui = np.argmin( np.abs( lons - lonbounds[1] ) ) 
print(latli, latui, lonli, lonui)





# Nino 3.4, ISMR correlation from ERA5
a = 'SST, Prec.nc'
f1 = netCDF4.Dataset(a)

#rainfall anmoly
rain = []
for i in range(0, 252, 4):
	j = f1['mtpr'][i, 240:320, 280:360]
	k = f1['mtpr'][i+1, 240:320, 280:360]
	l = f1['mtpr'][i+2, 240:320, 280:360]
	m = f1['mtpr'][i+3, 240:320, 280:360]
	rain.append(86400*np.average(j+k+l+m)/4)

rainan = []
for i in range(len(rain)):
	rainan.append(np.average(rain)-rain[i])

sstindia = []
for i in range(0, 252, 4):
	j = f1['sst'][i, 280:440, 200:440]
	k = f1['sst'][i+1, 280:440, 200:440]
	l = f1['sst'][i+2, 280:440, 200:440]
	m = f1['sst'][i+3, 280:440, 200:440]
	sstindia.append(np.average(j+k+l+m)/4)

sstindiaan = []
for i in range(len(sstindia)):
	sstindiaan.append(np.average(sstindia)-sstindia[i])

sstnino = []
for i in range(0, 252, 4):
	j = f1['sst'][i, 340:380, 960:1160]
	k = f1['sst'][i+1, 340:380, 960:1160]
	l = f1['sst'][i+2, 340:380, 960:1160]
	m = f1['sst'][i+3, 340:380, 960:1160]
	sstnino.append(np.average(j+k+l+m)/4)

sstninoan = []
for i in range(len(sstnino)):
	sstninoan.append(np.average(sstnino)-sstnino[i])
    
rainan = signal.detrend(rainan, axis=- 1, type='linear', bp=0, overwrite_data=False)
sstninoan = signal.detrend(sstninoan, axis=- 1, type='linear', bp=0, overwrite_data=False)
sstindiaan = signal.detrend(sstindiaan, axis=- 1, type='linear', bp=0, overwrite_data=False)

prainan = []
psstninoan = []
for i in range(20, len(sstnino)):
	if sstindiaan[i]<0:
		prainan.append(rainan[i])
		psstninoan.append(sstninoan[i])

plt.scatter(psstninoan, prainan, c = 'teal', label = '+ TIO, ERA5')
plt.xlabel("Nino 3.4 SST anomaly")
plt.ylabel("ISMR anomaly (mm/ day)")
plt.legend()
plt.show()

print(np.corrcoef(psstninoan, prainan))









# Nino 3.4, ISMR correlation from IMD, NOAA data
data = pd.read_csv("Precipitation 2.csv")
rain = data['JJAS AVG'].values
rain = rain[78:]
rainan =[]
for i in range(len(rain)):
    rainan.append((rain[i]/30.5) - (np.average(rain)/30.5))

b = 'ERSST.nc'
f2 = netCDF4.Dataset(b)

sstnino = []
for i in range(1505, 1932, 12):
    j = f2['sst'][i, 42:47, 120:146]
    k = f2['sst'][i+1, 42:47, 120:146]
    l = f2['sst'][i+2, 42:47, 120:146]
    m = f2['sst'][i+3, 42:47, 120:146]
    sstnino.append((np.average(j+k+l+m))/4)

sstninoan = []
for i in range(len(sstnino)):
    sstninoan.append(sstnino[i]-np.average(sstnino)) 
    
sstindia = []
for i in range(1505, 1932, 12):
    j = f2['sst'][i, 34:55, 25:56]
    k = f2['sst'][i+1, 34:55, 25:56]
    l = f2['sst'][i+2, 34:55, 25:56]
    m = f2['sst'][i+3, 34:55, 25:56]
    sstindia.append((np.average(j+k+l+m))/4)

sstindiaan = []
for i in range(len(sstindia)):
    sstindiaan.append(sstindia[i]-np.average(sstindia))
    
rainan = signal.detrend(rainan, axis=- 1, type='linear', bp=0, overwrite_data=False)
sstninoan = signal.detrend(sstninoan, axis=- 1, type='linear', bp=0, overwrite_data=False)
sstindiaan = signal.detrend(sstindiaan, axis=- 1, type='linear', bp=0, overwrite_data=False)

prainan = []
psstninoan = []
for i in range(0, len(sstnino)):
    if sstindiaan[i]>0:
        prainan.append(rainan[i])
        psstninoan.append(sstninoan[i])
        
nrainan = []
nsstninoan = []
for i in range(0, len(sstnino)):
    if sstindiaan[i]<0:
        nrainan.append(rainan[i])
        nsstninoan.append(sstninoan[i])

plt.scatter(psstninoan, prainan, c = 'darkred', label = "Positive TIO")
plt.xlabel("Nino 3.4 SST anomaly")
plt.ylabel("ISMR anomaly (mm/day)")
plt.savefig('filename.png', dpi=400, bbox_inches="tight")
plt.legend()
plt.show()

plt.scatter(nsstninoan, nrainan, c = 'c', label = "Negative TIO")
plt.xlabel("Nino 3.4 SST anomaly")
plt.ylabel("ISMR anomaly (mm/day)")
plt.legend()
plt.show()

print(np.corrcoef(psstninoan, prainan))
print(np.corrcoef(nsstninoan, nrainan))
