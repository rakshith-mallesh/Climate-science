# Some python packages for atmospheric sciences:
# Metpy
# Cartopy
# Xarray
# pyNGL and pyNIO
# geopandas 
# basemap
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


#constants
g = 9.80616
Cp = 1004.64
Lv = 2.501e6

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








#Comparing ERA and NCEP Pwat for a few years
c = #tcwv dataset here 
d = 'land sea ERA.nc'
f3 = netCDF4.Dataset(c)
f4 = netCDF4.Dataset(d)
pwat1979 = []
pwat1980 = []
pwat1982 = []
pwat1983 = []
pwat1987 = []
pwat1988 = []
pwat2002 = []
pwat2003 = []

for i in range(0, 365):
    v1 = f3['tcwv'][i, 20:61, 20:61]
    v2 = f3['tcwv'][i+365, 20:61, 20:61]
    v3 = f3['tcwv'][i+731, 20:61, 20:61]
    v4 = f3['tcwv'][i+1096, 20:61, 20:61]
    v5 = f3['tcwv'][i+1461, 20:61, 20:61]
    v6 = f3['tcwv'][i+1826, 20:61, 20:61]
    v7 = f3['tcwv'][i+2192, 20:61, 20:61]
    v8 = f3['tcwv'][i+2557, 20:61, 20:61]
    vl = f4['lsm'][0, 20:61, 20:61]
    x1 = np.multiply(vl,v1)
    x2 = np.multiply(vl,v2)
    x3 = np.multiply(vl,v3)
    x4 = np.multiply(vl,v4)
    x5 = np.multiply(vl,v5)
    x6 = np.multiply(vl,v6)
    x7 = np.multiply(vl,v7)
    x8 = np.multiply(vl,v8)
    pwat1979.append(np.average(x1[np.nonzero(x1)]))
    pwat1980.append(np.average(x2[np.nonzero(x2)]))
    pwat1982.append(np.average(x3[np.nonzero(x3)]))
    pwat1983.append(np.average(x4[np.nonzero(x4)]))
    pwat1987.append(np.average(x5[np.nonzero(x5)]))
    pwat1988.append(np.average(x6[np.nonzero(x6)]))
    pwat2002.append(np.average(x7[np.nonzero(x7)]))
    pwat2003.append(np.average(x8[np.nonzero(x8)]))

a = 'pr_wtr.eatm.1979.nc'
b = 'pr_wtr.eatm.1980.nc'
c = 'pr_wtr.eatm.1982.nc'
d = 'pr_wtr.eatm.1983.nc'
e = 'pr_wtr.eatm.1987.nc'
f = 'pr_wtr.eatm.1988.nc'
g = 'pr_wtr.eatm.2002.nc'
h = 'pr_wtr.eatm.2003.nc'
i = 'land sea.nc'
f1 = netCDF4.Dataset(a)
f2 = netCDF4.Dataset(b)
f3 = netCDF4.Dataset(c)
f4 = netCDF4.Dataset(d)
f5 = netCDF4.Dataset(e)
f6 = netCDF4.Dataset(f)
f7 = netCDF4.Dataset(g)
f8 = netCDF4.Dataset(h)
f9 = netCDF4.Dataset(i)

P1=[]
P2=[]
P3=[]
P4=[]
P5=[]
P6=[]
P7=[]
P8=[]

for i in range(0, 365):
	v1 = f1['pr_wtr'][i, 26:31, 30:35]
	v2 = f2['pr_wtr'][i, 26:31, 30:35]
	v3 = f3['pr_wtr'][i, 26:31, 30:35]
	v4 = f4['pr_wtr'][i, 26:31, 30:35]
	v5 = f5['pr_wtr'][i, 26:31, 30:35]
	v6 = f6['pr_wtr'][i, 26:31, 30:35]
	v7 = f7['pr_wtr'][i, 26:31, 30:35]
	v8 = f8['pr_wtr'][i, 26:31, 30:35]
	vl = f9['land'][0, 26:31, 30:35]
	x1 = np.multiply(vl,v1)
	x2 = np.multiply(vl,v2)
	x3 = np.multiply(vl,v3)
	x4 = np.multiply(vl,v4)
	x5 = np.multiply(vl,v5)
	x6 = np.multiply(vl,v6)
	x7 = np.multiply(vl,v7)
	x8 = np.multiply(vl,v8)
	P1.append(np.average(x1[np.nonzero(x1)]))
	P2.append(np.average(x2[np.nonzero(x2)]))
	P3.append(np.average(x3[np.nonzero(x3)]))
	P4.append(np.average(x4[np.nonzero(x4)]))
	P5.append(np.average(x5[np.nonzero(x5)]))
	P6.append(np.average(x6[np.nonzero(x6)]))
	P7.append(np.average(x7[np.nonzero(x7)]))
	P8.append(np.average(x8[np.nonzero(x8)]))

    
X = np.linspace(1, 365, 365, dtype='int')
 
plt.subplot(4, 2, 1)   
plt.plot(X, pwat1979, c = 'g', linewidth=0.4)
plt.plot(X, P1, c = 'r', linewidth=0.4)
plt.title('Pwat for 1979', fontsize=8)

plt.subplot(4, 2, 2)   
plt.plot(X, pwat1980, c = 'g', linewidth=0.4)
plt.plot(X, P2, c = 'r', linewidth=0.4)
plt.title('Pwat for 1980', fontsize=8)

plt.subplot(4, 2, 3)   
plt.plot(X, pwat1982, c = 'g', linewidth=0.4)
plt.plot(X, P3, c = 'r', linewidth=0.4)
plt.ylabel('Precipitable water content (Kg/m^2)                         ')
plt.title('Pwat for 1982', fontsize=8)

plt.subplot(4, 2, 4)   
plt.plot(X, pwat1983, c = 'g', linewidth=0.4)
plt.plot(X, P4, c = 'r', linewidth=0.4)
plt.title('Pwat for 1983', fontsize=8)

plt.subplot(4, 2, 5)   
plt.plot(X, pwat1987, c = 'g', linewidth=0.4)
plt.plot(X, P5, c = 'r', linewidth=0.4)
plt.title('Pwat for 1987', fontsize=8)

plt.subplot(4, 2, 6)   
plt.plot(X, pwat1988, c = 'g', linewidth=0.4)
plt.plot(X, P6, c = 'r', linewidth=0.4)
plt.title('Pwat for 1988', fontsize=8)

plt.subplot(4, 2, 7)   
plt.plot(X, pwat2002, c = 'g', linewidth=0.4)
plt.plot(X, P7, c = 'r', linewidth=0.4)
plt.xlabel('Day of the year')
plt.title('Pwat for 2002', fontsize=8)

plt.subplot(4, 2, 8)   
plt.plot(X, pwat2003, c = 'g', label = 'ERA5', linewidth=0.4)
plt.plot(X, P8, c = 'r', label = 'NCEP', linewidth=0.4)
plt.xlabel('Day of the year')
plt.title('Pwat for 2003', fontsize=8)
plt.legend(bbox_to_anchor = (1, 1))
plt.tight_layout()
plt.savefig('Pwat.png', dpi=1200, bbox_inches="tight")
plt.show()



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










#JJAS TGMS comparison for a particular year NCEP and ERA5

a = "land sea.nc"
b = "dswrf.ntat.nc"
c = "uswrf.ntat.nc"
d = "ulwrf.ntat.nc"
e = "lhtfl.nc" #this is the evaporation rate
f = "prate.nc"

f1 = netCDF4.Dataset(a)
f2 = netCDF4.Dataset(b)
f3 = netCDF4.Dataset(c)
f4 = netCDF4.Dataset(d)
f5 = netCDF4.Dataset(e)
f6 = netCDF4.Dataset(f)

QNCEP = []
PNCEP = []

year = int(input('What is the year you want to do the comparison for?\n'))
j=(year-1979)*12+5
for i in range(j,j+4):
		v1 = f1['land'][0, 31:47, 38:49]
		v2 = f2['dswrf'][i, 31:47, 38:49]
		v3 = f3['uswrf'][i, 31:47, 38:49]
		v4 = f4['ulwrf'][i, 31:47, 38:49]
		v5 = f5['lhtfl'][i, 31:47, 38:49]
		v6 = f6['prate'][i, 31:47, 38:49]
		x = np.multiply(v1, 86400*28.94*v6-v5)
		y = np.multiply(v1, v2 - v3 - v4)
		PNCEP.append(np.average(x[np.nonzero(x)]))
		QNCEP.append(np.average(y[np.nonzero(y)]))

#print(86400*28.94*v6, v5)

TGMSNCEP = []
for i in range(len(QNCEP)):
    TGMSNCEP.append(QNCEP[i]/PNCEP[i])

g = 'land sea ERA.nc'
h = 'TGMS ERA 1979-2022, India only.nc'
f7 = netCDF4.Dataset(g)
f8 = netCDF4.Dataset(h)

QERA = []
PERA = []


j=(year-1979)*4
for i in range(j, j+4):
    v1 = f7['lsm'][0, :, :]
    v2 = f8['mtnswrf'][i, :, :]
    v3 = f8['mtnlwrf'][i, :, :]
    v4 = f8['mtpr'][i, :, :]
    v5 = f8['mer'][i, :, :]
    x = np.multiply(v1, 86400*28.94*(v4+v5))
    y = np.multiply(v1, v2+v3)
    PERA.append(np.average(x[np.nonzero(x)]))
    QERA.append(np.average(y[np.nonzero(y)]))

TGMSERA = []
for i in range(len(QERA)):
    TGMSERA.append(QERA[i]/PERA[i])

plt.style.use('bmh') 
X = ['June', 'July', 'August', 'September']
plt.subplot(3, 1, 1)
plt.plot(X, QNCEP, c='b')
plt.plot(X, QERA, c='tab:orange')
plt.ylabel("Qnet (W/m^2)")
plt.title('Year: '+ str(year))

plt.subplot(3,1,2)
plt.plot(X, PNCEP, c='b')
plt.plot(X, PERA, c='tab:orange')
plt.ylabel("P-E (W/m^2)")

plt.subplot(3,1,3)
plt.plot(X, TGMSNCEP)
plt.plot(X, TGMSERA, c='tab:orange')
plt.ylabel("TGMS")
plt.legend(['NCEP', 'ERA'])
plt.tight_layout()
#plt.savefig('TGMS '+str(year)+'.png', dpi=150, bbox_inches="tight")
plt.show()








#Comparision of ERA, NCEP Qnet with CERES for a particular year
k = 'CERES.nc'
f9 = netCDF4.Dataset(k)

QNCEP = []
PNCEP = []

initial = 1979
year = int(input('What is the year you want to do the comparison for? (Only 2000-2021 for CERES)\n'))
j=(year-initial)*12+5

for i in range(j,j+4):
		v1 = f1['land'][0, 31:47, 38:49]
		v2 = f2['dswrf'][i, 31:47, 38:49]
		v3 = f3['uswrf'][i, 31:47, 38:49]
		v4 = f4['ulwrf'][i, 31:47, 38:49]
		v5 = f5['lhtfl'][i, 31:47, 38:49]
		v6 = f6['prate'][i, 31:47, 38:49]
		x = np.multiply(v1, 86400*28.94*v6-v5)
		y = np.multiply(v1, v2 - v3 - v4)
		QNCEP.append(np.average(v2-v3-v4))

#print(86400*28.94*v6, v5)

QERA = []
PERA = []


j=(year-initial)*4
for i in range(j, j+4):
    v1 = f7['lsm'][0, :, :]
    v2 = f8['mtnswrf'][i, :, :]
    v3 = f8['mtnlwrf'][i, :, :]
    v4 = f8['mtpr'][i, :, :]
    v5 = f8['mer'][i, :, :]
    x = np.multiply(v1, 86400*28.94*(v4+v5))
    y = np.multiply(v1, v2+v3)
    QERA.append(np.average(v2+v3))

initial = 2000
j=3+(year-initial)*12

QCERES=[]
for i in range(j, j+4):
	QCERES.append(np.average(f9['toa_net_all_mon'][i, 100:120, 70:90]))

plt.style.use('bmh') 
X = ['June', 'July', 'August', 'September']

plt.plot(X, QNCEP, c='b')
plt.plot(X, QERA, c='tab:orange')
if year>=2000:
	plt.plot(X, QCERES, c='g')
plt.ylabel("Qnet (W/m^2)")
plt.title('Year: '+ str(year))
plt.legend(['NCEP', 'ERA', 'CERES'])
#plt.savefig('TGMS '+str(year)+'.png', dpi=150, bbox_inches="tight")
plt.show()








#JJAS comparison of Qnet from CERES, ERA5, NCEP, 2000-2021
year1=2000
year2=2022 #It won't include this year's values
QNCEP = []
PNCEP = []

step = 12
in1 = (year1 - 1979)*step+5
in2 = (year2 - 1979)*step+5
for i in range(in1, in2, step):
		v1 = f1['land'][0, 31:47, 38:49]
		v2 = f2['dswrf'][i, 31:47, 38:49] + f2['dswrf'][i+1, 31:47, 38:49] + f2['dswrf'][i+2, 31:47, 38:49] + f2['dswrf'][i+3, 31:47, 38:49] 
		v3 = f3['uswrf'][i, 31:47, 38:49] + f3['uswrf'][i+1, 31:47, 38:49] + f3['uswrf'][i+2, 31:47, 38:49] + f3['uswrf'][i+3, 31:47, 38:49]
		v4 = f4['ulwrf'][i, 31:47, 38:49] + f4['ulwrf'][i+1, 31:47, 38:49] + f4['ulwrf'][i+2, 31:47, 38:49] + f4['ulwrf'][i+3, 31:47, 38:49] 
		y = np.multiply(v1, v2 - v3 - v4)
		QNCEP.append(np.average(v2 - v3 - v4)/4)

QERA = []
PERA = []

step = 4
in1 = (year1 - 1979)*step
in2 = (year2 - 1979)*step
for i in range(in1, in2, step):
    v1 = f7['lsm'][0, :, :]
    v2 = f8['mtnswrf'][i, :, :] + f8['mtnswrf'][i+1, :, :] + f8['mtnswrf'][i+2, :, :] + f8['mtnswrf'][i+3, :, :]
    v3 = f8['mtnlwrf'][i, :, :] + f8['mtnlwrf'][i+1, :, :] + f8['mtnlwrf'][i+2, :, :] + f8['mtnlwrf'][i+3, :, :]
    y = np.multiply(v1, v2+v3)
    QERA.append(np.average(v2+v3)/4)

QCERES=[]

step = 12
in1 = (year1 - 2000)*step + 3
in2 = (year2 - 2000)*step + 3
for i in range(in1, in2, step):
	QCERES.append(np.average(f9['toa_net_all_mon'][i, 100:120, 70:90]+f9['toa_net_all_mon'][i+1, 100:120, 70:90]+f9['toa_net_all_mon'][i+2, 100:120, 70:90]+f9['toa_net_all_mon'][i+3, 100:120, 70:90])/4)

X = np.linspace(2000, 2021, 22)
plt.style.use('bmh')

plt.title("JJAS average comparison for 2000-2021 over 10-30N, 70-90E")
plt.plot(X, QERA, c='tab:orange')
plt.plot(X, QNCEP, c='b')
plt.plot(X, QCERES, c='g')
plt.legend(['ERA', 'NCEP', 'CERES'])
plt.ylabel('Qnet (W/m^2)')
plt.show()





#JJAS TGMS using CERES Qnet and ERA P-E

PERA = []

year1=2000
year2=2022
step = 4
in1 = (year1 - 1979)*step
in2 = (year2 - 1979)*step
for i in range(in1, in2, step):
	v1 = f7['lsm'][0, :, :]
	v4 = f8['mtpr'][i, :, :] + f8['mtpr'][i+1, :, :] + f8['mtpr'][i+2, :, :] + f8['mtpr'][i+3, :, :]
	v5 = f8['mer'][i, :, :] + f8['mer'][i+1, :, :] + f8['mer'][i+2, :, :] + f8['mer'][i+3, :, :]
	x = np.multiply(v1, 86400*28.94*(v4+v5))
	PERA.append(np.average(x[np.nonzero(x)])/4)

QCERES=[]

step = 12
in1 = (year1 - 2000)*step + 3
in2 = (year2 - 2000)*step + 3
for i in range(in1, in2, step):
	v1 = f10['topo'][100:120, 70:90]
	v1[v1!=1] = 0
	v2 = f9['toa_net_all_mon'][i, 100:120, 70:90]+f9['toa_net_all_mon'][i+1, 100:120, 70:90]+f9['toa_net_all_mon'][i+2, 100:120, 70:90]+f9['toa_net_all_mon'][i+3, 100:120, 70:90]
	y = np.multiply(v1, v2)
	QCERES.append(np.average(y[np.nonzero(y)])/4)

TGMS = []
for i in range(len(PERA)):
    TGMS.append(QCERES[i]/PERA[i])

X = np.linspace(2000, 2021, 22)
plt.style.use('bmh')

plt.subplot(3,1,1)
plt.plot(X, PERA, c='tab:orange')
plt.title("JJAS average comparison for 2000-2021 over 10-30N, 70-90E")
plt.ylabel('P-E (W/m^2)')
plt.legend('ERA')

plt.subplot(3,1,2)
plt.plot(X, QCERES, c='g')
plt.ylabel('Qnet (W/m^2)')
plt.legend('CERES')

plt.subplot(3,1,3)
plt.plot(X, TGMS, c='b')
plt.ylabel('TGMS')
plt.show()







#calculating relative contributions from TGMS and Qnet for P-E
Pint = PERA
Qint = QCERES
Gint = TGMS

print(np.max(Pint), np.argmax(Pint))
print(np.min(Pint), np.argmin(Pint))


Q = np.average(Qint)
P = np.average(Pint)
G = Q/P

#yearly contribution
dQ = []
dG = []
for i in range(len(Pint)):
	dQ.append(Qint[i]-Q)
	dG.append(Gint[i]-G)


dP = []
Qcont = []
Tcont = []
for i in range(len(Pint)):
	Qcont.append((dQ[i]*Pint[i]/Q)/(1+(dG[i]/G)))
	Tcont.append((-dG[i]*Pint[i]/G)/(1+(dG[i]/G)))
	dP.append(Pint[i]-P)
	

sum = np.add(Tcont, Qcont).tolist()

arr = np.array(dP)
ind = X # the x locations for the groups
width = 0.35       # the width of the bars

plt.bar(ind, dP, width)
plt.bar(ind+width, Tcont, width, color='seagreen')
plt.bar(ind+width, Qcont, width, color='tab:orange')
plt.xticks(X, X)        #shows all the x labels
plt.xlabel('Year')
plt.ylabel('W/m^2')
plt.title('Contributions from Qnet and TGMS')
plt.legend(['Change in P', 'Contribution from TGMS', 'Contribution from Qnet']) 
plt.show()




#Decadal contribution for 2001-2020


Q1 = np.average(Qint[1:11])
Q2 = np.average(Qint[11:21])
P1 = np.average(Pint[1:11])
P2 = np.average(Pint[11:21])
G1 = Q1/P1
G2 = Q2/P2

Q = np.average(Qint)
P = np.average(Pint)
G = Q/P

#yearly contribution
dQ = Q2-Q1
dG = G2-G1

dP = P2-P1
Qcont = []
Tcont = []
Qcont.append((dQ*P/Q)/(1+(dG/G)))
Tcont.append((-dG*P/G)/(1+(dG/G)))

sum = np.add(Tcont, Qcont).tolist()
ind = np.arange(1)
width = 0.1       # the width of the bars

plt.style.use('bmh')
plt.bar(ind, dP, width)
plt.bar(ind+width, Tcont, width, color='seagreen')
plt.bar(ind+width, Qcont, width, color='tab:orange')
plt.xlabel('2001-2020')
plt.ylabel('W/m^2')
plt.title('Contributions from Qnet and TGMS (Decadal scale)')
plt.legend(['Change in P', 'Contribution from TGMS', 'Contribution from Qnet'], bbox_to_anchor = (1, 1)) 
plt.show()






#Holocene TGMS vs Pwat
data = pd.read_csv('Landsat.csv') # 22040 rows and 5 columns
# index =  22000 - how many years back 
df_3 = data.iloc[12000:22000,:] #Holocene

X1=df_3["Qdiv"].values
Y1=df_3["Pwat"].values
Z1=df_3["(P-E)"].values
T =df_3["TGMS"].values


xavg=np.array([]) #10 year means
yavg=np.array([])
zavg=np.array([])
for i in range(0, 1000): #calculating arrays of 10-year averages for each variable
	df = df_3.iloc[i*10:10+i*10,:]
	xavg = np.append(xavg, np.mean(df["Qdiv"].values))
	yavg = np.append(yavg, np.mean(df["Pwat"].values))
	zavg = np.append(zavg, np.mean(df["(P-E)"].values))

TGMS = []
for i in range(len(zavg)):
	TGMS.append(xavg[i]/zavg[i])

plt.style.use('bmh')
X = np.linspace(-10000, 0, 1000)

#print(np.max(zavg), np.argmax(zavg))

print(len(zavg))
plt.scatter(yavg, TGMS, c='black', s=4)
plt.xlabel('Pwat (kg/m^2)')
plt.ylabel('TGMS')
plt.title("-10,000 years to 1990 CE (decadal averages) ")
plt.show()

print(np.corrcoef(yavg, TGMS))





#Fitting a curve for TGMS vs Pwat, JJAS average from 1959-2021
a = 'ERA levels.nc'
f10 = netCDF4.Dataset(a)


#Variable structure = [month, height, expver, lat, lon]
#month = 256 (JJAS of 1959-2022), height = 37
#expver = 0 for ERA5, = 1 for ERA5T
#print(f8.variables.keys())
# m = Moist static energy (Cp.T + g.Z + Lv.q)
v1 = f7['lsm'][0, :, :]
VMS = []
for i in range(0,252,4):
	sum = 0
	for j in range(i, i+3):
		m1 = np.array(Cp*f10['t'][j,14,:,:] + f10['z'][j,14,:,:] + Lv*f10['q'][j,14,:,:])
		m2 = np.array(Cp*f10['t'][j,28,:,:] + f10['z'][j,28,:,:] + Lv*f10['q'][j,28,:,:])
		sum = sum + np.multiply(v1, m1-m2)
	VMS.append(np.average(sum[np.nonzero(sum)])/4)

#Qdiv
P = []
Qdiv = []
Pwat = []
for i in range(5,756,12):
	sum1 = 0
	sum2 = 0
	sum3 = 0
	for j in range(i, i+3):
		v2 = f8['mtnswrf'][j, :, :]
		v3 = f8['mtnlwrf'][j, :, :]
		v4 = f8['mslhf'][j, :, :]
		v5 = f8['msshf'][j, :, :]
		v6 = f8['msnswrf'][j, :, :]
		v7 = f8['msnlwrf'][j, :, :]
		v8 = f8['mtpr'][j, :, :]
		v9 = f8['mer'][j, :, :]
		v10 = f8['tcwv'][j, :, :]
		sum1 = sum1 + v2+v3+v4+v5+v6+v7
		sum2 = sum2 + 86400*28.96*(v8+v9)
		sum3 = sum3 + v10
	x = np.multiply(v1, sum1)
	y = np.multiply(v1, sum2)
	z = np.multiply(v1, sum3)
	Qdiv.append(np.average(x[np.nonzero(x)])/4)
	P.append(np.average(y[np.nonzero(y)])/4)
	Pwat.append(np.average(z[np.nonzero(z)])/4)

TGMS = []
rat = []
rec = []
for i in range(len(P)):
	TGMS.append(Qdiv[i]/P[i])
	rat.append(VMS[i]/Pwat[i])
	rec.append(1/Pwat[i])


'''
plt.scatter(rec, TGMS, c='black')
plt.xlabel('1/Pwat (1/Kg/m^2)')
plt.ylabel('TGMS')
plt.title('Yearly JJAS average TGMS vs 1/Pwat')
#plt.figtext(0.4, 0.2, 'Correlation ='+ str(np.corrcoef(rec, TGMS)[1]))
plt.legend(['ERA5'])

'''



def objective(x, a, b, c):
	return a * x + b * x**2 + c
# fit a second degree polynomial
from scipy.optimize import curve_fit
from matplotlib import pyplot
 
# define the true objective function
def objective(x, a, b, c):
	return a * x + b * x**2 + c
# choose the input and output variables
x, y = rec, TGMS 
# curve fit
popt, _ = curve_fit(objective, x, y)
# summarize the parameter values
a, b, c = popt

print('y = %.5f * x + %.5f * x^2 + %.5f' % (a, b, c))
# plot input vs output
plt.style.use("bmh")
pyplot.scatter(x, y)

# define a sequence of inputs between the smallest and largest known inputs
x_line = np.linspace(np.min(x), np.max(x), 100)
#x_line = np.linspace(0, 0.05, 100)

# calculate the output for the range
y_line = objective(x_line, a, b, c)
che = objective(x_line, -224, 5108, 3.2)
pyplot.plot(x_line, y_line, color='red', label = 'Best fit for ERA5')
pyplot.plot(x_line, che, color='g', label="Dr. Chetan's fit")
plt.ylabel('TGMS')
plt.xlabel('1/Pwat (m^2/kg)')
#plt.figtext(0.2, 0.6, 'TGMS = 8217/Pwat\u00b2 - 402/Pwat + 5')
plt.legend()
plt.title('JJAS average 1959-2021')
pyplot.show()

pred = []
for i in rec:
	pred.append(a * i + b * i**2 + c)

sum2 = 0
for i in range(len(rec)):
	sum2 = sum2 + abs(pred[i]-TGMS[i])

print(math.sqrt(sum2/len(rec)))
print(len(TGMS))
print(np.argmax(TGMS), TGMS[28])




