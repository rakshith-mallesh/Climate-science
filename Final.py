import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import scipy.optimize as opt
import math
import netCDF4
from mpl_toolkits.basemap import Basemap
data = pd.read_csv("Landsat.csv")




#Correlation
corr = data.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(data.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(data.columns)
ax.set_yticklabels(data.columns)
plt.show()




#Finding out the relation b/w the variables
X= data["Qdiv"].values
Y= data["Pwat"].values
Z= data["(P-E)"].values

C= ((X+Z)*Y)/Z
lis=C.tolist()
print("\nBest estimate for C = ",np.mean(C))

plt.scatter(X, Z, s=0.1)
plt.xlabel('Qdiv')
plt.ylabel('(P-E)')
plt.show()

plt.scatter(Y, Z, s=0.1)
plt.xlabel('Pwat')
plt.ylabel('(P-E)')
plt.show()




#Linear relationship b/w Pwat and (P-E)
X=data["Pwat"].values
Y=data["(P-E)"].values

df_1 = data.iloc[:7000,:]
df_2 = data.iloc[7000:,:]

X1=df_1["Pwat"].values
Y1=df_1["(P-E)"].values

mean_x1 = np.mean(X1)
mean_y1 = np.mean(Y1)

n = len(X1)

numerator = 0
denominator = 0
for i in range(n):
  numerator += (X1[i] - mean_x1) * (Y1[i] - mean_y1)
  denominator += (X1 [i] - mean_x1) ** 2
B1 = numerator / denominator
A1 = mean_y1 - (B1 * mean_x1)

x1 = np.linspace (20, 33, 13)
y1 = A1 + B1 * x1

#MSE
y_pred1=[]

for i in range(13):
 y_pred1.append(A1+B1*x1[i])

X2=df_2["Pwat"].values
Y2=df_2["(P-E)"].values

mean_x2 = np.mean(X2)
mean_y2 = np.mean(Y2)

n = len(X2)

numerator = 0
denominator = 0
for i in range(n):
  numerator += (X2[i] - mean_x2) * (Y2[i] - mean_y2)
  denominator += (X2[i] - mean_x2) ** 2
B2 = numerator / denominator
A2 = mean_y2 - (B2 * mean_x2)

x2 = np.linspace (30, 41, 11)
y2 = A2 + B2 * x2

#MSE
y_pred2=[]

for i in range(11):
 y_pred1.append(A2+B2*x2[i])

plt.plot(x1, y1, c='r')
plt.plot(x2, y2, c='b')
plt.scatter(X1, Y1, c='g', s=0.1)
plt.scatter(X2, Y2, c='tab:orange', s=0.1)

plt.xlabel("Pwat")
plt.ylabel("(P-E)")
plt.legend(["Prediction Line 1", "Prediction Line 2", "-22,000 -15,000 yrs", "-15,000 yrs to 1950"])
plt.show()




#Finding out the constants C1 and C2
X1=df_1["Qdiv"].values
Y1=df_1["Pwat"].values
Z1=df_1["(P-E)"].values

X2=df_2["Qdiv"].values
Y2=df_2["Pwat"].values
Z2=df_2["(P-E)"].values

def f(X, c1, c2):
    x1,y1 = X
    return (x1*y1)/(c1-c2*y1)

x = data["Qdiv"].values
y = data["Pwat"].values
z = data["(P-E)"].values

xavg=yavg=zavg=np.array([])
for i in range(0,221):
	df = data.iloc[i*100:100+i*100,:]
	xavg = np.append(xavg, np.mean(df["Qdiv"].values))
	yavg = np.append(yavg, np.mean(df["Pwat"].values))
	zavg = np.append(zavg, np.mean(df["(P-E)"].values))
	
p0 = 75, 1.5
popt, pcov = opt.curve_fit(f, (xavg, yavg), zavg, p0)
popt1, pcov1 = opt.curve_fit(f, (X1, Y1), Z1, p0)
popt2, pcov2 = opt.curve_fit(f, (X2, Y2), Z2, p0)

print(popt, popt1, popt2)

X = np.linspace(-22100, 0, 221, dtype='int')
pred1 = pred2 = np.array([])

for i in range(0,221):
	pred1 = np.append(pred1, (xavg[i]*yavg[i])/(80.14685808-1.22592726*yavg[i]))
	pred2 = np.append(pred2, (xavg[i]*yavg[i])/(87-1.43*yavg[i]))

plt.plot(X, zavg, label = "Actual line")
plt.plot(X, pred1, c='r', label = "Python model")
plt.plot(X, pred2, c='g', label = "Dr. Chetan's model")
plt.xlabel('Age (ka)')
plt.ylabel('P-E (mm/day)')
plt.legend()
plt.show()

sum1 = 0
sum2 = 0
for i in range(0,221):
	sum1 = sum1 + abs(zavg[i]-pred1[i])
	sum2 = sum2 + abs(zavg[i]-pred2[i])

print(math.sqrt(sum1/221), math.sqrt(sum2/221))




# Qdiv over 70-90E and 10-30N (All datasets are monthly means)

a = "land.nc"
b = "dswrf.ntat.nc"
c = "uswrf.ntat.nc"
d = "ulwrf.ntat.nc"
e = "lhtfl.nc" #this is the evaporation rate
f = "shtfl.nc"
g = "dswrf.sfc.nc"
h = "uswrf.sfc.nc"
i = "dlwrf.sfc.nc"
j = "ulwrf.sfc.nc"
k = "prate.nc"

f1 = netCDF4.Dataset(a)
f2 = netCDF4.Dataset(b)
f3 = netCDF4.Dataset(c)
f4 = netCDF4.Dataset(d)
f5 = netCDF4.Dataset(e)
f6 = netCDF4.Dataset(f)
f7 = netCDF4.Dataset(g)
f8 = netCDF4.Dataset(h)
f9 = netCDF4.Dataset(i)
f10 = netCDF4.Dataset(j)
f11 = netCDF4.Dataset(k)

#print(f11.variables.keys()) # get all variable names
print(f11['prate'])
#dimensions: Longitude = 192 (columns), Latitude = 94 (rows), time = 520

Qdiv = []
P = []

for j in range(0, 504, 12):
	for i in range(5+j, 8+j):
		v1 = f1['land'][0, 31:47, 38:49]
		v2 = f2['dswrf'][i, 31:47, 38:49]
		v3 = f3['uswrf'][i, 31:47, 38:49]
		v4 = f4['ulwrf'][i, 31:47, 38:49]
		v5 = f5['lhtfl'][i, 31:47, 38:49]
		v6 = f6['shtfl'][i, 31:47, 38:49]
		v7 = f7['dswrf'][i, 31:47, 38:49]
		v8 = f8['uswrf'][i, 31:47, 38:49]
		v9 = f9['dlwrf'][i, 31:47, 38:49]
		v10 = f10['ulwrf'][i, 31:47, 38:49]
		v11 = f11['prate'][i, 31:47, 38:49]
		x = np.multiply(v1, 86400*28.94*v11-v5)
		y = np.multiply(v1, v2 - v3 - v4 + v5 + v6 - v7 + v8 - v9 + v10)
		P.append(np.average(x[np.nonzero(x)]))
		Qdiv.append(np.average(y[np.nonzero(y)]))	

TGMS = []
for i in range(len(Qdiv)):
    TGMS.append(Qdiv[i]/P[i])

X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
plt.plot(X, Qdiv)
plt.ylabel("Qdiv (W/m^2)")
plt.show()

plt.plot(X, P, c='g')
plt.xlabel("Months")
plt.ylabel("(P-E) (W/m^2)")
plt.show()

plt.plot(X, TGMS, c='tab:orange')
plt.xlabel("Months")
plt.ylabel("TGMS")
plt.show()
plt.savefig('filename.png', dpi=1200, bbox_inches="tight")




#Holocene
df_3 = data.iloc[10000:,:]

X1=df_3["Qdiv"].values
Y1=df_3["Pwat"].values
Z1=df_3["(P-E)"].values

def f(X, c1, c2):
    x1,y1 = X
    return (x1*y1)/(c1-c2*y1)

x = data["Qdiv"].values
y = data["Pwat"].values
z = data["(P-E)"].values

xavg=np.array([])
yavg=np.array([])
zavg=np.array([])
for i in range(0,221):
	df = data.iloc[i*100:100+i*100,:]
	xavg = np.append(xavg, np.mean(df["Qdiv"].values))
	yavg = np.append(yavg, np.mean(df["Pwat"].values))
	zavg = np.append(zavg, np.mean(df["(P-E)"].values))
	
p0 = 80, 1.2
popt1, pcov1 = opt.curve_fit(f, (X1, Y1), Z1, p0)


print(popt1)

X1 = np.linspace(-10000, 0, 100, dtype='int')
pred1 = np.array([])

C1, C2 = 255.24, 5.886
C3, C4 = 263.33, 6.072
C5, C6 = 119.60198789, 2.29268188
C7, C8 = 80.146, 1.225

pred1 = np.array([])
for i in range(121,221):
	pred1 = np.append(pred1, (xavg[i]*yavg[i])/(C7-C8*yavg[i]))

plt.plot(X1, zavg[121: 221], label = "Actual line")
plt.plot(X1, pred1, c='r', label = "Prediction")
plt.xlabel('Age')
plt.ylabel('P-E (mm/day)')
plt.legend()
plt.show()

sum1 = 0
for i in range(121,221):
	sum1 = sum1 + abs(zavg[i]-pred1[i-121])

print(math.sqrt(sum1/100))





#Standard deviations
xavg=np.array([])
yavg=np.array([])
zavg=np.array([])
tavg=np.array([])
for i in range(0,220):
	df = data.iloc[i*100:100+i*100,:]
	xavg = np.append(xavg, np.mean(df["Qdiv"].values))
	yavg = np.append(yavg, np.mean(df["Pwat"].values))
	zavg = np.append(zavg, np.mean(df["(P-E)"].values))
	tavg = np.append(tavg, np.mean(df["TGMS"].values))

sd1 = []
sd2 = []
sd3 = [] 
sd4 = [] 
for i in range(0,220):
	sum1 = 0
	sum2 = 0
	sum3 = 0
	sum4 = 0
	count = 0
	df = data.iloc[i*100:100+i*100,:]
	x = df["Qdiv"].values
	y = df["Pwat"].values
	z = df["(P-E)"].values
	t = df["TGMS"].values
	for j in range(0, 100):
		if not(t[j]>10) and not(t[j]<0):
			sum1 = sum1 + (t[j]-tavg[i])**2 #TGMS
			sum2 = sum2 + (z[j]-zavg[i])**2 #(P-E)
			sum3 = sum3 + (x[j]-xavg[i])**2 #Qdiv
			sum4 = sum4 + (y[j]-yavg[i])**2 #Pwat
			count = count + 1
	sd1.append(math.sqrt(sum1/count))
	sd2.append(math.sqrt(sum2/count))
	sd3.append(math.sqrt(sum3/count))
	sd4.append(math.sqrt(sum4/count))

count=0
for i in sd2:
	count = count + 1
	if i>0.7:
		print(i)
		print(count)
print(np.corrcoef(sd2[120:220], sd1[120:220]))
print(np.corrcoef(sd2[120:220], sd4[120:220]))

#plt.scatter(sd2[120:220], sd1[120:220], c='tab:orange')
plt.scatter(sd2[120:220], sd4[120:220])
plt.xlabel('Standard deviation of (P-E)')
plt.ylabel('Standard deviation of Pwat')
plt.legend()
plt.show()

j = [178, 193, 199]
for i in j:
	df = data.iloc[i*100:100+i*100,:]
	x = df["Qdiv"].values
	y = df["Pwat"].values
	z = df["(P-E)"].values
	t = df["TGMS"].values
	for k in z:
		print(k)
	print("\n")

X = np.linspace(-220, 0, 220, dtype='int')

plt.plot(X, sd1, c='r', label = "Standard deviation of TGMS")
plt.plot(X, sd2, c='g', label = "Standard deviation of P-E")
plt.plot(X, sd3, c='tab:orange', label = "Standard deviation of Qdiv")
plt.xlabel('Age (ka)')
plt.ylabel('Standard deviation')
plt.legend()
plt.show()





#Comparing Pwat for couple of years
a = 'pr_wtr.eatm.1987.nc'
b = 'pr_wtr.eatm.1988.nc'
c = 'pr_wtr.eatm.1982.nc'
d = 'pr_wtr.eatm.1983.nc'
e = 'pr_wtr.eatm.1979.nc'
f = 'pr_wtr.eatm.1980.nc'
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

#print(f2['lon'][28:37])
#print(f2['lat'][24:33])

P1=[]
P2=[]
P3=[]
P4=[]
P5=[]
P6=[]
P7=[]
P8=[]

for i in range(0, 365):
	v1 = f1['pr_wtr'][i, 24:33, 28:37]
	v2 = f2['pr_wtr'][i, 24:33, 28:37]
	v3 = f3['pr_wtr'][i, 24:33, 28:37]
	v4 = f4['pr_wtr'][i, 24:33, 28:37]
	v5 = f5['pr_wtr'][i, 24:33, 28:37]
	v6 = f6['pr_wtr'][i, 24:33, 28:37]
	v7 = f7['pr_wtr'][i, 24:33, 28:37]
	v8 = f8['pr_wtr'][i, 24:33, 28:37]
	vl = f9['land'][0, 24:33, 28:37]
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

X=np.linspace(1, 365, 365)
plt.plot(X, P1, c='g', label='1987')
plt.plot(X, P2, c='r', label='1988')
plt.xlabel('Day of the year')
plt.ylabel('Precipitable water content (Kg/m^2)')
plt.legend()
#plt.savefig('Pwat 1987.png', dpi=1200, bbox_inches="tight")
plt.show()

plt.plot(X, P3, c='g', label='1982')
plt.plot(X, P4, c='r', label='1983')
plt.xlabel('Day of the year')
plt.ylabel('Precipitable water content (Kg/m^2)')
plt.legend()
plt.show()

plt.plot(X, P5, c='g', label='1979')
plt.plot(X, P6, c='r', label='1980')
plt.xlabel('Day of the year')
plt.ylabel('Precipitable water content (Kg/m^2)')
plt.legend()
plt.show()

plt.plot(X, P7, c='g', label='2002')
plt.plot(X, P8, c='r', label='2003')
plt.xlabel('Day of the year')
plt.ylabel('Precipitable water content (Kg/m^2)')
plt.legend()
plt.show()



#The same from ERA5 data
c = 'Pwat ERA.nc'
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
    v1 = f3['tcwv'][i, :, :]
    v2 = f3['tcwv'][i+365, :, :]
    v3 = f3['tcwv'][i+731, :, :]
    v4 = f3['tcwv'][i+1096, :, :]
    v5 = f3['tcwv'][i+1461, :, :]
    v6 = f3['tcwv'][i+1826, :, :]
    v7 = f3['tcwv'][i+2192, :, :]
    v8 = f3['tcwv'][i+2557, :, :]
    vl = f4['lsm'][0, :, :]
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
    
X = np.linspace(1, 365, 365, dtype='int')
    
plt.plot(X, pwat1979, c = 'g', label = 1979)
plt.plot(X, pwat1980, c = 'r', label = 1980)
plt.xlabel('Day of the year')
plt.ylabel('Precipitable water content (Kg/m^2)')
plt.legend()
plt.show()

plt.plot(X, pwat1982, c = 'g', label = '1982')
plt.plot(X, pwat1983, c = 'r', label = '1983')
plt.xlabel('Day of the year')
plt.ylabel('Precipitable water content (Kg/m^2)')
plt.legend()
plt.show()

plt.plot(X, pwat1987, c = 'g', label = '1987')
plt.plot(X, pwat1988, c = 'r', label = '1988')
plt.xlabel('Day of the year')
plt.ylabel('Precipitable water content (Kg/m^2)')
plt.legend()
plt.show()

plt.plot(X, pwat2002, c = 'g', label = '2002')
plt.plot(X, pwat2003, c = 'r', label = '2003')
plt.xlabel('Day of the year')
plt.ylabel('Precipitable water content (Kg/m^2)')
plt.legend()
plt.show()






#Spatial plot

a = 'ERA Pwat, Prec 1979-2020.nc'
f1 = netCDF4.Dataset(a)

lats = f1['latitude'][:]
lons = f1['longitude'][:]
mp = Basemap(projection = 'merc',
				llcrnrlon = 70,
				llcrnrlat = 10,
				urcrnrlon = 90,
				urcrnrlat = 30,
				resolution = 'i')
JJAS1 = (f1['tcwv'][0, :, :]+f1['tcwv'][1, :, :]+f1['tcwv'][2, :, :]+f1['tcwv'][3, :, :])
JJAS2 = (f1['tcwv'][60, :, :]+f1['tcwv'][61, :, :]+f1['tcwv'][62, :, :]+f1['tcwv'][63, :, :])
AS = (f1['tcwv'][2, :, :]+f1['tcwv'][3, :, :])/2
lon, lat = np.meshgrid(lons, lats)
x, y = mp(lon, lat)

Jmean = 0
Jumean = 0
Amean = 0
Smean = 0
count = 0
for i in range(0, 168, 4):
	Jmean = f1['tcwv'][i, :, :] + Jmean
	Jumean = f1['tcwv'][i+1, :, :] + Jumean
	Amean = f1['tcwv'][i+2, :, :] + Amean
	Smean = f1['tcwv'][i+3, :, :] + Smean
	count = count + 1

Jmean = Jmean/count
Jumean = Jumean/count
Amean = Amean/count
Smean = Smean/count

J1979 = f1['tcwv'][0, :, :] - Jmean
Ju1979 = f1['tcwv'][1, :, :] - Jumean
A1979 = f1['tcwv'][2, :, :] - Amean
S1979 = f1['tcwv'][3, :, :] - Smean

diff = JJAS1-JJAS2
c_scheme = mp.pcolor(x, y, np.squeeze(diff), cmap='jet')
mp.drawcoastlines()
mp.drawcountries()
cbar = mp.colorbar(c_scheme)
plt.title('Difference in total JJAS water vapour of 1994 and 1979')
plt.show()




#Relation between JJ rainfall and AS water vapour
prec = pd.read_csv('Precipitation 2.csv')
a = 'prate.sfc.nc'
c = 'land.nc'
d = 'ERA Pwat, Prec 1979-2020.nc'
e = 'ERA Land sea mask.nc'
f1 = netCDF4.Dataset(a)
f3 = netCDF4.Dataset(c)
f4 = netCDF4.Dataset(d)
f5 = netCDF4.Dataset(e)
values = prec["JJ"].values

for i in range(0, 144, 4):
	e1 = f4['tcwv'][i+2, :, :]
	e2 = f4['tcwv'][i+3, :, :]
	e5 = f5['lsm'][0, :, :]
	y1 = np.multiply(e1,e5)
	y2 = np.multiply(e2,e5)
	ya = (y1+y2)/2
	e3 = f4['mtpr'][i, :, :]
	e4 = f4['mtpr'][i+1, :, :]
	y3 = np.multiply(e3,e5)
	y4 = np.multiply(e4,e5)
	yb = 86400*30*(y3+y4)
	Pwat2.append(np.average(y1[np.nonzero(y1)]))
	Prec2.append(np.average(yb[np.nonzero(yb)]))

Prec3 = []
for i in range(5, 428, 12):
	v1 = f1['prate'][i, 31:42, 38:49]
	v2 = f1['prate'][i+1, 31:42, 38:49]
	v3 = f3['land'][0, 31:42, 38:49]
	x1 = np.multiply(v1,v3)
	x2 = np.multiply(v2,v3)
	xa = 86400*30*(x1+x2)
	Prec3.append(np.average(xa[np.nonzero(xa)]))

#plt.scatter(values[78:], Pwat2, c = 'g', label='Rainfall from Indian records')
#plt.scatter(Prec3, Pwat2, c = 'r', label='Rainfall from NCEP')
plt.scatter(Prec2, Pwat2, label = 'Rainfall from ERA5')
plt.xlabel('JJ rainfall (mm/month)')
plt.ylabel('August water vapour (kg/m^2)')
plt.legend()
plt.show()



#CMIP6 data
f = 'pr_Amon_GFDL-ESM4_historical_r1i1p1f1_gr1_195001-201412.nc'
g = 'prw_Amon_GFDL-ESM4_historical_r1i1p1f1_gr1_195001-201412.nc'
dsw = 'rsdt_Amon_GFDL-ESM4_historical_r1i1p1f1_gr1_195001-201412.nc'
usw = 'rsut_Amon_GFDL-ESM4_historical_r1i1p1f1_gr1_195001-201412.nc'
ulw = 'rlut_Amon_GFDL-ESM4_historical_r1i1p1f1_gr1_195001-201412.nc'
evp = 'evspsbl_Amon_GFDL-ESM4_historical_r1i1p1f1_gr1_195001-201412.nc'
#rsut = Outgoing Shortwave rlut = Outgoing Longwave rsdt =  Incoming Shortwave evspsbl = Evaporation

f6 = netCDF4.Dataset(f)
f7 = netCDF4.Dataset(g)
f8 = netCDF4.Dataset(dsw)
f9 = netCDF4.Dataset(usw)
f10 = netCDF4.Dataset(ulw)
f11 = netCDF4.Dataset(evp)

Jpwat = []
Jupwat = []
Apwat = []
Spwat = []
Jprec = []
Juprec = []
Aprec = []
Sprec = []
for i in range(5, 780, 12):
	e1 = f6['pr'][i, 100:120, 56:73]
	e2 = f6['pr'][i+1, 100:120, 56:73]
	e3 = f6['pr'][i+2, 100:120, 56:73]
	e4 = f6['pr'][i+3, 100:120, 56:73]
	e5 = f7['prw'][i, 100:120, 56:73]
	e6 = f7['prw'][i+1, 100:120, 56:73]
	e7 = f7['prw'][i+2, 100:120, 56:73]
	e8 = f7['prw'][i+3, 100:120, 56:73]
	Jprec.append(np.average(86400*30*(e1)))
	Jpwat.append(np.average(e5))
	Juprec.append(np.average(86400*30*(e2)))
	Jupwat.append(np.average(e6))
	Aprec.append(np.average(86400*30*(e3)))
	Apwat.append(np.average(e7))
	Sprec.append(np.average(86400*30*(e4)))
	Spwat.append(np.average(e8))

plt.scatter(Jprec, Jpwat, c = 'g', label = 'June')
plt.scatter(Juprec, Jupwat, c = 'b', label = 'July')
plt.scatter(Aprec, Apwat, c = 'r', label = 'August')
plt.scatter(Sprec, Spwat, c = 'tab:orange', label = 'September')
plt.xlabel('Precipitation (mm/month)')
plt.ylabel('Water vapour (kg/m^2)')
plt.legend()
plt.show()

#print(np.corrcoef(Prec1, Pwat1))

X = np.linspace(1, 12, 12)
Qnet = []
for i in range(0, 12):
	q1 = f8['rsdt'][i, 100:120, 56:73]
	q2 = f9['rsut'][i, 100:120, 56:73]
	q3 = f10['rlut'][i, 100:120, 56:73]
	Qnet.append(np.average(q1)-np.average(q2)-np.average(q3))
	
plt.plot(X, Qnet, c = 'tab:orange')
plt.xlabel("Month of the year")
plt.ylabel("Qnet (W/m^2)")
plt.title("Qnet for the year 1950")
plt.show()




#GFDL
#f = 'pr_Amon_GFDL-ESM4_historical_r1i1p1f1_gr1_195001-201412.nc'
#g = 'prw_Amon_GFDL-ESM4_historical_r1i1p1f1_gr1_195001-201412.nc'
#dsw = 'rsdt_Amon_GFDL-ESM4_historical_r1i1p1f1_gr1_195001-201412.nc'
#usw = 'rsut_Amon_GFDL-ESM4_historical_r1i1p1f1_gr1_195001-201412.nc'
#ulw = 'rlut_Amon_GFDL-ESM4_historical_r1i1p1f1_gr1_195001-201412.nc'
#evp = 'evspsbl_Amon_GFDL-ESM4_historical_r1i1p1f1_gr1_195001-201412.nc'

a = 'evaporation.nc'

f = 'pr_Amon_CMCC-CM2-SR5_historical_r1i1p1f1_gn_185001-201412.nc'
g = 'prw_Amon_CMCC-CM2-SR5_historical_r1i1p1f1_gn_185001-201412.nc'
dsw = 'rsdt_Amon_CMCC-CM2-SR5_historical_r1i1p1f1_gn_185001-201412.nc'
usw = 'rsut_Amon_CMCC-CM2-SR5_historical_r1i1p1f1_gn_185001-201412.nc'
ulw = 'rlut_Amon_CMCC-CM2-SR5_historical_r1i1p1f1_gn_185001-201412.nc'
evp = 'evspsbl_Amon_CMCC-CM2-SR5_historical_r1i1p1f1_gn_185001-201412.nc'
#rsut = Outgoing Shortwave rlut = Outgoing Longwave rsdt =  Incoming Shortwave evspsbl = Evaporation
#Qnet = rsdt - rsut- rlut 

f1 = netCDF4.Dataset(a)
f6 = netCDF4.Dataset(f)
f7 = netCDF4.Dataset(g)
f8 = netCDF4.Dataset(dsw)
f9 = netCDF4.Dataset(usw)
f10 = netCDF4.Dataset(ulw)
f11 = netCDF4.Dataset(evp)

#print(f6.variables.keys())
#print(f6['lat'][107:129])
#print(f6['lon'][56:73])
#print(f6['longitude'][280:361])
#print(f1['mer'])

eera = []
for i in range(36):
	eera.append(-86400*np.average(f1['mer'][i, :, :]))

Jp = []
Jup = []
Ap = []
Sp = []
JQnet = []
JuQnet = []
AQnet = []
SQnet = []

rain = []
ecmip = []
Juprw = []
Aprw = []
for i in range(5, 1980, 12):
	Juprw.append(np.average(f7['prw'][i+1, 107:129, 56:73]))
	Aprw.append(np.average(f7['prw'][i+2, 107:129, 56:73]))
	e1 = f6['pr'][i, 107:129, 56:73] #- f11['evspsbl'][i, 53:65, 38:49]
	e2 = f6['pr'][i+1, 107:129, 56:73] #- f11['evspsbl'][i+1, 53:65, 38:49]
	e3 = f6['pr'][i+2, 107:129, 56:73] #- f11['evspsbl'][i+2, 53:65, 38:49]
	e4 = f6['pr'][i+3, 107:129, 56:73] #- f11['evspsbl'][i+3, 53:65, 38:49]
	
	rain.append(86400*np.average(e3))
	e5 = f11['evspsbl'][i, 107:129, 56:73]
	e6 = f11['evspsbl'][i+1, 107:129, 56:73]
	e7 = f11['evspsbl'][i+2, 107:129, 56:73]
	e8 = f11['evspsbl'][i+3, 107:129, 56:73]
	ecmip.append(86400*np.average(e7))

	q1 = f8['rsdt'][i, 107:129, 56:73]
	q2 = f9['rsut'][i, 107:129, 56:73]
	q3 = f10['rlut'][i, 107:129, 56:73]

	q4 = f8['rsdt'][i+1, 107:129, 56:73]
	q5 = f9['rsut'][i+1, 107:129, 56:739]
	q6 = f10['rlut'][i+1, 107:129, 56:73]

	q7 = f8['rsdt'][i+2, 107:129, 56:73]
	q8 = f9['rsut'][i+2, 107:129, 56:73]
	q9 = f10['rlut'][i+2, 107:129, 56:73]

	q10 = f8['rsdt'][i+3, 107:129, 56:73]
	q11 = f9['rsut'][i+3, 107:129, 56:73]
	q12 = f10['rlut'][i+3, 107:129, 56:73]
	Jp.append(86400*28.94*(np.average(e1)-np.average(e5)))
	Jup.append(86400*28.94*(np.average(e2)-np.average(e6)))
	Ap.append(86400*28.94*(np.average(e3)-np.average(e7)))
	Sp.append(86400*28.94*(np.average(e4)-np.average(e8)))
	JQnet.append(np.average(q1)-abs(np.average(q2))-abs(np.average(q3)))
	JuQnet.append(np.average(q4)-abs(np.average(q5))-abs(np.average(q6)))
	AQnet.append(np.average(q7)-abs(np.average(q8))-abs(np.average(q9)))
	SQnet.append(np.average(q10)-abs(np.average(q11))-abs(np.average(q12)))
	


X = np.linspace(1979, 2014, 36)
#plt.plot(X, eera, c = 'g', label = 'Evaporation from ERA5')
#plt.plot(X, ecmip[129:165], label = 'Evaporation from CMIP6 for July')
#plt.plot(X, rain[129:165], label = 'Rainfall from CMIP6 for July')
plt.xlabel('Year')
plt.ylabel('mm/day')
plt.legend()
plt.show()

X = np.linspace(1850, 2014, 165, dtype = 'int')
JTGMS = []
JuTGMS = []
ATGMS = []
STGMS = []
for i in range(len(Jp)):
	JTGMS.append(JQnet[i]/Jp[i])
	JuTGMS.append(JuQnet[i]/Jup[i])
	ATGMS.append(AQnet[i]/Ap[i])
	STGMS.append(SQnet[i]/Sp[i])

plt.plot(X, Jp, label = 'June')
plt.plot(X, Jup, c = 'g', label = 'July')
plt.plot(X, Ap, c = 'r', label = 'August')
plt.plot(X, Sp, c = 'tab:orange', label = 'September')
plt.xlabel("Year")
plt.ylabel("P-E (mm/day)")
plt.legend()
plt.show()

for i in range(len(ATGMS)):
	if ATGMS[i]>1.5:
		print(i)

plt.scatter(ATGMS, Aprw, c = 'r')
plt.xlabel("TGMS for August")
plt.ylabel("Water vapour (kg/m2)")
plt.show()


#plt.plot(X, JTGMS, c = 'tab:orange', label = 'June')
plt.plot(X, JuTGMS, c = 'b', label = 'July')
plt.plot(X, ATGMS, c = 'r', label = 'August')
#plt.plot(X, STGMS, c = 'g', label = 'September')
plt.xlabel("Year")
plt.ylabel("TGMS")
plt.legend()
plt.show()
	



	






