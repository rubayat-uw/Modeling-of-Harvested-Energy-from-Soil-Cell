#import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pandas import DataFrame
from scipy.optimize import curve_fit
import matplotlib.dates as mdates

#Load data
var=pd.read_csv(r"C:\Users\mrrbhuyi\Dropbox\MASc\Data Collection\SC_OC_V 2020-02-03 14-26-45 0_mod.csv")
var1=pd.read_csv(r"C:\Users\mrrbhuyi\Dropbox\MASc\Data Collection\10K_A 2020-01-21 16-59-09 0_mod.csv")
var2=pd.read_csv(r"C:\Users\mrrbhuyi\Dropbox\MASc\Data Collection\1K_A 2020-01-22 12-36-51 0_mod.csv")
var3=pd.read_csv(r"C:\Users\mrrbhuyi\Dropbox\MASc\Data Collection\100ohm_A 2020-01-23 13-59-42 0 2020-01-23 14-06-27 0_Mod.csv")
var4=pd.read_csv(r"C:\Users\mrrbhuyi\Dropbox\MASc\Data Collection\56ohm_A 2020-01-24 13-11-47 0_mod.csv")
var5=pd.read_csv(r"C:\Users\mrrbhuyi\Dropbox\MASc\Data Collection\33ohm_A 2020-01-26 13-20-27 0_mod.csv")
var6=pd.read_csv(r"C:\Users\mrrbhuyi\Dropbox\MASc\Data Collection\22ohm_A 2020-01-27 17-55-16 0_mod.csv")
var7=pd.read_csv(r"C:\Users\mrrbhuyi\Dropbox\MASc\Data Collection\12ohm_A 2020-01-28 19-28-46 0_mod.csv")
var8=pd.read_csv(r"C:\Users\mrrbhuyi\Dropbox\MASc\Data Collection\6ohm_A 2020-01-30 12-05-29 0_mod.csv")
var9=pd.read_csv(r"C:\Users\mrrbhuyi\Dropbox\MASc\Data Collection\SC_test2 2020-02-06 12-55-03 0_mod.csv")
load=pd.read_csv(r"C:\Users\mrrbhuyi\Dropbox\MASc\Data Collection\PowerBox_12011_-8586188714277288678.csv")
charge=pd.read_csv(r"C:\Users\mrrbhuyi\Dropbox\MASc\Data Collection\charge.csv")
charge_2=pd.read_excel(r"C:\Users\mrrbhuyi\Dropbox\MASc\Data Collection\charge_2.xlsx")
discharge=pd.read_csv(r"C:\Users\mrrbhuyi\Dropbox\MASc\Data Collection\discharge.csv")
load_2=pd.read_csv(r"C:\Users\mrrbhuyi\Dropbox\MASc\Data Collection\load.csv")

plt.rcParams["figure.figsize"] = [6.4, 4.8]
t1_l=load.iloc[1:412525,0]
y_l=load.iloc[1:412525,1]
d_l=pd.DataFrame ({'t':t1_l,'y':y_l})
rolling_mean = d_l.y.rolling(window=1000).mean()
rolling_mean2 = d_l.y.rolling(window=25000).mean()
#plt.plot(d_l.t, d_l.y, label='AMD')
plt.plot(d_l.t, rolling_mean, color='blue',label="Measured")
plt.plot(d_l.t, rolling_mean2, color='red',label="Fit")

plt.axvspan(0, 20, color='grey', alpha=0.5)
plt.text(7.5,160, 'Startup', horizontalalignment='left', size='small', color='red')
plt.text(26,160, 'Message 1', horizontalalignment='left', size='small', color='red')
plt.text(46,160, 'Message 2', horizontalalignment='left', size='small', color='red')
plt.text(65,160, 'Message 3', horizontalalignment='left', size='small', color='red')

#fig = plt.figure()

#ax1_l = plt.subplot(111)
#plt.plot ('t','y', data=d_l)
#plt.text(100000,850, 'Open Circuit Voltage', horizontalalignment='left', size='small', color='black')
#plt.text(2000,15, 'Short Circuit Current', horizontalalignment='left', size='small', color='black')
#plt.text(1600,-1, '10K', horizontalalignment='left', size='small', color='red')
plt.title('Load Charecteristics')
plt.xlabel('Time (s)')
plt.ylabel('Current (mA)')
plt.legend()
plt.grid(True)



################# Avg. Load #########################
#def func (x,a,b):
##    return a*np.exp(-b*x)
#    return a*x+b
#initl=[1,1]
#pl,ql=curve_fit(func,d_l.t,rolling_mean,initl)
#print(pl)
#xFitl=np.arange(0,800,0.01)
#
#figl = plt.figure()
#ccxl = plt.subplot(111)
#plt.plot(d_l.t, rolling_mean,'bo',label='Measured data') 
#plt.plot(xFitl, func(xFitl,*pl),'r',label='fit') 
#plt.title('Load Charecteristics Avg')
#plt.xlabel('Time (s)')
#plt.ylabel('Current (mA)')
#plt.legend()
#ccxl.grid(True)


################# Avg. Load #########################

fig = plt.figure()
ax1_c = plt.subplot(111)
t1_c=charge.iloc[1:641286,0]
y_c=charge.iloc[1:641286,1]
d_c=pd.DataFrame ({'t':t1_c,'y':y_c})
#rolling_mean = d_c.y.rolling(window=1000).mean()
#plt.plot(d_l.t, d_l.y, label='AMD')
plt.plot(d_c.t, d_c.y, color='blue')
#plt.axvspan(0, 20, color='grey', alpha=0.5)
#plt.text(7.5,160, 'Startup', horizontalalignment='left', size='small', color='red')
#plt.text(26,160, 'Message 1', horizontalalignment='left', size='small', color='red')
#plt.text(46,160, 'Message 2', horizontalalignment='left', size='small', color='red')
#plt.text(65,160, 'Message 3', horizontalalignment='left', size='small', color='red')

plt.title('Charging Charecteristics')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (mV)')
plt.grid(True)




charge_2.Time = pd.to_datetime(charge_2.Time,dayfirst=True)

fig = plt.figure()
ax1_c2 = plt.subplot(111)
t1_c2=charge_2.iloc[1:641286,0]
y_c2=charge_2.iloc[1:641286,1]
d_c2=pd.DataFrame ({'t':t1_c2,'y':y_c2})
plt.plot(d_c2.t, d_c2.y, color='blue')


#[ttmin,vvmax]=d_c2.max()
#[ttmax,vvmin]=d_c2.min()
#aa=d_c2.loc[d_c2.y==vvmax, ["t"]]
#bb=d_c2.loc[d_c2.y==vvmin, ["t"]]
plt.axvspan(*mdates.datestr2num(['2020-01-08 19:29:10.632', '2020-01-09 17:21:56.263']), color='red', alpha=0.5)
plt.axvspan(*mdates.datestr2num(['2020-01-09 17:21:56.263', '2020-01-10 13:04:06.263']), color='green', alpha=0.5)
plt.axvspan(*mdates.datestr2num(['2020-01-10 13:04:06.263', '2020-01-11 09:08:03.572']), color='gray', alpha=0.5)
plt.text(pd.Timestamp('2020-01-09 02:00:00'),2.9, 'Cycle 1', horizontalalignment='left', size='large', color='white')
plt.text(pd.Timestamp('2020-01-10 00:00:00'),2.9, 'Cycle 2', horizontalalignment='left', size='large', color='white')
plt.text(pd.Timestamp('2020-01-10 19:00:00'),2.9, 'Cycle 3', horizontalalignment='left', size='large', color='white')



plt.title('Charging Charecteristics')
plt.xlabel('Time')
plt.ylabel('Voltage (mV)')
plt.grid(True)

fig = plt.figure()
ax1_dc2 = plt.subplot(111)
t1_dc2=charge_2.iloc[227580:227830,0]
y_dc2=charge_2.iloc[227580:227830,1]
d_dc2=pd.DataFrame ({'t':t1_dc2,'y':y_dc2})
plt.plot(d_dc2.t, d_dc2.y, color='blue')
plt.axvline(pd.Timestamp('2020-01-09 17:21:56.263'),color='r',linestyle="dashed")
plt.axvline(pd.Timestamp('2020-01-09 17:22:58.856'),color='r',linestyle="dashed")
[tmin,vmax]=d_dc2.max()
[tmax,vmin]=d_dc2.min()
td=tmax-tmin
#td_string=td.strftime("%Y-%m-%d %H:%M:%S")

date_time1 = tmin.strftime("%Y-%m-%d %H:%M:%S")
date_time2 = tmax.strftime("%Y-%m-%d %H:%M:%S")

a=d_dc2.loc[d_dc2.y==vmax, ["t"]]
b=d_dc2.loc[d_dc2.y==vmin, ["t"]]

tt1_dc2=charge_2.iloc[227614:227794,0]
yy_dc2=charge_2.iloc[227614:227794,1]
dd_dc2=pd.DataFrame ({'tt':tt1_dc2,'yy':yy_dc2})
plt.plot(dd_dc2.tt, dd_dc2.yy, color='r')

#plt.text(65,160, '0 days 00:01:26.474000', horizontalalignment='left', size='small', color='red')
#plt.legend()


#from datetime import datetime
#from matplotlib.dates import date2num

#plt.axvspan(*pd.to_datetime(["date_time1", "date_time2"]),
#facecolor='red', alpha=0.5)

plt.axvspan(*mdates.datestr2num(['2020-01-09 17:21:56.263', '2020-01-09 17:22:58.856']), color='red', alpha=0.5)

plt.text(pd.Timestamp('2020-01-09 17:22:20'),3.10, '00:01:02.593', horizontalalignment='left', size='large', color='white')

#plt.axvspan(a.t, b.t, color='green', alpha=0.5)

plt.title('Discharge Charecteristics')
plt.xlabel('Time')
plt.tick_params(axis='x', rotation=45)


#plt.tick_params(axis='x', colors='red', direction='out', length=13, width=3, rotation=45)
plt.ylabel('Voltage (mV)')
plt.grid(True)





#charge_2['Time']=pd.to_datetime(charge_2.Time)

#from pandas import read_csv
#from matplotlib import pyplot
#series = read_csv('charge_2.csv', skiprows=1)
##series['Time']=pd.to_datetime(series.Time)
#series.plot()
#pyplot.show()

#fig = plt.figure()
#ax1_c2 = plt.subplot(111)
#t1_c2=charge_2.iloc[1:641286,0]
#y_c2=charge_2.iloc[1:641286,1]
#d_c2=pd.DataFrame ({'t':t1_c2,'y':y_c2})
##rolling_mean = d_c.y.rolling(window=1000).mean()
##plt.plot(d_l.t, d_l.y, label='AMD')
#plt.plot(d_c2.t, d_c2.y, color='blue')
##plt.axvspan(0, 20, color='grey', alpha=0.5)
##plt.text(7.5,160, 'Startup', horizontalalignment='left', size='small', color='red')
##plt.text(26,160, 'Message 1', horizontalalignment='left', size='small', color='red')
##plt.text(46,160, 'Message 2', horizontalalignment='left', size='small', color='red')
##plt.text(65,160, 'Message 3', horizontalalignment='left', size='small', color='red')
#
#plt.title('Charging Charecteristics')
#plt.xlabel('Time (s)')
#plt.ylabel('Voltage (mV)')
#plt.grid(True)
#





fig = plt.figure()
ax1_dc = plt.subplot(111)
y_dc=discharge.iloc[1:298,1]
t1_dc=discharge.iloc[1:298,0]
d_dc=pd.DataFrame ({'t':t1_dc,'y':y_dc})
#rolling_mean = d_dc.y.rolling(window=100).mean()
#plt.plot(d_l.t, d_l.y, label='AMD')
plt.plot(d_dc.t, d_dc.y, color='blue')
#plt.axvspan(0, 20, color='grey', alpha=0.5)
#plt.text(7.5,160, 'Startup', horizontalalignment='left', size='small', color='red')
#plt.text(26,160, 'Message 1', horizontalalignment='left', size='small', color='red')
#plt.text(46,160, 'Message 2', horizontalalignment='left', size='small', color='red')
#plt.text(65,160, 'Message 3', horizontalalignment='left', size='small', color='red')

plt.title('Discharge Charecteristics')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (mV)')
plt.grid(True)



############################################Open Circuit Voltage###########################################

t1=var.iloc[1:160000,0]
y=var.iloc[1:160000,1]*1000
d=pd.DataFrame ({'t':t1,'y':y})
fig = plt.figure()
ax1 = plt.subplot(111)
plt.plot ('t','y','black', data=d)
plt.text(100000,850, 'Open Circuit Voltage', horizontalalignment='left', size='small', color='black')
plt.text(2000,15, 'Short Circuit Current', horizontalalignment='left', size='small', color='black')
#plt.text(1600,-1, '10K', horizontalalignment='left', size='small', color='red')
plt.title('Open Circuit Voltage')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (mV)')
ax1.grid(True)
max_voltage=max(y)
print (y)
############################################Open Circuit Voltage###########################################






########################################Short Circuit Current##############################################
t11=var9.iloc[1:111000,0]
y11=var9.iloc[1:111000,1]*1000
d11=pd.DataFrame ({'t':t11,'y':y11})
fig = plt.figure()
ax11 = plt.subplot(111)
plt.plot ('t','y', 'black',data=d11)
plt.title('Short Circuit Current')
plt.xlabel('Time (s)')
plt.ylabel('Current (mA)')
ax11.grid(True)
#############################################Short Circuit Current##########################################





data_size=2000
t=var1.iloc[1:data_size,0]
y1=var1.iloc[1:data_size,1]*1000
y2=var2.iloc[1:data_size,1]*1000
y3=var3.iloc[1:data_size,1]*1000
y4=var4.iloc[1:data_size,1]*1000
y5=var5.iloc[1:data_size,1]*1000
y6=var6.iloc[1:data_size,1]*1000
y7=var7.iloc[1:data_size,1]*1000
y8=var8.iloc[1:data_size,1]*1000
y9=var9.iloc[1:data_size,1]*1000


df=pd.DataFrame ({'t':t,'y':y,'y1':y1, 'y2':y2, 'y3':y3, 'y4':y4, 'y5':y5, 'y6':y6, 'y7':y7, 'y8':y8,'y9':y9})

##### Current vs Time plot #####

fig = plt.figure()
ax = plt.subplot(111)
plt.plot ('t','y1', data=df,color='black',label='10K')
plt.text(1600,-1, '10K', horizontalalignment='left', size='small', color='black')
plt.plot ('t','y2', data=df,color='black',label='1K')
plt.text(1600,1, '1K', horizontalalignment='left', size='small', color='black')
plt.plot ('t','y3', data=df,color='black',label='100 Ohm')
plt.text(1600,5, '100 Ohm', horizontalalignment='left', size='small', color='black')
plt.plot ('t','y4', data=df,color='black',label='56 Ohm')
plt.text(1600,6.5, '56 Ohm', horizontalalignment='left', size='small', color='black')
plt.plot ('t','y5', data=df,color='black',label='33 Ohm')
plt.text(1600,8.5, '33 Ohm', horizontalalignment='left', size='small', color='black')
plt.plot ('t','y6', data=df,color='black',label='22 Ohm')
plt.text(1600,10.5, '22 Ohm', horizontalalignment='left', size='small', color='black')
plt.plot ('t','y7', data=df,color='black',label='12 Ohm')
plt.text(1600,12.5, '12 Ohm', horizontalalignment='left', size='small', color='black')
plt.plot ('t','y8', data=df,color='black',label='6 Ohm')
plt.text(1600,15.5, '6 Ohm', horizontalalignment='left', size='small', color='black')
plt.plot ('t','y9', data=df,color='black',label='SC')
plt.text(1400,21, 'Short Circuit Current', horizontalalignment='left', size='small', color='black')
plt.title('Current vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Current (mA)')



#label_x = 1000
#label_y = 1
#
#arrow_x = 1000
#arrow_y = 5
#
#arrow_properties = dict(
#    facecolor="black", width=0.5,
#    headwidth=4, shrink=0.1)
#
#plt.annotate(
#    "R Decreasing", xy=(arrow_x, arrow_y),
#     xytext=(label_x, label_y),
#     arrowprops=arrow_properties)
#
#
ax.grid(True)
#ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
#          fancybox=True, shadow=True, ncol=5)
  

##### Voltage vs Time plot #####

v1=(10*1000*y1)
v2=(1*1000*y2)
v3=(100*y3)
v4=(56*y4)
v5=(33*y5)
v6=(22*y6)
v7=(12*y7)
v8=(6*y8)
v9=(0*y9)

df2=pd.DataFrame ({'t':t,'v1':v1,'v2':v2,'v3':v3,'v4':v4,'v5':v5,'v6':v6,'v7':v7,'v8':v8,'v9':v9})

fig = plt.figure()
bx = plt.subplot(111)
plt.plot ('t','v1', data=df2,color='black',label='10 Kohm')
plt.text(1600,740, '10 KOhm', horizontalalignment='left', size='small', color='black')
plt.plot ('t','v2', data=df2,color='black',label='1 Kohm')
plt.text(1600,600, '1 KOhm', horizontalalignment='left', size='small', color='black')
plt.plot ('t','v3', data=df2,color='black',label='100 ohm')
plt.text(1600,450, '100 Ohm', horizontalalignment='left', size='small', color='black')
plt.plot ('t','v4', data=df2,color='black',label='56 ohm')
plt.text(1600,320, '56 Ohm', horizontalalignment='left', size='small', color='black')
plt.plot ('t','v5', data=df2,color='black',label='33 ohm')
plt.text(1600,250, '33 Ohm', horizontalalignment='left', size='small', color='black')
plt.plot ('t','v6', data=df2,color='black',label='22 ohm')
plt.text(1600,200, '22 Ohm', horizontalalignment='left', size='small', color='black')
plt.plot ('t','v7', data=df2,color='black',label='12 ohm')
plt.text(1600,115, '12 Ohm', horizontalalignment='left', size='small', color='black')
plt.plot ('t','v8', data=df2,color='black',label='06 ohm')
plt.text(1600,58, '6 Ohm', horizontalalignment='left', size='small', color='black')
plt.plot ('t','v9', data=df2,color='black',label='SC')
plt.text(1400,-30, 'Short Circuit Voltage', horizontalalignment='left', size='small', color='black')

#plt.text(1600,730, 'OC Voltage', horizontalalignment='left', size='small', color='red')
#plt.text(1600,5, 'SC Voltage', horizontalalignment='left', size='small', color='red')
#
#label_x = 1000
#label_y = 450
#
#arrow_x = 1000
#arrow_y = 350
#
#arrow_properties = dict(
#    facecolor="black", width=0.5,
#    headwidth=4, shrink=0.1)
#
#plt.annotate(
#    "R Decreasing", xy=(arrow_x, arrow_y),
#     xytext=(label_x, label_y),
#     arrowprops=arrow_properties)


bx.grid(True)
plt.title('Voltage vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (mV)')
#bx.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
#          fancybox=True, shadow=True, ncol=5)


##### I-V Characteristics #####

fig = plt.figure()
cx = plt.subplot(111)
A=[v1[data_size-1],v2[data_size-1],v3[data_size-1],v4[data_size-1],v5[data_size-1],v6[data_size-1],v7[data_size-1],v8[data_size-1],v9[data_size-1]]
B=[y1[data_size-1],y2[data_size-1],y3[data_size-1],y4[data_size-1],y5[data_size-1],y6[data_size-1],y7[data_size-1],y8[data_size-1],y9[data_size-1]]
plt.plot (A,B,marker="*")
plt.title('I-V Characteristics')
plt.xlabel('V (mV)')
plt.ylabel('I (mA)')
cx.grid(True)

##### Curve-Fit #####

def func (x,a,b):
#    return a*np.exp(-b*x)
    return a*x+b
init=[1,1]
p,q=curve_fit(func,A,B,init)
print(p)
xFit=np.arange(0,800,0.01)

fig = plt.figure()
ccx = plt.subplot(111)
plt.plot(A, B,'bo',label='Measured data') 
plt.plot(xFit, func(xFit,*p),'r',label='fit') 
plt.title('I-V Charecteristics')
plt.xlabel('V (mV)')
plt.ylabel('I (mA)')
plt.legend()
ccx.grid(True)
z=func(xFit,*p)
R=100/(z[30000]-z[40000])




##### P-V Characteristics #####

fig = plt.figure()
dx = plt.subplot(111)
P=[v1[data_size-1]*y1[data_size-1]/1000,v2[data_size-1]*y2[data_size-1]/1000,v3[data_size-1]*y3[data_size-1]/1000,v4[data_size-1]*y4[data_size-1]/1000,v5[data_size-1]*y5[data_size-1]/1000,v6[data_size-1]*y6[data_size-1]/1000,v7[data_size-1]*y7[data_size-1]/1000,v8[data_size-1]*y8[data_size-1]/1000,v9[data_size-1]*y9[data_size-1]/1000]
plt.plot (A,P,marker="*", color='red')
plt.title('P-V Charecteristics')
plt.xlabel('V (mV)')
plt.ylabel('P (mW)')
dx.grid(True)


##### Curve-Fit #####

def func (x,a,b,c):
#    return a*np.exp(-b*x)
    return a+b*x+0.5*a*x**2
init=[1,1,-10]
p,q=curve_fit(func,A,P,init)
print(p)
xFit=np.arange(0,800,0.01)

fig = plt.figure()
ddx = plt.subplot(111)
plt.plot(A, P,'bo',label='Measured data') 
plt.plot(xFit, func(xFit,*p),'r',label='fit')

m=(func(xFit,*p))
max_y=max(func(xFit,*p))
max_x = xFit[m.argmax()]
print (max_x, max_y)

plt.text(max_x+10,2.5, 'MPP', horizontalalignment='left', size='large', color='red')

plt.vlines(max_x, 0, y, linestyle="dashed")
plt.title('P-V Charecteristics')
plt.xlabel('V (mV)')
plt.ylabel('P (mW)')
ddx.set_ylim(ymin=0)
plt.legend()
ddx.grid(True)


##### P-R Characteristics #####

fig = plt.figure()
ex = plt.subplot(111)
R= [10000,1000,100,56,33,22,12,6,0]
plt.plot (R,P,marker="*", color='green')
ex.set_xscale('log')
plt.title('P-R Charecteristics')
plt.xlabel('R (Ohm)')
plt.ylabel('P (mW)')
ex.grid(True)


fig = plt.figure()
ax1_load2 = plt.subplot(111)
y_load2=load_2.iloc[1:60,1]
t1_load2=load_2.iloc[1:60,0]
d_load2=pd.DataFrame ({'t':t1_load2,'y':y_load2})
#rolling_mean = d_dc.y.rolling(window=100).mean()
#plt.plot(d_l.t, d_l.y, label='AMD')
plt.plot(d_load2.t, d_load2.y, color='blue')
#plt.axvspan(0, 20, color='grey', alpha=0.5)
#plt.text(7.5,160, 'Startup', horizontalalignment='left', size='small', color='red')
#plt.text(26,160, 'Message 1', horizontalalignment='left', size='small', color='red')
#plt.text(46,160, 'Message 2', horizontalalignment='left', size='small', color='red')
#plt.text(65,160, 'Message 3', horizontalalignment='left', size='small', color='red')

plt.title('Load Resistance')
plt.xlabel('Time (s)')
plt.ylabel('Resistance (Ohm)')
plt.grid(True)





#
#saved_column1 = variable['Channel 1 (ADC)']
#
#
#
#
##rows =variable.sample(frac =.1) 
#
#import matplotlib.pyplot as plt
#plt.close('all')
#plt.figure(1)
#variable.plot(x='Time (s)', y='Channel 1 (ADC)')
#
#
#saved_column2 = variable1['Channel 1 (ADC)']
#plt.figure(1)
#variable1.plot(x='Time (s)', y='Channel 1 (ADC)')
#
#
#saved_column3 = variable2['Channel 1 (ADC)']
#plt.figure(1)
#variable2.plot(x='Time (s)', y='Channel 1 (ADC)')
#
#A=pd.append(variable,saved_column2,saved_column3)