from matplotlib.pyplot import *
import matplotlib.pyplot as plt
import string
import numpy as np
from matplotlib.pyplot import cm
import matplotlib.ticker as ticker
from scipy.stats import linregress as lreg

def log_10_product(x, pos):
    """The two args are the value and tick position.
    Label ticks with the product of the exponentiation"""
    return '{:.3e}'.format(x)

def GetNames(destT,destV,Date):

    path = '/home/gebhardt/SiPM_MPIK/scripts/Data/Date'+Date+'/'+str(destT)+'deg/'+str(destV)+'V/'
    name = 'SensL_T_' + str(destT) + '_Vb' + str(destV) + '.trc'
    compl_name = os.path.join(path,name)
    return compl_name

def GetSavePath(Date):
    savepath = '/home/gebhardt/SiPM_MPIK/scripts/Data/Date'+Date+'/'
    return savepath


np.set_printoptions(precision=3)
np.set_printoptions(formatter={'float':lambda x: format(x, '6.3E')})
formatter = FuncFormatter(log_10_product)

Date = '200616_1'
savepath = GetSavePath(Date)
datalist = np.load(savepath+'/TempVbGainFile.npy')


Temp = datalist[0]
Volt = datalist[1]
Gain = datalist[2]
print 'Temp ',Temp
print 'Volt ',Volt
print '"Gain" ',Gain
print
print
Vbr = []

slope = []
intercept = []
OVlist = []
AbsGain = []

slopeRG = [] #RealGain
interRG = []

#"Gain" vs Vb regression line
for i in range(len(Temp)):
    slo, inter, r_value, p_value, std_err = lreg(Volt,Gain[i])
    slope.append(slo)
    intercept.append(inter)
    Vbr.append(-intercept[i]/slope[i])
    OVlist.append(np.array(Volt) - np.array(Vbr[i]))
Vbr = np.asarray(Vbr)

#Gain vs OV plot from SensL Datasheet
GainOV = np.loadtxt('/home/gebhardt/SiPM_MPIK/scripts/Data/SensLFJ60035TSVGainOVPlot.csv',delimiter=',')
slopeGOV, interGOV, r_valueGOV, p_valueGOV, std_errGOV = lreg(GainOV)

for i in range(len(OVlist)):
    AbsGain.append(slopeGOV*(OVlist[i])+interGOV)
factor = np.divide(AbsGain,Gain)

#RealGain Vb regression line
for i in range(len(Temp)):
    slo, inter, r_value, p_value, std_err = lreg(Volt,AbsGain[i])
    slopeRG.append(slo)
    interRG.append(inter)
    





fig1 = plt.figure(1)
ax1 = fig1.add_axes([0.1, 0.1, 0.8, 0.8])
x = np.arange(1,6)
ax1.set_ylim([1e6,6e6])
ax1.grid(True)
ax1.yaxis.set_major_formatter(formatter)
ax1.plot(x,slopeGOV*x+interGOV,c='b',label='Vbr Gain')


fig2 = plt.figure(2)
ax2 = fig2.add_axes([0.1, 0.1, 0.8, 0.8])
ax2.grid(True)
ax2.set_xlabel('Bias Voltage in V')
ax2.set_ylabel('"Gain" in V*bins')
n = 3
x = np.arange(20,40)
null = np.zeros(len(Vbr))
colors=iter(cm.rainbow(np.linspace(0,1,n)))
for i in range(len(Temp)):
    c = next(colors)
    ax2.scatter(Vbr,null,c='red',s=20,label='Vbr')
    ax2.scatter(Volt,Gain[i],c=c,label='Original Data at'+str(Temp[i]))
    ax2.plot(x,slope[i]*x+intercept[i],c=c,label='Fitted line')
plt.legend()
ax2.text(np.mean(Vbr)+1,0.,str(Vbr)+' V')


fig3 = plt.figure(3)
ax3 = fig3.add_axes([0.1, 0.1, 0.8, 0.8])
ax3.grid(True)
ax3.yaxis.set_major_formatter(formatter)
ax3.set_xlabel('Bias Voltage in V')
ax3.set_ylabel('Gain')
n = 3
x = np.arange(20,40)
colors=iter(cm.rainbow(np.linspace(0,1,n)))
for i in range(len(Temp)):
    c = next(colors)
    ax3.scatter(Volt,AbsGain[i],c=c,label='Gain at'+str(Temp[i]))
plt.legend()


fig4 = plt.figure(4)
ax4 = fig4.add_axes([0.1, 0.1, 0.8, 0.8])
ax4.grid(True)
ax4.yaxis.set_major_formatter(formatter)
ax4.set_xlabel('Bias Voltage in V')
ax4.set_ylabel('Conversion Factor')
n = 3
x = np.arange(20,40)
colors=iter(cm.rainbow(np.linspace(0,1,n)))
for i in range(len(Temp)):
    c = next(colors)
    ax4.scatter(Volt,factor[i],c=c,label='Conversion Factor at'+str(Temp[i]))
plt.legend()
  



print 'OverVoltage'
print OVlist
print 'BreakdownVoltage'
print Vbr
print 'RealGain'
print AbsGain
print 'Conversion Factor'
print factor

plt.show()

np.save(savepath+'CONVTempVbGainFile',[Temp,Volt,AbsGain])
np.save(savepath+'CONVGainRegrLineData',[Temp,slopeRG,interRG])

