from matplotlib.pyplot import *
import matplotlib.pyplot as plt
import string
import numpy as np
from matplotlib.pyplot import cm
import matplotlib.ticker as ticker
from scipy.stats import linregress as lreg
import pandas as pd
import statsmodels.api as sm



def log_10_product(x, pos):
    """The two args are the value and tick position.
    Label ticks with the product of the exponentiation"""
    return '{:.3e}'.format(x)

def GetNames(destT,destV,Date):

    path = '/home/gebhardt/SiPM_MPIK/scripts/Data/Date'+Date+'/'+str(destT)+'deg/'+str(destV)+'V/'
    name = 'HAM_T_' + str(destT) + '_Vb' + str(destV) + '.trc'
    compl_name = os.path.join(path,name)
    return compl_name

def GetSavePath(Date):
    savepath = '/home/gebhardt/SiPM_MPIK/scripts/Data/Date'+Date+'/'
    return savepath


np.set_printoptions(precision=3)
np.set_printoptions(formatter={'float':lambda x: format(x, '6.3E')})
formatter = FuncFormatter(log_10_product)

Date = '030816' # will need loop later on, maybe runcards

savepath = GetSavePath(Date)
datalist = np.load(savepath+'HeightTempVbGainFile.npy')


Temp = datalist[0] #List of Temps
Volt = datalist[1] #List of Voltages
Gain = datalist[2] #NP array of lists of Gains per Voltage, per Temperature
Error = datalist[3] #List of Errors of the GainCalc
dDCR = datalist[4] #List of dirty DarkCountRate
cDCR = datalist[5] #List of clean DarkCountRate
dOCT = datalist[6] #List of dirty OpticalCrossTalk
cOCT = datalist[7] #List of clean OpticalCrossTalk

print 'Data :'
print 'Temp ',Temp
print 'Volt ',Volt
print '"Gain" ',Gain
print 'Error', Error
print 'dDCR', dDCR
print 'cDCR', cDCR
print 'dOCT',dOCT
print 'cOCT',cOCT
print

Vbr = []
Vbr2 = []
slope = []
slope2 = []
intercept = []
intercept2 = []
OVlist = []
AbsGain = []

slopeRG = [] #RealGain
interRG = []

#"Gain" vs Vb regression line weighted least squares method

for i in range(len(Temp)):
    #regression line for Vb = 68.5V - 71.5V , 0.5V steps
    X = Volt[0:2:1]
    X = sm.add_constant(X)
    Y = Gain[i][0:2:1]
    y_err = Error[i][0:2:1]
    weights = pd.Series(y_err)
    wls_model = sm.WLS(Y,X, weights=1 / weights)
    results = wls_model.fit()
    print 'results',results.params
    inter,slo = results.params
    slope.append(slo)
    intercept.append(inter)
    Vbr.append(-intercept[i]/slope[i])
    OVlist.append(np.array(Volt) - np.array(Vbr[i]))

Vbr = np.asarray(Vbr)
print 'Vbr ',Vbr
print 'OVlist ',OVlist

conv_factor = float(2.75e+8)
for i in range(len(Gain)):
    AbsGain.append(Gain[i]*conv_factor)
'''
#RealGain Vb regression line, to get conversion factor by comparison to SensL Datasheet
for i in range(len(Temp)):
    slo, inter, r_value, p_value, std_err = lreg(Volt,AbsGain[i])
    slopeRG.append(slo)
    interRG.append(inter)
'''
n = 3 

fig1 = plt.figure(1)
ax1 = fig1.add_axes([0.1, 0.1, 0.8, 0.8])
ax1.yaxis.set_minor_locator(ticker.MultipleLocator(1e6))
ax1.yaxis.set_major_formatter(formatter)
ax1.grid(True)
#ax1.set_ylim([1e4,1e6])
ax1.set_xlabel('OverVoltage [V]')
ax1.set_ylabel('Dirty DCR [Hz]')
x = np.arange(60,80)
colors=iter(cm.rainbow(np.linspace(0,1,n)))
for i in range(len(Temp)):
    c = next(colors)
    ax1.scatter(OVlist[i],dDCR[i],c=c,s=20,label='dirty DCR vs OV at'+str(Temp[i]))
ylims = ax1.get_ylim()
plt.legend(loc=2)


fig2 = plt.figure(2)
ax2 = fig2.add_axes([0.1, 0.1, 0.8, 0.8])
ax2.yaxis.set_minor_locator(ticker.MultipleLocator(1e6))
ax2.yaxis.set_major_formatter(formatter)
ax2.grid(True)
#ax2.set_ylim(ylims)
ax2.set_xlabel('OverVoltage [V]')
ax2.set_ylabel('Clean DCR [Hz]')
x = np.arange(60,80)
colors=iter(cm.rainbow(np.linspace(0,1,n)))
for i in range(len(Temp)):
    c = next(colors)
    ax2.scatter(OVlist[i],cDCR[i],c=c,s=20,label='clean DCR vs OV at'+str(Temp[i]))
plt.legend(loc=2)


fig3 = plt.figure(3)
ax3 = fig3.add_axes([0.1, 0.1, 0.8, 0.8])
ax3.yaxis.set_minor_locator(ticker.MultipleLocator(1e-1))
#ax3.yaxis.set_major_formatter(formatter)
ax3.grid(True)
ax3.set_xlabel('OverVoltage [V]')
ax3.set_ylabel('Dirty OCT')
#ax3.set_ylim([0.,1.])
x = np.arange(60,80)
colors=iter(cm.rainbow(np.linspace(0,1,n)))
for i in range(len(Temp)):
    c = next(colors)
    ax3.scatter(OVlist[i],dOCT[i],c=c,s=30,label='dirty OCT vs OV at'+str(Temp[i]))
ylims = ax3.get_ylim()
plt.legend(loc=2)


fig4 = plt.figure(4)
ax4 = fig4.add_axes([0.1, 0.1, 0.8, 0.8])
ax4.yaxis.set_minor_locator(ticker.MultipleLocator(1e-1))
#ax4.yaxis.set_major_formatter(formatter)
ax4.grid(True)
ax4.set_xlabel('OverVoltage [V]')
ax4.set_ylabel('Clean OCT')
#ax4.set_ylim(ylims)
x = np.arange(60,80)
colors=iter(cm.rainbow(np.linspace(0,1,n)))
for i in range(len(Temp)):
    c = next(colors)
    ax4.scatter(OVlist[i],cOCT[i],c=c,s=30,label='clean OCT vs OV at'+str(Temp[i]))
plt.legend(loc=2)

fig1.savefig(savepath+'DCRDirty_vs_OV.pdf',format='pdf', bbox_inches='tight')
fig2.savefig(savepath+'DCRClean_vs_OV.pdf',format='pdf', bbox_inches='tight')
fig3.savefig(savepath+'OCTDirty_vs_OV.pdf',format='pdf', bbox_inches='tight')
fig4.savefig(savepath+'OCTClean_vs_OV.pdf',format='pdf', bbox_inches='tight')


fig5 = plt.figure(5)
ax5 = fig5.add_axes([0.1, 0.1, 0.8, 0.8])
ax5.yaxis.set_minor_locator(ticker.MultipleLocator(25e-4))
ax5.xaxis.set_minor_locator(ticker.MultipleLocator(1))
ax5.grid(True)
#ax5.set_ylim([0.0,0.04])
#ax5.set_xlim(60,80)
ax5.set_xlabel('Bias Voltage [V]')
ax5.set_ylabel('"Gain" [V*bins]')
x = np.arange(60,80)
null = np.zeros(len(Vbr))
colors=iter(cm.rainbow(np.linspace(0,1,n)))
for i in range(len(Temp)):
    c = next(colors)
    ax5.scatter(Vbr,null,c='red',s=20)
    ax5.scatter(Volt,Gain[i],c=c)
    ax5.plot(x,slope[i]*x+intercept[i],c=c)
    ax5.errorbar(Volt,Gain[i],c=c,yerr=Error[i],fmt='none')
plt.legend(loc=2)
#ax5.text(0.,0.075,str(Vbr)+' V')
ylims = ax5.get_ylim()


fig6 = plt.figure(6)
ax6 = fig6.add_axes([0.1, 0.1, 0.8, 0.8])
ax6.yaxis.set_minor_locator(ticker.MultipleLocator(25e-4))
ax6.grid(True)
ax6.set_xlabel('OverVoltage [V]')
ax6.set_ylabel('"Gain" [V*bins]')
#ax6.set_ylim(ylims)
#ax6.set_xlim(1,9.1)
x = np.arange(60,80)
colors=iter(cm.rainbow(np.linspace(0,1,n)))
for i in range(len(Temp)):
    c = next(colors)
    ax6.scatter(OVlist[i],Gain[i],c=c,s=10,label='Gain vs OV at'+str(Temp[i]))
    ax6.errorbar(OVlist[i],Gain[i],c=c,yerr=Error[i],fmt='none')
    #ax6.fill_between(OVlist[i], Gain[i]-Error[i], Gain[i]+Error[i],facecolor=c)
plt.legend(loc=2)
#plt.show()

fig7 = plt.figure(7)
ax7 = fig7.add_axes([0.1, 0.1, 0.8, 0.8])
ax7.yaxis.set_minor_locator(ticker.MultipleLocator(2.5e6))
ax7.grid(True)
ax7.yaxis.set_major_formatter(formatter)
ax7.set_xlabel('Bias Voltage [V]')
ax7.set_ylabel('Gain')
x = np.arange(60,80)
colors=iter(cm.rainbow(np.linspace(0,1,n)))
for i in range(len(Temp)):
    c = next(colors)
    ax7.scatter(Volt,AbsGain[i],c=c,label='Gain at'+str(Temp[i]))
ylims = ax7.get_ylim()
plt.legend(loc=2)


fig8 = plt.figure(8)
ax8 = fig8.add_axes([0.1, 0.1, 0.8, 0.8])
ax8.yaxis.set_minor_locator(ticker.MultipleLocator(2.5e6))
ax8.grid(True)
#ax8.set_ylim(ylims)
ax8.yaxis.set_major_formatter(formatter)
ax8.set_xlabel('OverVoltage [V]')
ax8.set_ylabel('Gain')
x = np.arange(60,80)
colors=iter(cm.rainbow(np.linspace(0,1,n)))
for i in range(len(Temp)):
    c = next(colors)
    ax8.scatter(OVlist[i],AbsGain[i],c=c,label='Gain at'+str(Temp[i]))
plt.legend(loc=2)

fig5.savefig(savepath+'RelGain_vs_Vb.pdf',format='pdf', bbox_inches='tight')
fig6.savefig(savepath+'RelGain_vs_OV.pdf',format='pdf', bbox_inches='tight')
fig7.savefig(savepath+'AbsGain_vs_Vb.pdf',format='pdf', bbox_inches='tight')
fig8.savefig(savepath+'AbsGain_vs_OV.pdf',format='pdf', bbox_inches='tight')





fig9 = plt.figure(9)
ax9 = fig9.add_axes([0.1, 0.1, 0.8, 0.8])
ax9.yaxis.set_minor_locator(ticker.MultipleLocator(1e6))
ax9.yaxis.set_major_formatter(formatter)
ax9.grid(True)
#ax1.set_ylim([1e4,1e6])
ax9.set_xlabel('BiasVoltage [V]')
ax9.set_ylabel('Dirty DCR [Hz]')
x = np.arange(60,80)
colors=iter(cm.rainbow(np.linspace(0,1,n)))
for i in range(len(Temp)):
    c = next(colors)
    ax9.scatter(Volt,dDCR[i],c=c,s=20,label='dirty DCR vs Vb at'+str(Temp[i]))
ylims = ax9.get_ylim()
plt.legend(loc=2)


fig10 = plt.figure(10)
ax10 = fig10.add_axes([0.1, 0.1, 0.8, 0.8])
ax10.yaxis.set_minor_locator(ticker.MultipleLocator(1e6))
ax10.yaxis.set_major_formatter(formatter)
ax10.grid(True)
#ax10.set_ylim(ylims)
ax10.set_xlabel('BiasVoltage [V]')
ax10.set_ylabel('Clean DCR [Hz]')
x = np.arange(60,80)
colors=iter(cm.rainbow(np.linspace(0,1,n)))
for i in range(len(Temp)):
    c = next(colors)
    ax10.scatter(Volt,cDCR[i],c=c,s=20,label='clean DCR vs Vb at'+str(Temp[i]))
plt.legend(loc=2)



fig11 = plt.figure(11)
ax11 = fig11.add_axes([0.1, 0.1, 0.8, 0.8])
ax11.yaxis.set_minor_locator(ticker.MultipleLocator(1e-1))
#ax11.yaxis.set_major_formatter(formatter)
ax11.grid(True)
ax11.set_xlabel('BiasVoltage [V]')
ax11.set_ylabel('Dirty OCT')
#ax11.set_ylim([0.,1.])
x = np.arange(60,80)
colors=iter(cm.rainbow(np.linspace(0,1,n)))
for i in range(len(Temp)):
    c = next(colors)
    ax11.scatter(Volt,dOCT[i],c=c,s=20,label='dirty OCT vs Vb at'+str(Temp[i]))
ylims = ax11.get_ylim()
plt.legend(loc=2)


fig12 = plt.figure(12)
ax12 = fig12.add_axes([0.1, 0.1, 0.8, 0.8])
ax12.yaxis.set_minor_locator(ticker.MultipleLocator(1e-1))
#ax12.yaxis.set_major_formatter(formatter)
ax12.grid(True)
#ax12.set_ylim(ylims)
ax12.set_xlabel('BiasVoltage [V]')
ax12.set_ylabel('Clean OCT')
colors=iter(cm.rainbow(np.linspace(0,1,n)))
for i in range(len(Temp)):
    c = next(colors)
    ax12.scatter(Volt,cOCT[i],c=c,s=20,label='clean OCT vs Vb at'+str(Temp[i]))
plt.legend(loc=2)



fig9.savefig(savepath+'DCRDirty_vs_BiasVoltage.pdf',format='pdf', bbox_inches='tight')
fig10.savefig(savepath+'DCRClean_vs_BiasVoltage.pdf',format='pdf', bbox_inches='tight')
fig11.savefig(savepath+'OCTDirty_vs_BiasVoltage.pdf',format='pdf', bbox_inches='tight')
fig12.savefig(savepath+'OCTClean_vs_BiasVoltage.pdf',format='pdf', bbox_inches='tight')



'''
fig13 = plt.figure(13)
ax13 = fig13.add_axes([0.1, 0.1, 0.8, 0.8])
ax13.yaxis.set_minor_locator(ticker.MultipleLocator(1e5))
ax13.yaxis.set_major_formatter(formatter)
ax13.grid(True)
ax13.set_xlabel('Temp')
ax13.set_ylabel('Dirty DCR [Hz]')
for i in range(len(Temp)):
    ax13.scatter(Temp[i],dDCR[i][0],c='r',s=20,label='dirty DCR vs T at'+str(OVlist[i][0]))
    ax13.scatter(Temp[i],dDCR[i][6],c='b',s=20,label='dirty DCR vs T at'+str(OVlist[i][6]))
ylims = ax13.get_ylim()
#plt.legend(loc=2)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=3, mode="expand", borderaxespad=0.)

fig14 = plt.figure(14)
ax14 = fig14.add_axes([0.1, 0.1, 0.8, 0.8])
ax14.yaxis.set_minor_locator(ticker.MultipleLocator(1e5))
ax14.yaxis.set_major_formatter(formatter)
ax14.grid(True)
ax14.set_ylim(ylims)
ax14.set_xlabel('Temp')
ax14.set_ylabel('Clean DCR [Hz]')
for i in range(len(Temp)):
    #ax14.scatter(np.full((1,cDCR.shape[1]),Temp[i]),cDCR[i][0],c=c,s=20,label='clean DCR vs T at'+str(OVlist[i]))
    ax14.scatter(Temp[i],cDCR[i][0],c='r',s=20,label='clean DCR vs T at'+str(OVlist[i][0]))
    ax14.scatter(Temp[i],cDCR[i][6],c='b',s=20,label='clean DCR vs T at'+str(OVlist[i][6]))
#plt.legend(loc=2)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=6, mode="expand", borderaxespad=0.)


fig13.savefig(savepath+'DCRDirty_vs_Temp.pdf',format='pdf', bbox_inches='tight')
fig14.savefig(savepath+'DCRClean_vs_Temp.pdf',format='pdf', bbox_inches='tight')
'''

fig15 = plt.figure(15)
ax15 = fig15.add_axes([0.1, 0.1, 0.8, 0.8])
ax15.yaxis.set_minor_locator(ticker.MultipleLocator(1e-2))
ax15.xaxis.set_minor_locator(ticker.MultipleLocator(5e1))
#ax15.yaxis.set_major_formatter(formatter)
ax15.grid(True)
ax15.set_xlabel('Temp [$^{\circ}$C]')
ax15.set_ylabel('BreakDownVoltage [V]')
ax15.scatter(Temp,Vbr,c=c,s=20)
ylims = ax15.get_ylim()
'''
fig16 = plt.figure(16)
ax16 = fig16.add_axes([0.1, 0.1, 0.8, 0.8])
ax16.yaxis.set_minor_locator(ticker.MultipleLocator(1e5))
ax16.yaxis.set_major_formatter(formatter)
ax16.grid(True)
ax16.set_ylim(ylims)
ax16.set_xlabel('BiasVoltage')
ax16.set_ylabel('Clean OCT')
x = np.arange(20,40)
ax16.scatter(Volt,cOCT[i],c=c,s=20,label='clean OCT vs Vb at'+str(Temp[i]))
plt.legend(loc=2)
'''
fig15.savefig(savepath+'BreakDownVoltage_vs_Temp.pdf',format='pdf', bbox_inches='tight')


'''
fig16 = plt.figure(16)
ax16 = fig16.add_axes([0.1, 0.1, 0.8, 0.8])
ax16.yaxis.set_minor_locator(ticker.MultipleLocator(1e-2))
#ax16.yaxis.set_major_formatter(formatter)
ax16.grid(True)
#ax1.set_ylim([1e4,1e6])
ax16.set_xlabel('Temp')
ax16.set_ylabel('Dirty OCT')
ax16.set_ylim([0.,1.])
for i in range(len(Temp)):
    ax16.scatter(Temp[i],dOCT[i][0],c='r',s=20,label='dirty OCT vs T at'+str(OVlist[i][0]))
    ax16.scatter(Temp[i],dOCT[i][6],c='b',s=20,label='dirty OCT vs T at'+str(OVlist[i][6]))
ylims = ax16.get_ylim()
#plt.legend(loc=2)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=2, mode="expand", borderaxespad=0.)

fig17 = plt.figure(17)
ax17 = fig17.add_axes([0.1, 0.1, 0.8, 0.8])
ax17.yaxis.set_minor_locator(ticker.MultipleLocator(1e-2))
#ax17.yaxis.set_major_formatter(formatter)
ax17.grid(True)
ax17.set_ylim(ylims)
ax17.set_xlabel('Temp')
ax17.set_ylabel('Clean OCT')
for i in range(len(Temp)):
    #ax17.scatter(np.full((1,cDCR.shape[1]),Temp[i]),cDCR[i][0],c=c,s=20,label='clean DCR vs T at'+str(OVlist[i]))
    ax17.scatter(Temp[i],cOCT[i][0],c='r',s=20,label='clean OCT vs T at'+str(OVlist[i][0]))
    ax17.scatter(Temp[i],cOCT[i][6],c='b',s=20,label='clean OCT vs T at'+str(OVlist[i][6]))
#plt.legend(loc=2)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=2, mode="expand", borderaxespad=0.)



fig16.savefig(savepath+'OCTDirty_vs_Temp.pdf',format='pdf', bbox_inches='tight')
fig17.savefig(savepath+'OCTClean_vs_Temp.pdf',format='pdf', bbox_inches='tight')
'''





plt.show()
np.save(savepath+'HeightRelGainRegrLineData',[Temp,slope,intercept])

