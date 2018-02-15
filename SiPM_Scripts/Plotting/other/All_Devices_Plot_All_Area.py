

import matplotlib                                                               
#matplotlib.use('Agg')  
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
import string
import numpy as np
from matplotlib.pyplot import cm
import matplotlib.ticker as ticker
from scipy.stats import linregress as lreg
import pandas as pd
import statsmodels.api as sm
import numpy.ma as ma
import pdb


def log_10_product(x, pos):
    """The two args are the value and tick position.
    Label ticks with the product of the exponentiation"""
    return '{:.3e}'.format(x)

def GetNames(destT,destV,Date):

    path = '/home/gebhardt/00_SiPM_MPIK/scripts/Data/Date'+Date+'/'+str(destT)+'deg/'+str(destV)+'V/'
    name = 'HAM_T_' + str(destT) + '_Vb' + str(destV) + '.trc'
    compl_name = os.path.join(path,name)
    return compl_name

def GetSavePath(Date):
    savepath = '/home/gebhardt/00_SiPM_MPIK/scripts/Data/Date'+Date+'/'
    return savepath


np.set_printoptions(precision=3)
np.set_printoptions(formatter={'float ':lambda x: format(x, '6.3E')})
formatter = FuncFormatter(log_10_product)

date = ['0510161','0610161','2809163','1110164']
IDlist = ['6050CS-10432','6025CS-10487','FJ60035-160205-21','6050HWB-LVR-LCT']

ignorel = 0
ignorer = 30
ignore = 0


tickeriter = 0
for i,Date in enumerate(date):

    ignorel = 0
    ignorer = 30    

    if i ==3:
        ignorel = 0
        ignorer = 15
    if i ==1:
        ignorel = 2
        ignorer = 30



    
    savepath = GetSavePath(Date)
    datalist = np.load(savepath+Date+'DataPointsPlot.npy')

    c = ['red','red','magenta','blue']#,'blueviolet','mediumorchid','firebrick','red','orangered','gold']
    linestyles = ['-', '-.', '-', '-']

    Temp = datalist[0] #List of Temps
    Vbr = datalist[1]
    Volt = datalist[2] #List of Voltages
    OVlist = datalist[3]
    Gain = datalist[4] #NP array of lists of Gains per Voltage, per Temperature
    dDCR = datalist[5] #List of  DarkiRate
    dOCT = datalist[6] #List of  OpticalCrossTalk




    if len(Temp) == 1:
        j = 0
    else:
        j = 4
    #print Date,' ',Temp[j]




    x = np.arange(40,80)
    fig1 = plt.figure(1)
    ax1 = fig1.add_axes([0.1, 0.1, 0.8, 0.8])
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(1e6))
    ax1.yaxis.set_major_formatter(formatter)
    ax1.grid(True)
    ax1.set_title('DCR  vs OV')
    #ax1.set_ylim([1e4,1e6])
    ax1.set_xlabel('OverVoltage [V]')
    ax1.set_ylabel(' DCR [Hz]')

    ax1.scatter(OVlist[j][ignorel:ignorer:],dDCR[j][ignorel:ignorer:],c=c[i],s=60)
    ax1.plot(OVlist[j][ignorel:ignorer:],dDCR[j][ignorel:ignorer:],c=c[i],linestyle = linestyles[i],label=str(IDlist[i]))
    ylims = ax1.get_ylim()
    plt.legend(loc=2)
    figure = plt.gcf() # get current figure
    figure.set_size_inches(14, 7)




    fig3 = plt.figure(3)
    ax3 = fig3.add_axes([0.1, 0.1, 0.8, 0.8])
    ax3.yaxis.set_minor_locator(ticker.MultipleLocator(1e-1))
    #ax3.yaxis.set_major_formatter(formatter)
    ax3.grid(True)
    ax3.set_title('OCT  vs OV')
    ax3.set_xlabel('OverVoltage [V]')
    ax3.set_ylabel(' OCT')
    ax3.set_ylim([0.,1.])
    #colors=iter(cm.rainbow(np.linspace(0,1,n)))

    ax3.scatter(OVlist[j][ignorel:ignorer:],dOCT[j][ignorel:ignorer:],c=c[i],s=30)
    ax3.plot(OVlist[j][ignorel:ignorer:],dOCT[j][ignorel:ignorer:],c=c[i],linestyle = linestyles[i],label=str(IDlist[i]))
    ylims = ax3.get_ylim()
    plt.legend(loc=2)
    figure = plt.gcf() # get current figure
    figure.set_size_inches(14, 7)


    #fig1.savefig(savepath+'DCR_vs_OV.pdf',format='pdf', bbox_inches='tight')
    #fig3.savefig(savepath+'OCT_vs_OV.pdf',format='pdf', bbox_inches='tight')

    
    fig5 = plt.figure(5)
    ax5 = fig5.add_axes([0.1, 0.1, 0.8, 0.8])
    #ax5.yaxis.set_minor_locator(ticker.MultipleLocator(1e6))
    #ax5.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax5.grid(True)
    ax5.set_title('"Gain" vs Vb')
    #ax5.set_ylim([0.0,0.04])
    #ax5.set_xlim(63,71)
    ax5.set_xlabel('Bias Voltage [V]')
    ax5.set_ylabel('"Gain" [V*bins]')
    x = np.arange(50,62)
    null = np.zeros(len(Vbr))
    #colors=iter(cm.rainbow(np.linspace(0,1,n)))
    #for i in range(len(Temp)):
        #c = next(colors)
        #ax5.scatter(Vbrnofit,null,c='red',s=30)
        #ax5.scatter(Vbr,null,c='blue',s=30)

    ax5.scatter(Volt[ignorel:ignorer:],Gain[j][ignorel:ignorer:],c=c[i],s=30)
        #ax5.plot(x,slope[j]*x+intercept[j],c=c[i],linestyle = linestyles[i])
        #ax5.errorbar(Volt[ignorel:ignorer:],Gain[j][ignorel:ignorer:],c=c[i],linestyle = linestyles[i],yerr=Error[j][ignorel:ignorer:],fmt='none')
    plt.legend(loc=2)
    figure = plt.gcf() # get current figure
    figure.set_size_inches(14, 7)
    #ax5.text(0.,0.075,str(Vbr)+' V')
    ylims = ax5.get_ylim()
    


    #plt.figure()        
    fig6 = plt.figure(6)
    ax6 = fig6.add_axes([0.1, 0.1, 0.8, 0.8])
    ax6.yaxis.set_minor_locator(ticker.MultipleLocator(1e6))
    ax6.grid(True)
    ax6.set_title('"Gain" vs OV')
    ax6.set_xlabel('OverVoltage [V]')
    ax6.set_ylabel('"Gain" [V*bins]')
    #ax6.set_ylim(ylims)
    #ax6.set_xlim(66,71)
    #colors=iter(cm.rainbow(np.linspace(0,1,n)))
    #for i in range(len(Temp)):
        #c = next(colors)
    ax6.scatter(OVlist[j][ignorel:ignorer:],Gain[j][ignorel:ignorer:],c=c[i],s=30)
        #ax6.errorbar(OVlist[j][ignorel:ignorer:],Gain[j][ignorel:ignorer:],c=c[i],linestyle = linestyles[i],yerr=Error[j],fmt='none')
    ax6.plot(OVlist[j][ignorel:ignorer:],Gain[j][ignorel:ignorer:],c=c[i],linestyle = linestyles[i],label=str(IDlist[i]))
        #ax6.fill_between(OVlist[j], Gain[j]-Error[j], Gain[j]+Error[j],facecolor=c)
    plt.legend(loc=2)
    figure = plt.gcf() # get current figure
    figure.set_size_inches(14, 7)

    fig7 = plt.figure(7)
    ax7 = fig7.add_axes([0.1, 0.1, 0.8, 0.8])
    #ax7.yaxis.set_minor_locator(ticker.MultipleLocator(1e6))
    #ax7.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax7.set_title('"Gain" vs Vb')
    ax7.grid(True)
    #ax7.set_ylim([0.0,0.04])
    #ax7.set_xlim(66,71)
    ax7.set_xlabel('Bias Voltage [V]')
    ax7.set_ylabel('"Gain" [V*bins]')
    #for i in range(len(Temp)):
        #c = next(colors)
        #ax7.scatter(Vbr,null,c='red',s=30)
    ax7.scatter(Volt[ignorel:ignorer:],Gain[j][ignorel:ignorer:],c=c[i],s=30)
    ax7.plot(Volt[ignorel:ignorer:],Gain[j][ignorel:ignorer:],c=c[i],linestyle = linestyles[i],label=str(IDlist[i]))
        #ax7.plot(x,slope[j]*x+intercept[j],c=c[i],linestyle = linestyles[i])
        #ax7.errorbar(Volt,Gain[j],c=c[i],linestyle = linestyles[i],yerr=Error[j],fmt='none')
    plt.legend(loc=2)
    figure = plt.gcf() # get current figure
    figure.set_size_inches(14, 7)
    #ax7.text(0.,0.075,str(Vbr)+' V')
    ylims = ax7.get_ylim()
    

    
    #fig5.savefig(savepath+'RelGain_vs_Vb.pdf',format='pdf', bbox_inches='tight')
    #fig6.savefig(savepath+'RelGain_vs_OV.pdf',format='pdf', bbox_inches='tight')
    #fig7.savefig(savepath+'RelGain_vs_Vb_Clean.pdf',format='pdf', bbox_inches='tight')




    fig9 = plt.figure(9)
    ax9 = fig9.add_axes([0.1, 0.1, 0.8, 0.8])
    ax9.yaxis.set_minor_locator(ticker.MultipleLocator(1e6))
    ax9.yaxis.set_major_formatter(formatter)
    ax9.grid(True)
    ax9.set_title('DCR  vs Vb')
    #ax1.set_ylim([1e4,1e6])
    ax9.set_xlabel('BiasVoltage [V]')
    ax9.set_ylabel(' DCR [Hz]')
    #colors=iter(cm.rainbow(np.linspace(0,1,n)))
    #for i in range(len(Temp)):
        #c = next(colors)
    ax9.scatter(Volt[ignorel:ignorer:],dDCR[j][ignorel:ignorer:],c=c[i],s=40)
    ax9.plot(Volt[ignorel:ignorer:],dDCR[j][ignorel:ignorer:],c=c[i],linestyle = linestyles[i],label=str(IDlist[i]))
    ylims = ax9.get_ylim()
    plt.legend(loc=2)
    figure = plt.gcf() # get current figure
    figure.set_size_inches(14, 7)




    fig11 = plt.figure(11)
    ax11 = fig11.add_axes([0.1, 0.1, 0.8, 0.8])
    ax11.yaxis.set_minor_locator(ticker.MultipleLocator(1e-1))
    #ax11.yaxis.set_major_formatter(formatter)
    ax11.grid(True)
    ax11.set_title('OCT  vs Vb')    
    ax11.set_xlabel('BiasVoltage [V]')
    ax11.set_ylabel(' OCT')
    ax11.set_ylim([0.,1.])
    #colors=iter(cm.rainbow(np.linspace(0,1,n)))
    #for i in range(len(Temp)):
        #c = next(colors)
    ax11.scatter(Volt[ignorel:ignorer:],dOCT[j][ignorel:ignorer:],c=c[i],s=40,label=str(IDlist[i]))
    ax11.plot(Volt[ignorel:ignorer:],dOCT[j][ignorel:ignorer:],c=c[i],linestyle = linestyles[i],label=str(IDlist[i]))
    ylims = ax11.get_ylim()
    plt.legend(loc=2)
    figure = plt.gcf() # get current figure
    figure.set_size_inches(14, 7)




    #fig9.savefig(savepath+'DCR_vs_BiasVoltage.pdf',format='pdf', bbox_inches='tight')
    #fig11.savefig(savepath+'OCT_vs_BiasVoltage.pdf',format='pdf', bbox_inches='tight')




    fig15 = plt.figure(15)
    ax15 = fig15.add_axes([0.1, 0.1, 0.8, 0.8])
    #ax15.yaxis.set_minor_locator(ticker.MultipleLocator(1e2))
    #ax15.xaxis.set_minor_locator(ticker.MultipleLocator(5e1))
    #ax15.yaxis.set_major_formatter(formatter)
    ax15.grid(True)
    ax15.set_xlabel('Temp [$^{\circ}$C]')
    ax15.set_ylabel('BreakDownVoltage [V]')
    #ax15.scatter(Temp,Vbrnofit,c=c[i+1],s=40)

    ax15.scatter(Temp,Vbr,c=c[i],s=40)
    ylims = ax15.get_ylim()
    figure = plt.gcf() # get current figure
    figure.set_size_inches(14, 7)

    #fig15.savefig(savepath+'BreakDownVoltage_vs_Temp.pdf',format='pdf', bbox_inches='tight')



    plt.draw()


    tickeriter =1

#Footer = ['Temp(Exp)','Vbr(Exp)','Volt(Exp)','OV(T)','Gain(T)','DCR(T)','OCT(T)']
#np.save(savepath+'AreaRelGainRegrLineData',[Temp,slope,intercept])
#np.save(savepath+Date+'DataPointsPlot',[Temp,Vbr,Volt,OVlist,Gain,dDCR,dOCT,Footer])
    print 'Saved'



savepath = '/home/gebhardt/00_SiPM_MPIK/DeviceCompare/'
fig1.savefig(savepath+'DCR_vs_OV.pdf',format='pdf', bbox_inches='tight')
fig3.savefig(savepath+'OCT_vs_OV.pdf',format='pdf', bbox_inches='tight')
fig5.savefig(savepath+'RelGain_vs_Vb.pdf',format='pdf', bbox_inches='tight')
fig6.savefig(savepath+'RelGain_vs_OV.pdf',format='pdf', bbox_inches='tight')
fig7.savefig(savepath+'RelGain_vs_Vb_Clean.pdf',format='pdf', bbox_inches='tight')

fig9.savefig(savepath+'DCR_vs_BiasVoltage.pdf',format='pdf', bbox_inches='tight')
fig11.savefig(savepath+'OCT_vs_BiasVoltage.pdf',format='pdf', bbox_inches='tight')

fig15.savefig(savepath+'BreakDownVoltage_vs_Temp.pdf',format='pdf', bbox_inches='tight')

'''
xdata = np.ndarray(4,)
ydata = np.ndarray(4,)


for n in range(len(date)):
    line = fig1.gca().get_lines()[n]
    xdata[n] = line.get_xdata()
    ydata[n] = line.get_ydata()
print xdata
print ydata
'''

plt.show()
