

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

date = ['1411161']#,'2508163','1011161','1411161']#'2508162','2508163','0411161']

ignorel = 0#3
ignorer = 40#19
ignore = 0
#regrleft=5
#regrright=14


tickeriter = 0 #different colors in case of multiple experiments on the same canvas
for Date in date:
    
    savepath = GetSavePath(Date)
    datalist = np.load(savepath+'AreaNPYDataTable.npy')

    if Date == ('2508162')or('2508163'):

        regrleft=5
        regrright=15
    if Date == '1411161':
        regrleft=8
        regrright=20







    #colors=iter(cm.rainbow(np.linspace(0,1,10))) 
    #c = ['r','b','g','c','m','y','k','w']
    #c = ['blue','indigo','purple','blueviolet','mediumorchid','firebrick','red','orangered','gold']

    print tickeriter
    if tickeriter ==0:c = ['b','b']

    if tickeriter ==2:c = ['g','g']
    print c
    Temp = datalist[0] #List of Temps
    Volt = datalist[1] #List of Voltages
    Gain = datalist[2] #NP array of lists of Gains per Voltage, per Temperature
    Error = datalist[3]#List of Errors of the GainCalc
    dDCR = datalist[4] #List of  DarkCountRate
    cDCR = datalist[5] #List of clean DarkCountRate
    dOCT = datalist[6] #List of  OpticalCrossTalk
    cOCT = datalist[7]#List of clean OpticalCrossTalk
    '''
    for j in range(Gain.shape[0]):
        for i in range(Gain.shape[1]):
            if i==20:
                continue
            else:
                if 2 * (Gain[j][i]) < Gain[j][i+1] :
                    Gain[j][i] = None
    '''

    print 'Data :'
    print 'Temp ',Temp
    print 'Volt ',Volt
    print '"Gain" ',Gain
    print 'Error', Error
    print 'dDCR', dDCR
    print 'cDCR', cDCR
    print 'dOCT',dOCT
    print 'cOCT',cOCT

    #T = [26.]
    

    #Gain=ma.masked_where(unmaskedGain<0,unmaskedGain)
    #print 'Masked Gain ',Gain
    n = 7

    Vbr = []
    Vbr = []
    slope = []
    slope2 = []
    intercept = []
    intercept2 = []
    OVlist = []
    AbsGain = []
    err = []

    slopeRG = [] #RealGain
    interRG = []

    #"Gain" vs Vb regression line weighted least squares method

    for i in range(len(Temp)):  
        '''
        if i == 0:
            G_ign = 7
            Gainsave = Gain[0,G_ign]
            Gain[0,G_ign]=(Gain[0,G_ign-1]+Gain[0,G_ign+1])/2
        '''
        #regression line for Gain
        
        X = Volt[regrleft:regrright:]#4:1]
        #print X.dtype
        X = sm.add_constant(X)
        Y = Gain[i][regrleft:regrright:]#[2:]#4:1]
        #print Gain[i]
        y_err = Error[i][regrleft:regrright:]#[2:]#4:1]
        weights1 = pd.Series(y_err)
        #print type(weights1)
        wls_model1 = sm.WLS(Y,X, weights=1 / weights1)
        results1 = wls_model1.fit()
        print 'results ',results1.params
        #print results1.params.shape
        inter1,slo1 = results1.params
        slope.append(slo1)
        intercept.append(inter1)
        vbreak1 = -intercept[i]/slope[i]
        
        #vbreak1 = 64.2
        #if (vbreak<60) or (vbreak>70):vbreak = 60
        Vbr.append(vbreak1)
        #err.append()
        '''
        if i == 0:
            Gain[0,G_ign]=Gainsave
        '''
    print 'slope ',slope
    print 'intercept ',intercept
    np.save(savepath+'AreaRelGainRegrLineData',[Temp,slope,intercept])

   

    for i in range(len(Temp)):  
        OVlist.append(np.array(Volt) - np.array(Vbr[i]))
    Vbr = np.asarray(Vbr)
    print 'Vbr ',Vbr
    print 'OVlist ',OVlist



#conv_factor = float(2.75e+8)
    q = (1.602e-19)
    r = 50.
    t = 10.e-9
    f = t/(r*q)



    #conv_factor = float(1./17)   # not correct when using preamp
    for i in range(len(Gain)):
        AbsGain.append(Gain[i]/17*f) 




    x = np.arange(60,80)
    fig1 = plt.figure(1)
    ax1 = fig1.add_axes([0.1, 0.1, 0.8, 0.8])
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(1e6))
    ax1.yaxis.set_major_formatter(formatter)
    ax1.grid(True)
    ax1.set_title('DCR  vs OV')
    #ax1.set_ylim([1e4,1e6])
    ax1.set_xlabel('OverVoltage [V]')
    ax1.set_ylabel(' DCR [Hz]')
    #colors=iter(cm.rainbow(np.linspace(0,1,n)))
    #print colors
    for i in range(len(Temp)):
        #c = next(colors)
        ax1.scatter(OVlist[i][ignorel:ignorer:],dDCR[i][ignorel:ignorer:],c=c[i],s=60,label=' DCR vs OV at '+str(Temp[i])+'$^\circ$ '+str(Date))
        ax1.plot(OVlist[i][ignorel:ignorer:],dDCR[i][ignorel:ignorer:],c=c[i])
 
    #HIRO
    Hiro_OV = [2,3,4,5]
    Hiro_DCR = [3.5e6,4e6,5.5e6,6.25e6]
    plt.scatter(Hiro_OV,Hiro_DCR,c = 'grey',label='NU Japan')
    plt.plot(Hiro_OV,Hiro_DCR,c = 'grey')


    plt.legend(loc=2)


    fig3 = plt.figure(3)
    ax3 = fig3.add_axes([0.1, 0.1, 0.8, 0.8])
    #ax3.yaxis.set_minor_locator(ticker.MultipleLocator(1e-1))
    #ax3.yaxis.set_major_formatter(formatter)
    ax3.grid(True)
    ax3.set_title('OCT  vs OV')
    ax3.set_xlabel('OverVoltage [V]')
    ax3.set_ylabel(' OCT')
    #ax3.set_ylim([0.,1.])
    #colors=iter(cm.rainbow(np.linspace(0,1,n)))
    for i in range(len(Temp)):
        #c = next(colors)
        ax3.scatter(OVlist[i][ignorel:ignorer:],dOCT[i][ignorel:ignorer:],c=c[i],s=30,label=' OCT vs OV at '+str(Temp[i])+'$^\circ$ '+str(Date))
        ax3.plot(OVlist[i][ignorel:ignorer:],dOCT[i][ignorel:ignorer:],c=c[i])

    #HIRO
    Hiro_OV = np.array([2.0619,3.0619,4.0619,5.0619,6.0619,7.0619,8.0619,9.0619,10.0619,11.0619])
    Hiro_OCT = np.array([0.38,2.91,6.01,8.22,10.75,13.24,16.24,18.91,21.55,24.8])
    plt.scatter(Hiro_OV,Hiro_OCT/100,c = 'lightgrey',label='NU Japan')
    plt.plot(Hiro_OV,Hiro_OCT/100,c = 'lightgrey')

    #HAM
    Ham_OV = np.array([1,2,3,4])
    Ham_OCT = np.array([0.08,0.29,0.53,0.78])
    plt.scatter(Ham_OV,Ham_OCT,c = 'grey',label='Hamamatsu')
    plt.plot(Ham_OV,Ham_OCT,c = 'grey')


    plt.legend(loc=2)

    fig1.savefig(savepath+'DCR_vs_OV.pdf',format='pdf', bbox_inches='tight')
    fig3.savefig(savepath+'OCT_vs_OV.pdf',format='pdf', bbox_inches='tight')

    
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
    x = np.arange(64,72)
    null = np.zeros(len(Vbr))
    #colors=iter(cm.rainbow(np.linspace(0,1,n)))
    for i in range(len(Temp)):
        #c = next(colors)
        ax5.scatter(Vbr,null,c='blue',s=30)
        ax5.scatter(64.3,0,c='red',s=30)

        ax5.scatter(Volt[ignorel:ignorer:],Gain[i][ignorel:ignorer:],c=c[i],s=30,label=str(Temp[i])+'$^\circ$ '+str(Date))
        ax5.plot(x,slope[i]*x+intercept[i],c=c[i])
        #ax5.errorbar(Volt[ignorel:ignorer:],Gain[i][ignorel:ignorer:],c=c[i],yerr=Error[i][ignorel:ignorer:],fmt='none')
   
    plt.legend(loc=2)
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
    for i in range(len(Temp)):
        #c = next(colors)
        ax6.scatter(OVlist[i][ignorel:ignorer:],Gain[i][ignorel:ignorer:],c=c[i],s=30,label='Gain vs OV at '+str(Temp[i])+'$^\circ$ '+str(Date))
        #ax6.errorbar(OVlist[i][ignorel:ignorer:],Gain[i][ignorel:ignorer:],c=c[i],yerr=Error[i],fmt='none')
        ax6.plot(OVlist[i][ignorel:ignorer:],Gain[i][ignorel:ignorer:],c=c[i])
        #ax6.fill_between(OVlist[i], Gain[i]-Error[i], Gain[i]+Error[i],facecolor=c)
    plt.legend(loc=2)

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
    for i in range(len(Temp)):
        #c = next(colors)
        #ax7.scatter(Vbr,null,c='red',s=30)
        ax7.scatter(Vbr,null,c='blue',s=30)
        #ax7.scatter(64.2,null,c='red',s=30)

        ax7.scatter(64.3,0,c='red',s=30)
        ax7.scatter(Volt[ignorel:ignorer:],Gain[i][ignorel:ignorer:],c=c[i],s=30,label=str(Temp[i])+'$^\circ$ '+str(Date))
        ax7.plot(Volt[ignorel:ignorer:],Gain[i][ignorel:ignorer:],c=c[i])
        #ax7.plot(x,slope[i]*x+intercept[i],c=c[i])
        #ax7.errorbar(Volt,Gain[i],c=c[i],yerr=Error[i],fmt='none')
    plt.legend(loc=2)
    #ax7.text(0.,0.075,str(Vbr)+' V')
    ylims = ax7.get_ylim()
    


    fig8 = plt.figure(8)
    ax8 = fig8.add_axes([0.1, 0.1, 0.8, 0.8])
    #ax8.yaxis.set_major_formatter(formatter(0.001))
    #ax8.set_yticks(np.arange(-0.05, 0.05, 0.001))
    #ax8.yaxis.set_minor_locator(ticker.MultipleLocator(2.5e-1))
    ax8.grid(True)
    #ax8.set_ylim(ylims)
    #ax8.yaxis.set_major_formatter(formatter)
    ax8.set_xlabel('OverVoltage [V]')
    ax8.set_ylabel('Gain [V] {RelGain/Int Window}')
    #colors=iter(cm.rainbow(np.linspace(0,1,n)))
    for i in range(len(Temp)):
        #c = next(colors)
        ax8.scatter(OVlist[i][ignorel:ignorer:],AbsGain[i][ignorel:ignorer:],c=c[i],label='Gain at '+str(Temp[i])+'$^\circ$ '+str(Date))
   

    #HAM
    Ham_OV = [1,2,3,4]
    Ham_Gain = [5e5,1.2e6,1.7e6,2.3e6]
    plt.scatter(Ham_OV,Ham_Gain,c = 'grey',label='Hamamatsu')
    plt.plot(Ham_OV,Ham_Gain,c = 'grey')




    plt.legend(loc=2)
    
    fig5.savefig(savepath+'RelGain_vs_Vb.pdf',format='pdf', bbox_inches='tight')
    fig6.savefig(savepath+'RelGain_vs_OV.pdf',format='pdf', bbox_inches='tight')
    fig7.savefig(savepath+'RelGain_vs_Vb_Clean.pdf',format='pdf', bbox_inches='tight')
    fig8.savefig(savepath+'AbsGain_vs_OV.pdf',format='pdf', bbox_inches='tight')



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
    for i in range(len(Temp)):
        #c = next(colors)
        ax9.scatter(Volt[ignorel:ignorer:],dDCR[i][ignorel:ignorer:],c=c[i],s=40,label=' DCR vs Vb at '+str(Temp[i])+'$^\circ$ '+str(Date))
        ax9.plot(Volt[ignorel:ignorer:],dDCR[i][ignorel:ignorer:],c=c[i])
    ylims = ax9.get_ylim()
    plt.legend(loc=2)

 

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
    for i in range(len(Temp)):
        #c = next(colors)
        ax11.scatter(Volt[ignorel:ignorer:],dOCT[i][ignorel:ignorer:],c=c[i],s=40,label=' OCT vs Vb at '+str(Temp[i])+'$^\circ$ '+str(Date))
        ax11.plot(Volt[ignorel:ignorer:],dOCT[i][ignorel:ignorer:],c=c[i])
    ylims = ax11.get_ylim()
    plt.legend(loc=2)


    fig9.savefig(savepath+'DCR_vs_BiasVoltage.pdf',format='pdf', bbox_inches='tight')
    fig11.savefig(savepath+'OCT_vs_BiasVoltage.pdf',format='pdf', bbox_inches='tight')


    fig15 = plt.figure(15)
    ax15 = fig15.add_axes([0.1, 0.1, 0.8, 0.8])
    #ax15.yaxis.set_minor_locator(ticker.MultipleLocator(1e2))
    #ax15.xaxis.set_minor_locator(ticker.MultipleLocator(5e1))
    #ax15.yaxis.set_major_formatter(formatter)
    ax15.grid(True)
    ax15.set_xlabel('Temp [$^{\circ}$C]')
    ax15.set_ylabel('BreakDownVoltage [V]')
    ax15.scatter(Temp,Vbr,c=c[i+1],s=40)
    ax15.scatter(Temp,Vbr,c=c[i],s=40)
    ylims = ax15.get_ylim()

    fig15.savefig(savepath+'BreakDownVoltage_vs_Temp.pdf',format='pdf', bbox_inches='tight')






    plt.draw()
    tickeriter +=1
    OVlist = np.asarray(OVlist)

Footer = ['Temp(Exp)','Vbr(Exp)','Volt(Exp)','OV(T)','Gain(T)','DCR(T)','OCT(T)']
np.save(savepath+Date+'DataPointsPlot',[Temp,Vbr,Volt,OVlist,Gain,dDCR,dOCT,Footer])
print 'Saved ',savepath
plt.show()
