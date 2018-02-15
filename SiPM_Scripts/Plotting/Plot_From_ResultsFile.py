from matplotlib.pyplot import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def GetSavePath(Date):
    savepath = '/home/gebhardt/00_SiPM_MPIK/scripts/Data/Date'+Date+'/'
    return savepath

def log_10_product(x, pos):
    """The two args are the value and tick position.
    Label ticks with the product of the exponentiation"""
    return '{:.2e}'.format(x)
formatter = FuncFormatter(log_10_product)

np.set_printoptions(precision=3)
np.set_printoptions(formatter={'float ':lambda x: format(x, '6.3E')})
#step1str = 'FixedGuess3mV520IntWinMPD63DeltaX'# 1 'FixedGuess' 2 'Gainx0p25' 3 'FixedValXXmV'
#step2str = 'FixedGuess3mV520IntWinMPD63Xbar'# 1 'FixedGuess' 2 'Gainx0p25' 3 'FixedValXXmV'
#step2str = 'Gainx0p25'# 1 'FixedGuess' 2 'Gainx0p25' 3 'FixedValXXmV'
#step3str = 'FixedVal2mV'# 1 'FixedGuess' 2 'Gainx0p25' 3 'FixedValXXmV'


savestr = 'Calc_CHECS_px15_OV_Complete'


#Temperatures
date = ['']#,'1711161','2111162']
datafolderpath = ['FixedGuess3mV520IntWinMPD25']
stepstrfull = ['FixedGuess3mV520IntWinMPD25']
regrnamefull = ['FixedGuess3mV520IntWinMPD25_Combined']
#namefull = ['FixedGuess3mV520IntWinMPD25']
namefull = ['15','20','25','30','35']

'''
# Analysis Parameters
#datafolderpath = ['FixedGuess3mV520IntWinMPD63','FixedGuess3mV520','FixedGuess3mV520IntWinMPD25','FixedGuess3mV520IntWinMPD13','FixedGuess3mV55IntWinMPD25']
#stepstrfull = ['FixedGuess3mV520IntWinMPD63','FixedGuess3mV520IntWin','FixedGuess3mV520IntWinMPD25','FixedGuess3mV520IntWinMPD13','FixedGuess3mV55IntWinMPD25']
#regrnamefull = ['FixedGuess3mV520IntWinMPD63_Combined','FixedGuess3mV520IntWin_Combined','FixedGuess3mV520IntWinMPD25_Combined','FixedGuess3mV520IntWinMPD13_Combined','FixedGuess3mV55IntWinMPD25_Combined']
#namefull = ['3mV520IntWinMPD63','3mV520IntWinMPD100','3mV520IntWinMPD25','3mV520IntWinMPD13','3mV55IntWinMPD25']
'''

'''
Area and Height
Date = '1611161'
datafolderpath = ['FixedGuess3mV520IntWinMPD25','HeightFixedGuess3mV520IntWinMPD25']
stepstrfull = ['FixedGuess3mV520IntWinMPD25','HeightFixedGuess3mV520IntWinMPD25']
regrnamefull = ['FixedGuess3mV520IntWinMPD25_Combined','HeightFixedGuess3mV520IntWinMPD25_Combined']
namefull = ['Area','Height']
'''




c = ['b','r','g','k','magenta']


colorticker = 0
#for a,stepstrlong in enumerate(stepstrfull):  #one experiment
for a,Date in enumerate(date): #Temps
    a = 0 #single variable-setup
 
    #stepstr = stepstrlong[10:]
    j = colorticker
    AbsGain1 = []
    AbsGain2 = []

    savepath = GetSavePath(Date)
    datalist1 = np.load(savepath+str(datafolderpath[a])+'/'+str(stepstrfull[a])+Date+'DataPointsPlot.npy') #[Temp,Vbr,Volt,OVlist,Gain,dDCR,dOCT,Footer,AbsGain,deltax,xbar,combined])
   
    Temp1 = datalist1[0]
    Vbr1xbar = datalist1[1]
    Gain1 = datalist1[4][0]
    Volt1 = datalist1[2];Volt1[Gain1 == 0 ] = 'nan' 
    OV1xbar = datalist1[3][0];OV1xbar[Gain1 == 0 ] = 'nan' 
    #check if any Gain-value is previously set to 0
    dDCR1 = datalist1[5][0];dDCR1[Gain1 == 0 ] = 'nan' 
    dOCT1 = datalist1[6][0];dOCT1[Gain1 == 0 ] = 'nan' 
    combined = datalist1[11][0];combined[Gain1 == 0] = 'nan' 
    Gain1[Gain1 == 0] = 'nan' 

    print Date,' ',Temp1
    #print Vbr1




    allDCRdatalist = np.load(savepath+str(datafolderpath[a])+'/'+str(stepstrfull[a])+'DCRfromtotal.npy') #[Temp,Vbr,Volt,OVlist,Gain,dDCR,dOCT,Footer,AbsGain])
    #print allDCRdatalist
    allDCR1 = allDCRdatalist[2][0];allDCR1[allDCR1 == 0] = 'nan' 

    HAMOCT = [7.5/100,28.75/100,52.5/100,77.5/100]
    HAMVB = [1,2,3,4]


    
    regrfile = np.load(savepath+str(datafolderpath[a])+'/'+str(regrnamefull[a])+'AreaRelGainRegrLineData.npy')
    slope = regrfile[1][0] #T
    inter = regrfile[2][0] #T
    print slope,inter

    OV1 = np.zeros_like(Volt1)
    Vbr1 = -inter/slope
    print Vbr1
    for i in range(len(Volt1)):
        OV1[i]=Volt1[i]-Vbr1



    #AbsGain
    wl_bin = 4*2.5     
    wr_bin = 5*2.5
    pulse_geom = 1.3
    conv_factor = (wl_bin+wr_bin)*pulse_geom
    for i in range(len(Gain1)):
        AbsGain1.append(Gain1[i]*1000/conv_factor) 




    fig1 = plt.figure(1)
    ax1 = fig1.add_axes([0.1, 0.1, 0.8, 0.8])
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(1e6))
    ax1.yaxis.set_major_formatter(formatter)

    ax1.grid(True)
    #ax1.set_ylim(0,7e6)

    ax1.set_title('DCR vs OV '+str(savestr))
    ax1.set_xlabel('OverVoltage [V]')
    ax1.set_ylabel('DCR [Hz]')
    for i in range(len(Temp1)):
        ax1.scatter(OV1,dDCR1,c=c[i+j],s=60,label=namefull[j])
        ax1.plot(OV1,dDCR1,c=c[i+j],ls='solid')
        ax1.scatter(OV1,allDCR1,c=c[i+j],s=100,marker='+')
        ax1.plot(OV1,allDCR1,c=c[i+j],ls='dashed')
    ax1.text(0.5,5e6,'allDCR dashed, star = regr', style='italic',bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})
    if a == 0:
        datalist1 = np.load(savepath+str(datafolderpath[a])+'/'+str(stepstrfull[a])+'AreaNPYDataTableRegr.npy') #[T,Vb,TGainl,TDCRl,TOCTl]))
        dDCRregr = datalist1[3][0]
        ax1.scatter(OV1,dDCRregr,c=c[i+j],s=200,label=namefull[j]+'Regr',marker=(5,1))
        ax1.plot(OV1,dDCRregr,c=c[i+j],ls='solid',linestyle=':')

    plt.legend(loc=2)

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()   


    fig2= plt.figure(2)
    ax2 = fig2.add_axes([0.1, 0.1, 0.8, 0.8])
    ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax2.grid(True)
    ax2.set_ylim(0,0.9)

    ax2.set_title('OCT vs OV '+str(savestr))
    ax2.set_xlabel('OverVoltage [V]')
    ax2.set_ylabel('OCT ')
    for i in range(len(Temp1)):
        ax2.scatter(OV1,dOCT1,c=c[i+j],s=60,label=namefull[j])
        ax2.plot(OV1,dOCT1,c=c[i+j])
    if j == 0:
        ax2.plot(HAMVB,HAMOCT,color = 'lightgrey',linewidth = 5,label='HAM')


    plt.legend(loc=2)

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()

    x = np.arange(0,6)


    fig3= plt.figure(3)
    ax3 = fig3.add_axes([0.1, 0.1, 0.8, 0.8])
    ax3.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    ax3.grid(True)
    #ax3.set_ylim(0,0.3)
    ax3.set_title('Gain vs OV '+str(savestr))
    ax3.set_xlabel('OverVoltage [V]')
    ax3.set_ylabel('Gain [V*bins] [V] ')
    for i in range(len(Temp1)):
        #ax3.scatter(64.3,0)
        ax3.scatter(OV1,combined,c=c[i+j])

        ax3.scatter(OV1,Gain1,c=c[i+j],s=60,label=namefull[j])
        ax3.plot(OV1,Gain1,c=c[i+j])
        #ax3.plot(x,(x)*slope+inter,c=c[i+j],label=namefull[j]+' '+str(     '%.2f' %(-inter/slope))      ,linestyle='--') 
    xlims = ax3.get_xlim()
    ax3.text(0.5,0.15,'small dots: combined Gain (xbar and deltax) used for regr',style='italic',bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})

    plt.legend(loc=2)

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()


    fig4= plt.figure(4)
    ax4 = fig4.add_axes([0.1, 0.1, 0.8, 0.8])
    ax4.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    ax4.grid(True)
    ax4.set_xlim(xlims)
    ax4.set_title('combined Gain (xbar and deltax) used for regr '+str(savestr))
    ax4.set_xlabel('OverVoltage [V]')
    ax4.set_ylabel('Gain [V*bins] [V]')
    for i in range(len(Temp1)):
        #ax4.scatter(64.3,0)
        ax4.scatter(OV1,combined,c=c[i+j],s=60,label=namefull[j])
        ax4.plot(OV1,combined,c=c[i+j])

        #ax4.plot(x,(x)*slope+inter,c=c[i+j],label=namefull[j]+' '+str(     '%.2f' %(-inter/slope))      ,linestyle='--') 

    plt.legend(loc=2)




    plt.draw()
    colorticker += 1



    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()

plt.show()

fig1.savefig(savepath+str(savestr)+'_DCR_vs_Vb.pdf',format='pdf', bbox_inches='tight')
fig2.savefig(savepath+str(savestr)+'_OCT_vs_Vb.pdf',format='pdf', bbox_inches='tight')
fig3.savefig(savepath+str(savestr)+'_Gain_vs_Vb.pdf',format='pdf', bbox_inches='tight')
fig4.savefig(savepath+str(savestr)+'_Combined_Gain_vs_Vb.pdf',format='pdf', bbox_inches='tight')
