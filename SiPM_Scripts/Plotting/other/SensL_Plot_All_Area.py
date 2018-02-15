

import matplotlib                                                               
#matplotlib.use('Agg')   # comment out, only used for remote control
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
def notlog_10_product(x, pos):
    return '{:.1e}'.format(x)


def GetNames(destT,destV,Date):

    path = '/home/gebhardt/00_SiPM_MPIK/scripts/Data/Date'+Date+'/'+str(destT)+'deg/'+str(destV)+'V/'
    name = 'SensL_T_' + str(destT) + '_Vb' + str(destV) + '.trc'
    compl_name = os.path.join(path,name)
    return compl_name

def GetSavePath(Date):
    savepath = '/home/gebhardt/00_SiPM_MPIK/scripts/Data/Date'+Date+'/'
    return savepath


np.set_printoptions(precision=3)
np.set_printoptions(formatter={'float ':lambda x: format(x, '6.3E')})
formatter = FuncFormatter(log_10_product)
formatter2 = FuncFormatter(notlog_10_product)


date = ['2809161']#,'2809161','2809163']

ignorel = 0
ignorer = 7
ignore = 0

#tickeriter = 0
#color = ['blue','indigo','purple','blueviolet','mediumorchid','firebrick','red','orangered','gold']
color = ['blue','green','red']
IDlist = ['150925-05','160205-25','160205-21']










for count, Date in enumerate(date):
    c = []
    savepath = GetSavePath(Date)
    datalist = np.load(savepath+'AreaNPYDataTable.npy')

    #colors=iter(cm.rainbow(np.linspace(0,1,10))) 
    #c = ['r','b','g','c','m','y','k','w','grey']
    #print tickeriter
    #if tickeriter ==1:c = ['b','r']
    c.append(color[count])
    print c
    Temp = datalist[0] #List of Temps
    Volt = datalist[1] #List of Voltages
    Gain = datalist[2] #NP array of lists of Gains per Voltage, per Temperature
    Error = datalist[3]#List of Errors of the GainCalc
    dDCR = datalist[4] #List of  DarkCountRate
    cDCR = datalist[5] #List of clean DarkCountRate
    dOCT = datalist[6] #List of  OpticalCrossTalk
    cOCT = datalist[7]#List of clean OpticalCrossTalk

   
    print 'Data :'
    print 'Temp ',Temp
    print 'Volt ',Volt
    print '"Gain" ',Gain
    print 'Error', Error
    print 'dDCR', dDCR
    print 'cDCR', cDCR
    print 'dOCT',dOCT
    print 'cOCT',cOCT

    #T = [25.]
    #Volt = [26.,27.,28.,29.,30.,31.,32.]

    #Gain=ma.masked_where(unmaskedGain<0,unmaskedGain)
    #print 'Masked Gain ',Gain
    n = 12

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

    conv_factor = float(2.41e+8) #moved to test regression line on abs gain
    #conv_factor = float(1./6)
    for i in range(len(Gain)):
        AbsGain.append(Gain[i]*conv_factor)

    print 'AbsGain ',AbsGain




    for i in range(len(Temp)):
        #regression line for Vb = 28V - 30V , 0.5V steps
        X = Volt#[2:10:]#4:1]
        X = sm.add_constant(X)
        Y = AbsGain[i]#[2:10:]#[2:]#4:1]
        y_err = Error[i]#[2:10:]#[2:]#4:1]
        weights = pd.Series(y_err)
        wls_model = sm.WLS(Y,X, weights=1 / weights)
        results = wls_model.fit()
        print 'results',results.params
        inter,slo = results.params
        slope.append(slo)
        intercept.append(inter)
        vbreak = -intercept[i]/slope[i]
        #if (vbreak<60) or (vbreak>70):vbreak = 60
        Vbr.append(vbreak)
        Vbrmean = np.mean(Vbr)

        OVlist.append(np.array(Volt) - np.array(Vbr[i]))

    Vbr = np.asarray(Vbr)
    print 'Vbr ',Vbr
    print 'OVlist ',OVlist



    #conv_factor = float(2.41e+8)
    #conv_factor = float(1./6)
    #for i in range(len(Gain)):
    #    AbsGain.append(Gain[i]*conv_factor)





    '''
    #RealGain Vb regression line, to get conversion factor by comparison to SensL Datasheet
    for i in range(len(Temp)):
        slo, inter, r_value, p_value, std_err = lreg(Volt,AbsGain[i])
        slopeRG.append(slo)
        interRG.append(inter)
    '''

    x = np.arange(20,40)
    fig1 = plt.figure(1)
    ax1 = fig1.add_axes([0.1, 0.1, 0.8, 0.8])
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(5e5))
    ax1.yaxis.set_major_formatter(formatter)
    ax1.grid(True)
    ax1.set_title('DCR  vs OV at 20$^\circ$C')
    ax1.set_ylim([0,5e6])
    ax1.set_xlabel('OverVoltage [V]')
    ax1.set_ylabel(' DCR [Hz]')
    #colors=iter(cm.rainbow(np.linspace(0,1,n)))
    #print colors
    for i in range(len(Temp)):
        #c = next(colors)
        ax1.scatter(OVlist[i][ignorel:ignorer:],dDCR[i][ignorel:ignorer:],c=c[i],s=30,label=str(IDlist[count]))#Temp[i])+'$^\circ$C')
        ax1.plot(OVlist[i][ignorel:ignorer:],dDCR[i][ignorel:ignorer:],c=c[i])
    #ylims = ax1.get_ylim()
    plt.legend(loc=2)
    figure = plt.gcf() # get current figure
    figure.set_size_inches(7, 7)
    # when saving, specify the DPI
    #plt.savefig("myplot.png", dpi = 100)






    fig2 = plt.figure(2)
    ax2 = fig2.add_axes([0.1, 0.1, 0.8, 0.8])
    ax2.yaxis.set_minor_locator(ticker.MultipleLocator(1e6))
    ax2.yaxis.set_major_formatter(formatter)
    ax2.grid(True)
    #ax2.set_ylim(ylims)
    ax2.set_xlabel('OverVoltage [V]')
    ax2.set_ylabel('Clean DCR [Hz]')
    #colors=iter(cm.rainbow(np.linspace(0,1,n)))
    for i in range(len(Temp)):
        #c = next(colors)
        #print c
        ax2.scatter(OVlist[i],cDCR[i],c=c[i],s=30,label=str(IDlist[count]))#Temp[i])+'$^\circ$C')
    plt.legend(loc=2)
    figure = plt.gcf() # get current figure
    figure.set_size_inches(7, 7)

    fig3 = plt.figure(3)
    ax3 = fig3.add_axes([0.1, 0.1, 0.8, 0.8])
    ax3.yaxis.set_minor_locator(ticker.MultipleLocator(1e-1))
    #ax3.yaxis.set_major_formatter(formatter)
    ax3.grid(True)
    ax3.set_title('OCT  vs OV at 20$^\circ$C')
    ax3.set_xlabel('OverVoltage [V]')
    ax3.set_ylabel(' OCT')
    ax3.set_ylim([0.,1.])
    #colors=iter(cm.rainbow(np.linspace(0,1,n)))
    for i in range(len(Temp)):
        #c = next(colors)
        ax3.scatter(OVlist[i][ignorel:ignorer:],dOCT[i][ignorel:ignorer:],c=c[i],s=30,label=str(IDlist[count]))#Temp[i])+'$^\circ$C')
        ax3.plot(OVlist[i][ignorel:ignorer:],dOCT[i][ignorel:ignorer:],c=c[i])
    ylims = ax3.get_ylim()
    plt.legend(loc=2)
    figure = plt.gcf() # get current figure
    figure.set_size_inches(7, 7)

    fig4 = plt.figure(4)
    ax4 = fig4.add_axes([0.1, 0.1, 0.8, 0.8])
    ax4.yaxis.set_minor_locator(ticker.MultipleLocator(1e-1))
    #ax4.yaxis.set_major_formatter(formatter)
    ax4.grid(True)
    ax4.set_xlabel('OverVoltage [V]')
    ax4.set_ylabel('Clean OCT')
    ax4.set_ylim(ylims)
    #colors=iter(cm.rainbow(np.linspace(0,1,n)))
    for i in range(len(Temp)):
        #c = next(colors)
        ax4.scatter(OVlist[i],cOCT[i],c=c[i],s=30,label=str(IDlist[count]))#Temp[i])+'$^\circ$C')
    plt.legend(loc=2)
    figure = plt.gcf() # get current figure
    figure.set_size_inches(7, 7)

    fig1.savefig(savepath+'DCR_vs_OV.pdf',format='pdf', bbox_inches='tight',dpi = 100)
    #fig2.savefig(savepath+'DCRClean_vs_OV.pdf',format='pdf', bbox_inches='tight',dpi = 100)
    fig3.savefig(savepath+'OCT_vs_OV.pdf',format='pdf', bbox_inches='tight',dpi = 100)
    #fig4.savefig(savepath+'OCTClean_vs_OV.pdf',format='pdf', bbox_inches='tight',dpi = 100)

    
    fig5 = plt.figure(5)
    ax5 = fig5.add_axes([0.1, 0.1, 0.8, 0.8])
    ax5.yaxis.set_major_formatter(formatter2)

    ax5.yaxis.set_minor_locator(ticker.MultipleLocator(5e5))
    #ax5.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax5.grid(True)
    ax5.set_title('Gain vs Vb at 20$^\circ$C')
    #ax5.set_ylim([0.0,0.04])
    ax5.set_xlim(22,31)
    ax5.set_ylim(-1e6,1e7)
    ax5.set_xlabel('Bias Voltage [V]')
    ax5.set_ylabel('Gain ')
    #x = np.arange(64,72)
    null = np.zeros(len(Vbr))
    #colors=iter(cm.rainbow(np.linspace(0,1,n)))
    for i in range(len(Temp)):
        #c = next(colors)
        ax5.scatter(Volt[ignorel:ignorer:],AbsGain[i][ignorel:ignorer:],c=c[i],s=30,label=str(IDlist[count]))#Temp[i])+'$^\circ$C')
        ax5.plot(x,slope[i]*x+intercept[i],c=c[i])
        ax5.errorbar(Volt[ignorel:ignorer:],AbsGain[i][ignorel:ignorer:],c=c[i],yerr=Error[i][ignorel:ignorer:],fmt='none')
        if count == 2:
            ax5.scatter(Vbr[i],0,c=c[i],s=40,marker=(5, 2),label='Vbr')
    plt.legend(loc=2)
    #ax5.text(0.,0.075,str(Vbr)+' V')
    ylims = ax5.get_ylim()
    figure = plt.gcf() # get current figure
    figure.set_size_inches(7, 7)
    '''
    #fig5 = plt.figure(5)
    #ax5 = fig5.add_axes([0.1, 0.1, 0.8, 0.8])
    fig5, ax5 = plt.subplots(1,1, figsize=(15, 6))#, facecolor='w', edgecolor='k')
    #fig5.subplots_adjust(hspace = .5, wspace=.001)

    #ax5.yaxis.set_minor_locator(ticker.MultipleLocator(1e6))
    #ax5.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    #ax5.grid(True)
    #ax5.set_ylim([0.0,0.04])
    #ax5.set_xlim(60,80)
    #ax5.set_xlabel('Bias Voltage [V]')
    #ax5.set_ylabel('"Gain" [V*bins]')
    null = np.zeros(len(Vbr))
    colors=iter(cm.rainbow(np.linspace(0,1,n)))
    #ax5 = ax5.ravel()


    for i in range(len(Temp)):
        #c = next(colors)
        ax5.grid(True)
        ax5.scatter(Vbr[i],0,color='red',s=30)
        ax5.scatter(Volt,Gain[i],c=c[i],s=30,label=str(IDlist[count]))#Temp[i])+'$^\circ$C')
        ax5.plot(x,slope[i]*x+intercept[i],c=c[i])
        ax5.errorbar(Volt,Gain[i],c=c[i],yerr=Error[i],fmt='none')
        plt.legend(loc=2)
    #ax5.text(0.,0.075,str(Vbr)+' V')
    #ylims = ax5.get_ylim()
    '''
    

    #plt.figure()        
    fig6 = plt.figure(6)
    ax6 = fig6.add_axes([0.1, 0.1, 0.8, 0.8])
    ax6.yaxis.set_minor_locator(ticker.MultipleLocator(2.5e5))
    ax6.yaxis.set_major_formatter(formatter2)

    ax6.grid(True)
    ax6.set_title('Gain vs OV at 20$^\circ$C')
    ax6.set_xlabel('OverVoltage [V]')
    ax6.set_ylabel('Gain')
    #ax6.set_ylim(ylims)
    #ax6.set_xlim(66,71)
    #colors=iter(cm.rainbow(np.linspace(0,1,n)))
    for i in range(len(Temp)):
        #c = next(colors)
        ax6.scatter(OVlist[i][ignorel:ignorer:],AbsGain[i][ignorel:ignorer:],c=c[i],s=30,label=str(IDlist[count]))#Temp[i])+'$^\circ$C')
        #ax6.errorbar(OVlist[i][ignorel:ignorer:],Gain[i][ignorel:ignorer:],c=c[i],yerr=Error[i],fmt='none')
        ax6.plot(OVlist[i][ignorel:ignorer:],AbsGain[i][ignorel:ignorer:],c=c[i])
        #ax6.fill_between(OVlist[i], Gain[i]-Error[i], Gain[i]+Error[i],facecolor=c)
    plt.legend(loc=2)
    figure = plt.gcf() # get current figure
    figure.set_size_inches(7, 7)

    fig7 = plt.figure(7)
    ax7 = fig7.add_axes([0.1, 0.1, 0.8, 0.8])
    #ax7.yaxis.set_minor_locator(ticker.MultipleLocator(1e6))
    ax7.yaxis.set_major_formatter(formatter2)

    #ax7.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax7.set_title('Gain vs Vb')
    ax7.grid(True)
    #ax7.set_ylim([0.0,0.04])
    ax7.set_xlim(26.5,32.5)
    #ax7.set_ylim(0.,0.04)
    ax7.set_xlabel('Bias Voltage [V]')
    ax7.set_ylabel('Gain')
    for i in range(len(Temp)):
        #c = next(colors)
        #ax7.scatter(Vbr,null,c='red',s=30)
        ax7.scatter(Volt[ignorel:ignorer:],AbsGain[i][ignorel:ignorer:],c=c[i],s=30,label=str(IDlist[count]))#Temp[i])+'$^\circ$C')
        ax7.plot(Volt[ignorel:ignorer:],AbsGain[i][ignorel:ignorer:],c=c[i])
        #ax7.plot(x,slope[i]*x+intercept[i],c=c[i])
        #ax7.errorbar(Volt,Gain[i],c=c[i],yerr=Error[i],fmt='none')
    plt.legend(loc=2)
    #ax7.text(0.,0.075,str(Vbr)+' V')
    ylims = ax7.get_ylim()
    figure = plt.gcf() # get current figure
    figure.set_size_inches(7, 7)


    '''
    fig7 = plt.figure(7)
    ax7 = fig7.add_axes([0.1, 0.1, 0.8, 0.8])
    ax7.yaxis.set_minor_locator(ticker.MultipleLocator(2.5e6))
    ax7.grid(True)
    ax7.yaxis.set_major_formatter(formatter)
    ax7.set_xlabel('Bias Voltage [V]')
    ax7.set_ylabel('Gain')
    colors=iter(cm.rainbow(np.linspace(0,1,n)))
    for i in range(len(Temp)):
        #c = next(colors)
        ax7.scatter(Volt,AbsGain[i],c=c[i],label='Gain at '+str(Temp[i])+'$^\circ$C')
    ylims = ax7.get_ylim()
    plt.legend(loc=2)
    '''

    fig8 = plt.figure(8)
    ax8 = fig8.add_axes([0.1, 0.1, 0.8, 0.8])
    #ax8.yaxis.set_major_formatter(formatter(0.001))
    ax8.set_yticks(np.arange(-0.05, 0.05, 0.001))
    ax8.yaxis.set_minor_locator(ticker.MultipleLocator(2.5e5))
    ax8.yaxis.set_major_formatter(formatter2)

    ax8.grid(True)
    #ax8.set_xlim(26.5,31.5)
    #ax8.set_ylim(0.,0.006)
    #ax8.set_ylim(ylims)
    #ax8.yaxis.set_major_formatter(formatter)
    ax8.set_xlabel('OverVoltage [V]')
    ax8.set_ylabel('Gain')
    #colors=iter(cm.rainbow(np.linspace(0,1,n)))
    for i in range(len(Temp)):
        #c = next(colors)
        ax8.scatter(OVlist[i][ignorel:ignorer:],AbsGain[i][ignorel:ignorer:],c=c[i],s=30,label=str(IDlist[count]))#Temp[i])+'$^\circ$C')
    plt.legend(loc=2)
    figure = plt.gcf() # get current figure
    figure.set_size_inches(7, 7)


    fig5.savefig(savepath+'RelGain_vs_Vb.pdf',format='pdf', bbox_inches='tight')
    fig6.savefig(savepath+'RelGain_vs_OV.pdf',format='pdf', bbox_inches='tight')
    fig7.savefig(savepath+'RelGain_vs_Vb_Clean.pdf',format='pdf', bbox_inches='tight')
    #fig8.savefig(savepath+'AbsGain_vs_OV.pdf',format='pdf', bbox_inches='tight')





    fig9 = plt.figure(9)
    ax9 = fig9.add_axes([0.1, 0.1, 0.8, 0.8])
    ax9.yaxis.set_minor_locator(ticker.MultipleLocator(1e6))
    ax9.yaxis.set_major_formatter(formatter)
    ax9.grid(True)
    ax9.set_title('DCR  vs Vb at 20$^\circ$C')
    ax9.set_ylim([0,5e6])
    ax9.set_xlabel('BiasVoltage [V]')
    ax9.set_ylabel(' DCR [Hz]')
    #colors=iter(cm.rainbow(np.linspace(0,1,n)))
    for i in range(len(Temp)):
        #c = next(colors)
        ax9.scatter(Volt[ignorel:ignorer:],dDCR[i][ignorel:ignorer:],c=c[i],s=40,label=str(IDlist[count]))#Temp[i])+'$^\circ$C')
        ax9.plot(Volt[ignorel:ignorer:],dDCR[i][ignorel:ignorer:],c=c[i])
    ylims = ax9.get_ylim()
    plt.legend(loc=2)
    figure = plt.gcf() # get current figure
    figure.set_size_inches(7, 7)

    fig10 = plt.figure(10)
    ax10 = fig10.add_axes([0.1, 0.1, 0.8, 0.8])
    ax10.yaxis.set_minor_locator(ticker.MultipleLocator(1e6))
    ax10.yaxis.set_major_formatter(formatter)
    ax10.grid(True)
    #ax10.set_ylim(ylims)
    ax10.set_xlabel('BiasVoltage [V]')
    ax10.set_ylabel('Clean DCR [Hz]')
    #colors=iter(cm.rainbow(np.linspace(0,1,n)))
    for i in range(len(Temp)):
        #c = next(colors)
        ax10.scatter(Volt[ignorel:ignorer:],cDCR[i][ignorel:ignorer:],c=c[i],s=40,label=str(IDlist[count]))#Temp[i])+'$^\circ$C')
    plt.legend(loc=2)
    figure = plt.gcf() # get current figure
    figure.set_size_inches(7, 7)


    fig11 = plt.figure(11)
    ax11 = fig11.add_axes([0.1, 0.1, 0.8, 0.8])
    ax11.yaxis.set_minor_locator(ticker.MultipleLocator(1e-1))
    #ax11.yaxis.set_major_formatter(formatter)
    ax11.grid(True)
    ax11.set_title('OCT  vs Vb at 20$^\circ$C')    
    ax11.set_xlabel('BiasVoltage [V]')
    ax11.set_ylabel(' OCT')
    ax11.set_ylim([0.,1.])
    #colors=iter(cm.rainbow(np.linspace(0,1,n)))
    for i in range(len(Temp)):
        #c = next(colors)
        ax11.scatter(Volt[ignorel:ignorer:],dOCT[i][ignorel:ignorer:],c=c[i],s=40,label=str(IDlist[count]))#Temp[i])+'$^\circ$C')
        ax11.plot(Volt[ignorel:ignorer:],dOCT[i][ignorel:ignorer:],c=c[i])
    ylims = ax11.get_ylim()
    plt.legend(loc=2)
    figure = plt.gcf() # get current figure
    figure.set_size_inches(7, 7)

    fig12 = plt.figure(12)
    ax12 = fig12.add_axes([0.1, 0.1, 0.8, 0.8])
    ax12.yaxis.set_minor_locator(ticker.MultipleLocator(1e-1))
    #ax12.yaxis.set_major_formatter(formatter)
    ax12.grid(True)
    ax12.set_ylim(ylims)
    ax12.set_xlabel('BiasVoltage [V]')
    ax12.set_ylabel('Clean OCT')
    #colors=iter(cm.rainbow(np.linspace(0,1,n)))
    for i in range(len(Temp)):
        #c = next(colors)
        ax12.scatter(Volt[ignorel:ignorer:],cOCT[i][ignorel:ignorer:],c=c[i],s=40,label=str(IDlist[count]))#Temp[i])+'$^\circ$C')
    plt.legend(loc=2)
    figure = plt.gcf() # get current figure
    figure.set_size_inches(7, 7)


    fig9.savefig(savepath+'DCR_vs_BiasVoltage.pdf',format='pdf', bbox_inches='tight')
    #fig10.savefig(savepath+'DCRClean_vs_BiasVoltage.pdf',format='pdf', bbox_inches='tight')
    fig11.savefig(savepath+'OCT_vs_BiasVoltage.pdf',format='pdf', bbox_inches='tight')
    #fig12.savefig(savepath+'OCTClean_vs_BiasVoltage.pdf',format='pdf', bbox_inches='tight')



    '''
    fig13 = plt.figure(13)
    ax13 = fig13.add_axes([0.1, 0.1, 0.8, 0.8])
    ax13.yaxis.set_minor_locator(ticker.MultipleLocator(1e5))
    ax13.yaxis.set_major_formatter(formatter)
    ax13.grid(True)
    ax13.set_xlabel('Temp')
    ax13.set_ylabel(' DCR [Hz]')
    for i in range(len(Temp)):
        ax13.scatter(Temp[i],dDCR[i][0],c='r',s=40,label=' DCR vs T at '+str(OVlist[i][0]))
        ax13.scatter(Temp[i],dDCR[i][6],c='b',s=40,label=' DCR vs T at '+str(OVlist[i][6]))
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
        #ax14.scatter(np.full((1,cDCR.shape[1]),Temp[i]),cDCR[i][0],c=c[i],s=40,label='clean DCR vs T at '+str(OVlist[i]))
        ax14.scatter(Temp[i],cDCR[i][0],c='r',s=40,label='clean DCR vs T at '+str(OVlist[i][0]))
        ax14.scatter(Temp[i],cDCR[i][6],c='b',s=40,label='clean DCR vs T at '+str(OVlist[i][6]))
    #plt.legend(loc=2)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=6, mode="expand", borderaxespad=0.)


    fig13.savefig(savepath+'DCR_vs_Temp.pdf',format='pdf', bbox_inches='tight')
    fig14.savefig(savepath+'DCRClean_vs_Temp.pdf',format='pdf', bbox_inches='tight')
    '''

    fig15 = plt.figure(15)
    ax15 = fig15.add_axes([0.1, 0.1, 0.8, 0.8])
    #ax15.yaxis.set_minor_locator(ticker.MultipleLocator(1e2))
    #ax15.xaxis.set_minor_locator(ticker.MultipleLocator(5e1))
    #ax15.yaxis.set_major_formatter(formatter)
    ax15.grid(True)
    ax15.set_xlabel('Temp [$^{\circ}$C]')
    ax15.set_ylabel('BreakDownVoltage [V]')
    ax15.scatter(Temp[ignorel:ignorer:],Vbr[ignorel:ignorer:],c=c[i],s=40)
    ylims = ax15.get_ylim()
    figure = plt.gcf() # get current figure
    figure.set_size_inches(7, 7)

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
    ax16.scatter(Volt,cOCT[i],c=c[i],s=40,label='clean OCT vs Vb at '+str(Temp[i])+'$^\circ$C')
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
    ax16.set_ylabel(' OCT')
    ax16.set_ylim([0.,1.])
    for i in range(len(Temp)):
        ax16.scatter(Temp[i],dOCT[i][0],c='r',s=40,label=' OCT vs T at '+str(OVlist[i][0]))
        ax16.scatter(Temp[i],dOCT[i][6],c='b',s=40,label=' OCT vs T at '+str(OVlist[i][6]))
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
        #ax17.scatter(np.full((1,cDCR.shape[1]),Temp[i]),cDCR[i][0],c=c[i],s=40,label='clean DCR vs T at '+str(OVlist[i]))
        ax17.scatter(Temp[i],cOCT[i][0],c='r',s=40,label='clean OCT vs T at '+str(OVlist[i][0]))
        ax17.scatter(Temp[i],cOCT[i][6],c='b',s=40,label='clean OCT vs T at '+str(OVlist[i][6]))
    #plt.legend(loc=2)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=2, mode="expand", borderaxespad=0.)



    fig16.savefig(savepath+'OCT_vs_Temp.pdf',format='pdf', bbox_inches='tight')
    fig17.savefig(savepath+'OCTClean_vs_Temp.pdf',format='pdf', bbox_inches='tight')
    '''






    if count == 0:
        GainRef=AbsGain
        DCRRef=datalist[4]
        OCTRef=datalist[6]
        fig18 = plt.figure(18)
        ax18 = fig18.add_axes([0.1, 0.1, 0.8, 0.8])
        #ax18.yaxis.set_minor_locator(ticker.MultipleLocator(5e5))
        #ax18.yaxis.set_major_formatter(formatter)
        ax18.grid(True)
        ax18.set_title('150925-05 Reference at 20$^\circ$C')
        #ax18.set_ylim([-1,1])
        ax18.set_xlabel('OverVoltage [V]')
        for i in range(len(Temp)):
            ax18.plot((0,7),(1,1),'k')


            #ax18.scatter(OVlist[i][ignorel:ignorer:],[1,1,1,1,1],c=c[i],s=30,label='Gain '+str(IDlist[count]),marker=">")#Temp[i])+'$^\circ$C')
            #ax18.scatter(OVlist[i][ignorel:ignorer:],[1,1,1,1,1],c=c[i],s=30,label='DCR '+str(IDlist[count]),marker=(5,0))#Temp[i])+'$^\circ$C')
            #ax18.scatter(OVlist[i][ignorel:ignorer:],[1,1,1,1,1],c=c[i],s=30,label='OCT '+str(IDlist[count]),marker=(5,1))#Temp[i])+'$^\circ$C')

        #plt.legend(loc=2)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
        figure = plt.gcf() # get current figure
        figure.set_size_inches(7, 7)

    else: 
        if count == 1:
            Gain1=AbsGain
            DCR1=datalist[4]
            OCT1=datalist[6]

            GainRef1=np.divide(Gain1,GainRef)
            DCRRef1=np.divide(DCR1,DCRRef)
            OCTRef1=np.divide(OCT1,OCTRef)


            fig18 = plt.figure(18)
            ax18 = fig18.add_axes([0.1, 0.1, 0.8, 0.8])
            #ax18.yaxis.set_minor_locator(ticker.MultipleLocator(5e5))
            #ax18.yaxis.set_major_formatter(formatter)
            ax18.grid(True)
            ax18.set_title('150925-05 Reference at 20$^\circ$C')
            #ax18.set_ylim([-1,1])
            ax18.set_xlabel('OverVoltage [V]')
            for i in range(len(Temp)):
                ax18.scatter(OVlist[i][ignorel:ignorer:],GainRef1,c=c[i],s=50,label='Gain '+str(IDlist[count]),marker=">")#Temp[i])+'$^\circ$C')
                ax18.plot(OVlist[i][ignorel:ignorer:],GainRef1[i],c=c[i])
                ax18.scatter(OVlist[i][ignorel:ignorer:],DCRRef1,c=c[i],s=50,label='DCR '+str(IDlist[count]),marker=(5,0))#Temp[i])+'$^\circ$C')
                ax18.plot(OVlist[i][ignorel:ignorer:],DCRRef1[i],c=c[i],linestyle=':')
                ax18.scatter(OVlist[i][ignorel:ignorer:],OCTRef1,c=c[i],s=50,label='OCT '+str(IDlist[count]),marker=(5,1))#Temp[i])+'$^\circ$C')
                ax18.plot(OVlist[i][ignorel:ignorer:],OCTRef1[i],c=c[i],linestyle='--')
            #plt.legend(loc=2)
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
            ncol=2, mode="expand", borderaxespad=0.)
            figure = plt.gcf() # get current figure
            figure.set_size_inches(7, 7)











        else:
            if count == 2:

                Gain2=AbsGain
                DCR2=datalist[4]
                OCT2=datalist[6]

                GainRef2=np.divide(Gain2,GainRef)
                DCRRef2=np.divide(DCR2,DCRRef)
                OCTRef2=np.divide(OCT2,OCTRef)

                fig18 = plt.figure(18)
                ax18 = fig18.add_axes([0.1, 0.1, 0.8, 0.8])
                #ax18.yaxis.set_minor_locator(ticker.MultipleLocator(5e5))
                #ax18.yaxis.set_major_formatter(formatter)
                ax18.grid(True)
                ax18.set_title('150925-05 Reference at 20$^\circ$C',y=0.95)
                ax18.set_xlim([1,6])
                ax18.set_xlabel('OverVoltage [V]')
                for i in range(len(Temp)):
                    ax18.scatter(OVlist[i][ignorel:ignorer:],GainRef2,c=c[i],s=50,label='Gain '+str(IDlist[count]),marker=">")#Temp[i])+'$^\circ$C')
                    ax18.plot(OVlist[i][ignorel:ignorer:],GainRef2[i],c=c[i])
                    ax18.scatter(OVlist[i][ignorel:ignorer:],DCRRef2,c=c[i],s=50,label='DCR '+str(IDlist[count]),marker=(5,0))#Temp[i])+'$^\circ$C')
                    ax18.plot(OVlist[i][ignorel:ignorer:],DCRRef2[i],c=c[i],linestyle=':')
                    ax18.scatter(OVlist[i][ignorel:ignorer:],OCTRef2,c=c[i],s=50,label='OCT '+str(IDlist[count]),marker=(5,1))#Temp[i])+'$^\circ$C')
                    ax18.plot(OVlist[i][ignorel:ignorer:],OCTRef2[i],c=c[i],linestyle='--')

                #plt.legend(loc=2)
                plt.legend(bbox_to_anchor=(0., 1.02, 1., .002), loc=3,
                ncol=2, mode="expand", borderaxespad=0.)
                figure = plt.gcf() # get current figure
                figure.set_size_inches(7, 7)


                fig18.savefig(savepath+'ReferencePlot.pdf',format='pdf', bbox_inches='tight')
    




















    plt.draw()


    #tickeriter =1

Footer = ['Temp(Exp)','Vbr(Exp)','Volt(Exp)','OV(T)','Gain(T)','DCR(T)','OCT(T)']

np.save(savepath+'AreaRelGainRegrLineData',[Temp,slope,intercept])
np.save(savepath+Date+'DataPointsPlot',[Temp,Vbr,Volt,OVlist,Gain,dDCR,dOCT,Footer])

print 'Saved'
plt.show()
