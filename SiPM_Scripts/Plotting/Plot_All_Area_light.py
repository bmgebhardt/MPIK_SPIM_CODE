#plots results from GaussAreaMinuitfit.py

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


#plt.style.use('ggplot')



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
    savepath = '/home/gebhardt/00_SiPM_MPIK/scripts/Data/Date'+str(Date)+'/'+str(savefolderpath)+'/'
    return savepath


np.set_printoptions(precision=3)
np.set_printoptions(formatter={'float ':lambda x: format(x, '6.3E')})
formatter = FuncFormatter(log_10_product)

date = ['1611161']#['2811162','2111161','1611161','2111162','2811163']#['0510162-0','0510162-5','0510162-10','0510162-15','0510162-20','0510162-25','0510162-30','0510162-35']#['2811164','0112161','0112162','0212161','0212162']
ignorel = 0#3
ignorer = 50#19

LCT5NagOV =  [2       ,3      ,4     ,5       ,6        ,7        ,8        ,9        ,10       ,11]
LCT5NagOCT = [0.38/100,2.9/100,6./100,8.22/100,10.75/100,13.24/100,16.24/100,18.91/100,21.55/100,24.8/100]
CHECSUSOCT = [32./100,42./100,51./100,60./100]
CHECSUSOV = [2,2.5,3,3.5]
CHECSHAMOCT = [7.5/100,28.75/100,52.5/100,77.5/100]
CHECSHAMOV = [1,2,3,4]


#path to data from GaussAreaMinuitFit.py
datastr = 'FixedGuess4mV1010IntWinMPD10'


#name of experiment save-string
savestr =    datastr#'FixedGuess4mV1010IntWinMPD10'
#slopes and intercepts from xbar,deltax,comvined
step1str =   savestr+'_xbar'
step2str =   savestr+'_deltax'
step3str =   savestr+'_Combined'



#savepath
savefolderpath  =datastr#'FixedGuess4mV1010IntWinMPD10'

x = np.arange(64,70)



c = ['purple','blue','green','red','orange']
#c = ['cyan','magenta','black','purple','blue','green','red','orange']

tickeriter = 0 #different colors in case of multiple experiments on the same canvas
for tickeriter,Date in enumerate(date):
    print tickeriter
    savepath = GetSavePath(Date)
    datalist = np.load(savepath+str(datastr)+'AreaNPYDataTable.npy')
    datalist2 = np.load(savepath+str(datastr)+'AreaNPYDataTable.npy')




    #c = ['r','b','g','c','m','y','k','w']
    #c = ['blue','indigo','purple','blueviolet','mediumorchid','firebrick','red','orangered','gold']

    print datalist

    Temp = datalist[0] #List of Temps
    Volt = datalist[1] #List of Voltages
    Gain= datalist[2] #NP array of lists of Gains per Voltage, per Temperature
    Error = datalist[3]#List of Errors of the GainCalc
    dDCR = datalist[4] #List of  DarkCountRate
    dOCT = datalist[5] #List of  OpticalCrossTalk
    xbar = datalist[6]#List of c1pe peak pos
    deltax = datalist[7]

    #continuous_data = ma.flatnotmasked_contiguous(Gain)[0]
    #cd_left,cd_right = continuous_data.indices()
    #from IPython import embed;embed();1/0

    #right = ma.flatnotmasked_contiguous(Gain)

    regrleftlist= [8]#[4 ,4 ,8 ,6 ,2 ,2 ,2 ,3 ] 
    regrrightlist=[24]#[19,23,22,22,14,13,14,13]
    #xbar
    regrleft=regrleftlist[tickeriter]
    regrright=regrrightlist[tickeriter]
    #deltax
    regrleft2=regrleftlist[tickeriter]
    regrright2=regrrightlist[tickeriter]
    #combined
    regrleft3=regrleftlist[tickeriter]
    regrright3=regrrightlist[tickeriter]


    ignorer = 5000
    delta_x_border = 20




    '''
    for j in range(Gain.shape[0]):
        for i in range(Gain.shape[1]):
            if i==20:
                continue
            else:
                if 2 * (Gain[j][i]) < Gain[j][i+1] :
                    Gain[j][i] = None
    '''
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
    '''
    #T = [26.]
    

    #Gain=ma.masked_where(unmaskedGain<0,unmaskedGain)
    #print 'Masked Gain ',Gain
    n = 7

    Vbr = []
    Vbr = []
    slope = []
    slope2 = []
    slope3 = []
    intercept = []
    intercept2 = []
    intercept3= []
    OVlist = []
    AbsGain = []
    err = []

    slopeRG = [] #RealGain
    interRG = []

    #"Gain" vs Vb regression line weighted least squares method

    for i in range(len(Temp)):  
        print i
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
        Y = xbar[i][regrleft:regrright:]#[2:]#4:1]
        #print Gain[i]
        #y_err = Error[i][regrleft:regrright:]#[2:]#4:1]
        #weights1 = pd.Series(y_err)
        #print type(weights1)
        wls_model1 = sm.RLM(Y,X,M=sm.robust.norms.AndrewWave())
        results1 = wls_model1.fit()
        print 'results1 ',results1.params
        #print results1.params.shape
        inter1,slo1 = results1.params
        slope.append(slo1)
        intercept.append(inter1)
        vbreak1 = -intercept[i]/slope[i]


        ''' original
        #regression line for Gain     
        X = Volt[regrleft:regrright:]#4:1]
        #print X.dtype
        X = sm.add_constant(X)
        Y = xbar[i][regrleft:regrright:]#[2:]#4:1]
        #print Gain[i]
        y_err = Error[i][regrleft:regrright:]#[2:]#4:1]
        weights1 = pd.Series(y_err)
        #print type(weights1)
        wls_model1 = sm.WLS(Y,X, weights=1 / weights1)
        results1 = wls_model1.fit()
        print 'results1 ',results1.params
        #print results1.params.shape
        inter1,slo1 = results1.params
        slope.append(slo1)
        intercept.append(inter1)
        vbreak1 = -intercept[i]/slope[i]
        '''




        #vbreak1 = 64.2
        #if (vbreak<60) or (vbreak>70):vbreak = 60
        Vbr.append(vbreak1)
        #err.append()
        '''
        if i == 0:
            Gain[0,G_ign]=Gainsave
        '''

        a = Volt*slo1+inter1
        #print a


    for j in range(len(Temp)):  

        #regression line for Gain1pe
        X2 = Volt[regrleft2:regrright2:]#4:1]
        X2 = sm.add_constant(X2)
        Y2 = deltax[j][regrleft2:regrright2:]#[2:]#4:1]
        wls_model2 = sm.RLM(Y2,X2, M=sm.robust.norms.AndrewWave())
        results2 = wls_model2.fit(cov='H2')
        print 'results2 ',results2.params
        inter2,slo2 = results2.params
        slope2.append(slo2)
        intercept2.append(inter2)




        '''original
        X2 = Volt[regrleft2:regrright2:]#4:1]
        X2 = sm.add_constant(X2)
        Y2 = deltax[j][regrleft2:regrright2:]#[2:]#4:1]
        y_err2 = Error[j][regrleft2:regrright2:]#[2:]#4:1]
        weights2 = pd.Series(y_err2)
        wls_model2 = sm.WLS(Y2,X2, weights=1 / weights2)
        results2 = wls_model2.fit()
        print 'results2 ',results2.params
        inter2,slo2 = results2.params
        slope2.append(slo2)
        intercept2.append(inter2)
        '''


    combined = np.empty_like(xbar)
    for b in range(len(Temp)):
        for a in range(len(xbar[b])):
            if (slope[b]*Volt[a]+intercept[b])>(slope2[b]*Volt[a]+intercept2[b]):
                combined[b][a]=xbar[b][a]
            else:
                combined[b][a]=deltax[b][a]

    print xbar
    print deltax
    print combined




    for k in range(len(Temp)):  

        #regression line for combined
        
        X3 = Volt[regrleft3:regrright3:]#4:1]
        X3 = sm.add_constant(X3)
        Y3 = combined[k][regrleft3:regrright3:]#[2:]#4:1]
        y_err3 = Error[k][regrleft3:regrright3:]#[2:]#4:1]
        weights3 = pd.Series(y_err3)
        wls_model3 = sm.WLS(Y3,X3, weights=1 / weights3)
        results3 = wls_model3.fit()
        print 'results3 ',results3.params
        inter3,slo3 = results3.params
        slope3.append(slo3)
        intercept3.append(inter3)


    print 'Vbr xbar', -inter1/slo1
    print 'Vbr deltax', -inter2/slo2

    print 'Vbr combined', -inter3/slo3



    #print 'slope ',slope,slope2
    #print 'intercept ',intercept,intercept2
    np.save(savepath+str(step1str)+'AreaRelGainRegrLineData',[Temp,slope,intercept])
    np.save(savepath+str(step2str)+'AreaRelGainRegrLineData',[Temp,slope2,intercept2])
    np.save(savepath+str(step3str)+'AreaRelGainRegrLineData',[Temp,slope3,intercept3])



    for i in range(len(Temp)):  
        OVlist.append(np.array(Volt) - np.array(Vbr[i]))
    Vbr = np.asarray(Vbr)
    print 'Vbr ',Vbr
    print 'OVlist ',OVlist


    '''
    #conv_factor = float(2.75e+8)
    q = (1.602e-19)
    r = 50.
    t = 10.e-9
    f = t/(r*q)
    '''

    wl_bin = 4*2.5
    wr_bin = 5*2.5
    pulse_geom = 1.3
    conv_factor = (wl_bin+wr_bin)*pulse_geom
    for i in range(len(Gain)):
        AbsGain.append(Gain[i]*1000/conv_factor) 



    delta_x_Gain = deltax[i][delta_x_border:]
    delta_x_DCR = dDCR[i][delta_x_border:]
    delta_x_OCT = dOCT[i][delta_x_border:]
    delta_x_AbsGain = AbsGain[i][delta_x_border:]
    delta_x_OVlist = OVlist[i][delta_x_border:]
    delta_x_Volt = Volt[delta_x_border:]

    #print delta_x_OVlist
    #print delta_x_DCR



    '''
    #RealGain Vb regression line, to get conversion factor by comparison to SensL Datasheet
    for i in range(len(Temp)):
        slo, inter, r_value, p_value, std_err = lreg(Volt,AbsGain[i])
        slopeRG.append(slo)
        interRG.append(inter)
    '''



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
        ax1.scatter(OVlist[i][ignorel:ignorer:],dDCR[i][ignorel:ignorer:],c=c[tickeriter],s=30,label=' DCR '+str(Temp[i])+'$^\circ$ ')
        ax1.plot(OVlist[i][ignorel:ignorer:],dDCR[i][ignorel:ignorer:],c=c[tickeriter])
        ax1.scatter(delta_x_OVlist[ignorel:ignorer:],delta_x_DCR[ignorel:ignorer:],s=50,c=c[tickeriter],marker='+')
        ax1.plot(delta_x_OVlist[ignorel:ignorer:],delta_x_DCR[ignorel:ignorer:],c=c[tickeriter])
        if tickeriter == 0:
            ax1.text(0.75, 0.15, 'o xbar\n+ deltax', transform=ax1.transAxes,
      fontsize=12,style='italic')
  
    ylims = ax1.get_ylim()
    plt.legend(loc=2)
    
    '''
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
        ax2.scatter(OVlist[i],cDCR[i],c=c[tickeriter],s=40,label='clean DCR vs OV at '+str(Temp[i])+'$^\circ$')
    plt.legend(loc=2)
    '''

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
    for i in range(len(Temp)):
        #c = next(colors)
        ax3.scatter(OVlist[i][ignorel:ignorer:],dOCT[i][ignorel:ignorer:],c=c[tickeriter],s=30,label=' OCT '+str(Temp[i])+'$^\circ$ ')
        ax3.plot(OVlist[i][ignorel:ignorer:],dOCT[i][ignorel:ignorer:],c=c[tickeriter])
        ax3.scatter(delta_x_OVlist[ignorel:ignorer:],delta_x_OCT[ignorel:ignorer:],s=50,c=c[tickeriter],marker='+')
        ax3.plot(delta_x_OVlist[ignorel:ignorer:],delta_x_OCT[ignorel:ignorer:],c=c[tickeriter])
        ax3.plot(CHECSHAMOV,CHECSHAMOCT,color = 'lightgrey',linewidth = 15,label='Nag', zorder=0)
        ax3.plot(CHECSUSOV,CHECSUSOCT,color = 'lightblue',linewidth = 15,label='Nag', zorder=0)

        if tickeriter == 0:
            ax3.text(0.75, 0.75, 'o xbar\n+ deltax', transform=ax3.transAxes,
      fontsize=12,style='italic')

    ylims = ax3.get_ylim()
    plt.legend(loc=2)

    '''
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
        ax4.scatter(OVlist[i],cOCT[i],c=c[tickeriter],s=30,label='clean OCT vs OV at '+str(Temp[i])+'$^\circ$')
    plt.legend(loc=2)
    '''
    fig1.savefig(savepath+str(savestr)+'DCR_vs_OV.pdf',format='pdf', bbox_inches='tight')
    #fig2.savefig(savepath+'DCRClean_vs_OV.pdf',format='pdf', bbox_inches='tight')
    fig3.savefig(savepath+str(savestr)+'OCT_vs_OV.pdf',format='pdf', bbox_inches='tight')
    #fig4.savefig(savepath+'OCTClean_vs_OV.pdf',format='pdf', bbox_inches='tight')

    

    fig4 = plt.figure(4)
    ax4 = fig4.add_axes([0.1, 0.1, 0.8, 0.8])
    thresh = np.zeros_like(Volt) +2
    ax4.plot(Volt,thresh,c=c[tickeriter])
    #ax4.scatter(Volt,Gain[i]/36.5,c=c[tickeriter])
    ax4.scatter(Volt[ignorel:ignorer:],AbsGain[i][ignorel:ignorer:],c=c[tickeriter],label='Gain '+str(Temp[i])+'$^\circ$ ')

    '''
    wl_bin = 4*2.5
    wr_bin = 5*2.5
    pulse_geom = 1.3
    conv_factor = (wl_bin+wr_bin)*pulse_geom
    '''

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
    null = np.zeros(len(Vbr))
    #colors=iter(cm.rainbow(np.linspace(0,1,n)))
    for i in range(len(Temp)):
        #c = next(colors)

        
        #ax5.scatter(Vbr,null,c='blue',s=30)
        ax5.scatter(64.3,0,c='red',s=30)

        ax5.scatter(Volt[ignorel:ignorer:],xbar[i][ignorel:ignorer:],s=30,color=c[tickeriter])
        
        ax5.scatter(Volt[ignorel:ignorer:],deltax[i][ignorel:ignorer:],c=c[tickeriter],marker='+',s=50,edgecolors='k')
        
        ax5.plot(x,slope[i]*x+intercept[i],c=c[tickeriter],label=Temp)
        #ax5.scatter(delta_x_Volt[ignorel:ignorer:],delta_x_Gain[ignorel:ignorer:],s=40,c=c[i+1],label=step1str)


        
        ax5.plot(x,slope2[i]*x+intercept2[i],c[tickeriter],linestyle='--')
        if tickeriter == 0:
            ax5.text(0.75, 0.15, 'o xbar\n+ deltax', transform=ax5.transAxes,
      fontsize=12,style='italic')

        
        #ax5.plot(x,slope3[i]*x+intercept3[i],'k--')
        #ax5.scatter(Volt[ignorel:ignorer:],combined[i][ignorel:ignorer:],s=40,label='combined',color='k',marker='+')
        


        #ax5.scatter(Volt[ignorel:ignorer:],deltax[i][ignorel:ignorer:],s=30,label=step2str,facecolors='none', edgecolors='g')
        #ax5.scatter(Volt[ignorel:ignorer:],xbar[i][ignorel:ignorer:],c='g',s=30,label=step2str)

       
        #ax5.errorbar(Volt[ignorel:ignorer:],Gain[i][ignorel:ignorer:],c=c[tickeriter],yerr=Error[i][ignorel:ignorer:],fmt='none')
    plt.legend(loc=2)
    #ax5.text(0.,0.075,str(Vbr)+' V')
    ylims = ax5.get_ylim()
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()


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
        ax5.scatter(Volt,Gain[i],c=c[tickeriter],s=30,label=str(Temp[i])+'$^\circ$')
        ax5.plot(x,slope[i]*x+intercept[i],c=c[tickeriter])
        ax5.errorbar(Volt,Gain[i],c=c[tickeriter],yerr=Error[i],fmt='none')
        plt.legend(loc=2)
    #ax5.text(0.,0.075,str(Vbr)+' V')
    #ylims = ax5.get_ylim()
    '''
    

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
        ax6.scatter(OVlist[i][ignorel:ignorer:],Gain[i][ignorel:ignorer:],c=c[tickeriter],s=30,label='Gain '+str(Temp[i])+'$^\circ$ ')
        ax6.plot(OVlist[i][ignorel:ignorer:],Gain[i][ignorel:ignorer:],c=c[tickeriter])
        ax6.scatter(delta_x_OVlist[ignorel:ignorer:],delta_x_Gain[ignorel:ignorer:],s=50,c=c[tickeriter],marker='+')
        ax6.plot(delta_x_OVlist[ignorel:ignorer:],delta_x_Gain[ignorel:ignorer:],c=c[tickeriter],linestyle='--')
        if tickeriter == 0:
            ax6.text(0.75, 0.15, 'o xbar\n+ deltax', transform=ax6.transAxes,
      fontsize=12,style='italic')

        #ax6.errorbar(OVlist[i][ignorel:ignorer:],Gain[i][ignorel:ignorer:],c=c[tickeriter],yerr=Error[i],fmt='none')
        #ax6.fill_between(OVlist[i], Gain[i]-Error[i], Gain[i]+Error[i],facecolor=c)
    plt.legend(loc=2)
    '''
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
        ax7.scatter(Volt[ignorel:ignorer:],Gain[i][ignorel:ignorer:],c=c[tickeriter],s=30,label=str(Temp[i])+'$^\circ$ ')
        ax7.plot(Volt[ignorel:ignorer:],Gain[i][ignorel:ignorer:],c=c[tickeriter])
        #ax7.plot(x,slope[i]*x+intercept[i],c=c[tickeriter])
        #ax7.errorbar(Volt,Gain[i],c=c[tickeriter],yerr=Error[i],fmt='none')
    plt.legend(loc=2)
    #ax7.text(0.,0.075,str(Vbr)+' V')
    ylims = ax7.get_ylim()
    '''

    '''
    fig18 = plt.figure(18)
    ax18 = fig18.add_axes([0.1, 0.1, 0.8, 0.8])
    #ax18.yaxis.set_minor_locator(ticker.MultipleLocator(1e6))
    #ax18.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax18.grid(True)
    ax18.set_title('"Gain" vs Vb')
    #ax18.set_ylim([0.0,0.04])
    #ax18.set_xlim(63,71)
    ax18.set_xlabel('Bias Voltage [V]')
    ax18.set_ylabel('"Gain" [V*bins]')
    null = np.zeros(len(Vbr))
    #colors=iter(cm.rainbow(np.linspace(0,1,n)))
    for i in range(len(Temp)):
        #c = next(colors)
        ax18.scatter(Vbr,null,c='blue',s=30)
        ax18.text(Vbr,null,'Vbr Regr')

        ax18.scatter(64.3,0,c='red',s=30)
        ax18.text(64.3,0,'Vbr')


        ax18.scatter(Volt[ignorel:ignorer:],deltax[i][ignorel:ignorer:],s=30,label=step2str,facecolors='none', edgecolors='red')
        ax18.scatter(Volt[ignorel:ignorer:],xbar[i][ignorel:ignorer:],c=c[tickeriter],s=30,label=step1str)
        #ax18.plot(x,slope[i]*x+intercept[i],c=c[tickeriter])
        ax18.scatter(delta_x_Volt[ignorel:ignorer:],delta_x_Gain[ignorel:ignorer:],s=40,c=c[i+1],label=step1str)

        ax18.plot(x,slope2[i]*x+intercept2[i],'r--')


        #ax18.scatter(Volt[ignorel:ignorer:],Gain1pePREV[i][ignorel:ignorer:],s=30,label=step2str,c='g',marker='+')
        #ax18.scatter(Volt[ignorel:ignorer:],GainPREV[i][ignorel:ignorer:],c='g',s=30,label=step2str,marker='+')

       
        #ax18.errorbar(Volt[ignorel:ignorer:],Gain[i][ignorel:ignorer:],c=c[tickeriter],yerr=Error[i][ignorel:ignorer:],fmt='none')
    plt.legend(loc=2)
    
    '''
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
        ax7.scatter(Volt,AbsGain[i],c=c[tickeriter],label='Gain at '+str(Temp[i])+'$^\circ$')
    ylims = ax7.get_ylim()
    plt.legend(loc=2)
    '''
    fig7 = plt.figure(7)
    ax7 = fig7.add_axes([0.1, 0.1, 0.8, 0.8])
    #ax8.yaxis.set_major_formatter(formatter(0.002))
    #ax8.set_yticks(np.arange(-0.002, 0.006, 0.002))
    #ax8.yaxis.set_minor_locator(ticker.MultipleLocator(2.5e-1))
    ax7.grid(True)
    ax7.set_ylim(1,8)
    #ax8.yaxis.set_major_formatter(formatter)
    ax7.set_xlabel('OV [V]')
    ax7.set_ylabel('Gain [mV/p.e.] {RelGain/Conv_Factor}')
    #colors=iter(cm.rainbow(np.linspace(0,1,n)))
    for i in range(len(Temp)):
        #c = next(colors)
        ax7.scatter(OVlist[i][ignorel:ignorer:],AbsGain[i][ignorel:ignorer:],s=30,c=c[tickeriter],label='Gain '+str(Temp[i])+'$^\circ$ ')
        ax7.plot(OVlist[i][ignorel:ignorer:],AbsGain[i][ignorel:ignorer:],c=c[tickeriter])
        ax7.scatter(delta_x_OVlist[ignorel:ignorer:],delta_x_AbsGain[ignorel:ignorer:],s=50,c=c[tickeriter],marker='+')
        ax7.plot(delta_x_OVlist[ignorel:ignorer:],delta_x_AbsGain[ignorel:ignorer:],c=c[tickeriter],ls='--')
        if tickeriter == 0:
            ax7.text(0.75, 0.15, 'o xbar\n+ deltax', transform=ax7.transAxes,
      fontsize=12,style='italic')

        #ax8.scatter(Volt[ignorel:ignorer:],AbsGain[i][ignorel:ignorer:],c=c[tickeriter],label='Gain at '+str(Temp[i])+'$^\circ$ ')

    plt.legend(loc=2)

    fig8 = plt.figure(8)
    ax8 = fig8.add_axes([0.1, 0.1, 0.8, 0.8])
    #ax8.yaxis.set_major_formatter(formatter(0.002))
    #ax8.set_yticks(np.arange(-0.002, 0.006, 0.002))
    #ax8.yaxis.set_minor_locator(ticker.MultipleLocator(2.5e-1))
    ax8.grid(True)
    ax8.set_ylim(1,8)
    #ax8.yaxis.set_major_formatter(formatter)
    ax8.set_xlabel('Vb [V]')
    ax8.set_ylabel('Gain [mV/p.e.] {RelGain/Conv_Factor}')
    #colors=iter(cm.rainbow(np.linspace(0,1,n)))
    for i in range(len(Temp)):
        #c = next(colors)
        ax8.scatter(Volt[ignorel:ignorer:],AbsGain[i][ignorel:ignorer:],c=c[tickeriter],s=30,label='Gain '+str(Temp[i])+'$^\circ$ ')
        ax8.plot(Volt[ignorel:ignorer:],AbsGain[i][ignorel:ignorer:],c=c[tickeriter])
        ax8.scatter(delta_x_Volt[ignorel:ignorer:],delta_x_AbsGain[ignorel:ignorer:],s=50,c=c[tickeriter],marker='+')
        ax8.plot(delta_x_Volt[ignorel:ignorer:],delta_x_AbsGain[ignorel:ignorer:],c=c[tickeriter],ls='--')
        if tickeriter == 0:
            ax8.text(0.75, 0.15, 'o xbar\n+ deltax', transform=ax8.transAxes,
      fontsize=12,style='italic')


        #ax8.scatter(Volt[ignorel:ignorer:],AbsGain[i][ignorel:ignorer:],c=c[tickeriter],label='Gain at '+str(Temp[i])+'$^\circ$ ')

    plt.legend(loc=2)
    
    fig5.savefig(savepath+str(savestr)+'RelGain_vs_Vb.pdf',format='pdf', bbox_inches='tight')
    fig6.savefig(savepath+str(savestr)+'RelGain_vs_OV.pdf',format='pdf', bbox_inches='tight')
    #fig7.savefig(savepath+str(step1str)+'AbsGain_vs_OV.pdf',format='pdf', bbox_inches='tight')
    #fig8.savefig(savepath+str(step1str)+'AbsGain_vs_Vb.pdf',format='pdf', bbox_inches='tight')
    #fig18.savefig(savepath+str(step1str)+'RelGain_vs_Vb_Compare.pdf',format='pdf', bbox_inches='tight')




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
        ax9.scatter(Volt[ignorel:ignorer:],dDCR[i][ignorel:ignorer:],c=c[tickeriter],s=30,label=' DCR '+str(Temp[i])+'$^\circ$ ')
        ax9.plot(Volt[ignorel:ignorer:],dDCR[i][ignorel:ignorer:],c=c[tickeriter])
        ax9.scatter(delta_x_Volt[ignorel:ignorer:],delta_x_DCR[ignorel:ignorer:],s=50,c=c[tickeriter],marker='+')
        ax9.plot(delta_x_Volt[ignorel:ignorer:],delta_x_DCR[ignorel:ignorer:],c=c[tickeriter],ls='--')
        if tickeriter == 0:
            ax9.text(0.75, 0.15, 'o xbar\n+ deltax', transform=ax9.transAxes,
      fontsize=12,style='italic')
 
    ylims = ax9.get_ylim()
    plt.legend(loc=2)

    '''
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
        ax10.scatter(Volt,cDCR[i],c=c[tickeriter],s=40,label='clean DCR vs Vb at '+str(Temp[i])+'$^\circ$')
    plt.legend(loc=2)
    '''


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
        ax11.scatter(Volt[ignorel:ignorer:],dOCT[i][ignorel:ignorer:],c=c[tickeriter],s=30,label=' OCT '+str(Temp[i])+'$^\circ$ ')
        ax11.plot(Volt[ignorel:ignorer:],dOCT[i][ignorel:ignorer:],c=c[tickeriter])
        ax11.scatter(delta_x_Volt[ignorel:ignorer:],delta_x_OCT[ignorel:ignorer:],s=50,c=c[tickeriter],marker='+')
        ax11.plot(delta_x_Volt[ignorel:ignorer:],delta_x_OCT[ignorel:ignorer:],c=c[tickeriter],ls='--')
        if tickeriter == 0:
            ax11.text(0.75, 0.75, 'o xbar\n+ deltax', transform=ax11.transAxes,
      fontsize=12,style='italic')
 
    ylims = ax11.get_ylim()
    plt.legend(loc=2)

    '''
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
        ax12.scatter(Volt,cOCT[i],c=c[tickeriter],s=40,label='clean OCT vs Vb at '+str(Temp[i])+'$^\circ$')
    plt.legend(loc=2)
    '''


    fig9.savefig(savepath+str(savestr)+'DCR_vs_BiasVoltage.pdf',format='pdf', bbox_inches='tight')
    #fig10.savefig(savepath+'DCRClean_vs_BiasVoltage.pdf',format='pdf', bbox_inches='tight')
    fig11.savefig(savepath+str(savestr)+'OCT_vs_BiasVoltage.pdf',format='pdf', bbox_inches='tight')
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
        #ax14.scatter(np.full((1,cDCR.shape[1]),Temp[i]),cDCR[i][0],c=c[tickeriter],s=40,label='clean DCR vs T at '+str(OVlist[i]))
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
    ax15.scatter(Temp,Vbr,c=c[tickeriter],s=40)
    ax15.scatter(Temp,Vbr,c=c[tickeriter],s=40)
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
    ax16.scatter(Volt,cOCT[i],c=c[tickeriter],s=40,label='clean OCT vs Vb at '+str(Temp[i])+'$^\circ$')
    plt.legend(loc=2)
    '''
    fig15.savefig(savepath+str(savestr)+'BreakDownVoltage_vs_Temp.pdf',format='pdf', bbox_inches='tight')


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
        #ax17.scatter(np.full((1,cDCR.shape[1]),Temp[i]),cDCR[i][0],c=c[tickeriter],s=40,label='clean DCR vs T at '+str(OVlist[i]))
        ax17.scatter(Temp[i],cOCT[i][0],c='r',s=40,label='clean OCT vs T at '+str(OVlist[i][0]))
        ax17.scatter(Temp[i],cOCT[i][6],c='b',s=40,label='clean OCT vs T at '+str(OVlist[i][6]))
    #plt.legend(loc=2)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=2, mode="expand", borderaxespad=0.)



    fig16.savefig(savepath+'OCT_vs_Temp.pdf',format='pdf', bbox_inches='tight')
    fig17.savefig(savepath+'OCTClean_vs_Temp.pdf',format='pdf', bbox_inches='tight')
    '''
    plt.draw()
    #plt.show()

    OVlist = np.asarray(OVlist)

Footer = ['Temp(Exp)','Vbr(Exp)','Volt(Exp)','OV(T)','Gain(T)','DCR(T)','OCT(T)','footer','AbsGain','deltax','xbar','Combined_Gain']
np.save(savepath+str(savestr)+Date+'DataPointsPlot',[Temp,Vbr,Volt,OVlist,Gain,dDCR,dOCT,Footer,AbsGain,deltax,xbar,combined])
print 'Saved ',savepath
plt.show()
