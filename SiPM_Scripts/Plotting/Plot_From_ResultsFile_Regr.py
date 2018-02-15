from matplotlib.pyplot import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle


def GetSavePath(Date):
    savepath = '/home/gebhardt/00_SiPM_MPIK/scripts/Data/Date'+Date+'/'
    #savepath = '/home/gebhardt/ownCloud/02_Results/LCT5_50um_6mm/SN10432new/'
    return savepath

def log_10_product(x, pos):
    """The two args are the value and tick position.
    Label ticks with the product of the exponentiation"""
    return '{:.2e}'.format(x)
formatter = FuncFormatter(log_10_product)

def tick_func(set_xticks,Vbr):
    Val = set_xticks+Vbr
    return ["%.2f"% z for z in Val]
    #return ["%.3f" % z for z in V]


def calc_error(ref,comp):
    mean =(ref+comp)/2
    #error = np.zeros_like(ref)
    error = abs((ref-mean))
    return error





np.set_printoptions(precision=3)
np.set_printoptions(formatter={'float ':lambda x: format(x, '6.3E')})


savestr = 'HPK_S12642_IntWin520'# 1 'FixedGuess' 2 'Gainx0p25' 3 'FixedValXXmV'


#experiment date


date = date2 = ['1611161']#['2811162','2111161','1611161','2111162','2811163'] #HPK S12642 combined
#date = date2 = ['0212162','0212161','0112161','2811164','0112162'] #LVR 'FixedGuess1p8mV510IntWinMPD25' deltax
#date2 = ['0510161-0','0510161-5','0510161-10','0510161-15','0510161-20','0510161-25','0510161-30','0510161-35'] #LCT5-1 xbar 'FixedGuess2mV55IntWinMPD10'
#date =date2= ['0510162-25','0510162-25','0510162-25','0510162-25']#['0510162-0','0510162-5','0510162-10','0510162-15','0510162-20','0510162-25','0510162-30','0510162-35'] #LCT5-2 xbar 'FixedGuess2mV55IntWinMPD10'

#compare
#date =date2= ['0510161-25','2811164','1611161']
#folderpath of data .npy
datafolderpath = ['FixedGuess3mV520IntWinMPD25']#,'FixedGuess1p8mV510IntWinMPD25','FixedGuess3mV520IntWinMPD25']
# data .npy name
stepstrfull = ['FixedGuess3mV520IntWinMPD25']#,'FixedGuess1p8mV510IntWinMPD25','FixedGuess3mV520IntWinMPD25']
#regr datafolder path for ONE regr plots
datafolderpathregr = ['FixedGuess3mV520IntWinMPD25']#,'FixedGuess1p8mV510IntWinMPD25','FixedGuess3mV520IntWinMPD25']
#name of regr file
regrnamefull = ['FixedGuess3mV520IntWinMPD25_xbar']#,'FixedGuess1p8mV510IntWinMPD25_xbar','FixedGuess3mV520IntWinMPD25_xbar']
# name of all dcr file
alldcrfull = ['FixedGuess3mV520IntWinMPD25']#,'FixedGuess1p8mV510IntWinMPD25','FixedGuess3mV520IntWinMPD25']
#plot name for label
namefull = ['520']#['0$^\circ$','5$^\circ$','10$^\circ$','15$^\circ$','20$^\circ$','25$^\circ$','30$^\circ$']
#namefull = ['LCT5_50um_6mm_25','LVR_50um_6mm_25','HPK_S12642_25']
#ownCloud savepath
savepathOC = '/home/gebhardt/ownCloud/HomeWork/HPK_S12642/IntWinCompare/'

####
#FixedGuess4mV1010IntWinMPD10
####

combined =True #T



CHECSUSOCT = [32./100,42./100,51./100,60./100]
CHECSUSOV = [2,2.5,3,3.5]
CHECSHAMOCT = [7.5/100,28.75/100,52.5/100,77.5/100]
CHECSHAMOV = [1,2,3,4]
LVRNagOV = [1.8     ,2.8    ,3.8    ,4.8     ,5.8     ,6.8     ,7.8     ,8.8      ,9.8]
LVRNagOCT = [0.7/100,2.9/100,9.3/100,12.9/100,16.3/100,20.2/100,24.2/100,27.97/100,31.3/100]
LCT5NagOV =  [2       ,3      ,4     ,5       ,6        ,7        ,8        ,9        ,10       ,11]
LCT5NagOCT = [0.38/100,2.9/100,6./100,8.22/100,10.75/100,13.24/100,16.24/100,18.91/100,21.55/100,24.8/100]


linestyle = ['solid','-','--','-.',':']
#c = ['cyan','magenta','black','purple','blue','green','red','orange']
c = ['black','purple','blue','green','red']

colorticker = 0
#for a,stepstrlong in enumerate(stepstrfull): #one

fac=[1.,1.] #did only 10% of the checs data so *10 for the actual time
x = np.arange(65,70)
for a,Date in enumerate(date):  #a = j
    #a = 0 #single analysis_step-setup
    print Date
   
    #print stepstrlong
    j = colorticker #additional ticker for colors etc
    AbsGain1 = []
    AbsGain2 = []

    savepath = GetSavePath(date[j])
    savepath2 = GetSavePath(date2[j])




    datalistregr = np.load(savepath+str(datafolderpath[a])+'/'+str(stepstrfull[a])+'AreaNPYDataTableRegr.npy') #[T,Vb,TGainl,TDCRl,TOCTl]))
    Temp1 = datalistregr[0]
    Volt1 = datalistregr[1]
    Gain1 = datalistregr[2][0]
    dDCR1 = datalistregr[3][0]*fac[a]
    dOCT1 = datalistregr[4][0]
    datalistcalc = np.load(savepath+str(datafolderpath[a])+'/'+str(stepstrfull[a])+'AreaNPYDataTable.npy')
    print datalistcalc
#    Plot_Gain = datalistcalc[6][0] #6 xbar # 7 deltax # 8 combined
    Plot_Gain = Gain1

    datalistregr2 = np.load(savepath2+str(datafolderpath[a])+'/'+str(stepstrfull[a])+'AreaNPYDataTableRegr.npy') #[T,Vb,TGainl,TDCRl,TOCTl]))
    Gain2 = datalistregr2[2][0]
    dDCR2 = datalistregr2[3][0]*fac[a]
    dOCT2 = datalistregr2[4][0]
    datalistcalc2 = np.load(savepath2+str(datafolderpath[a])+'/'+str(stepstrfull[a])+'AreaNPYDataTable.npy')
    #Plot_Gain2 = datalistcalc2[6][0] #6 xbar # 7 deltax # 8 combined
    Plot_Gain2 = Gain2








    regrfile = np.load(savepath+str(datafolderpathregr[a])+'/'+str(regrnamefull[a])+'AreaRelGainRegrLineData.npy')
    slope = regrfile[1][0] #T
    inter = regrfile[2][0] #T

    regrfile2 = np.load(savepath2+str(datafolderpathregr[a])+'/'+str(regrnamefull[a])+'AreaRelGainRegrLineData.npy')
    slope2 = regrfile2[1][0] #T
    inter2 = regrfile2[2][0] #T


    OV1 = np.zeros_like(Volt1)
    Vbr1 = -inter/slope
    print Vbr1
    for i in range(len(Volt1)):
        OV1[i]=Volt1[i]-Vbr1


    OV2 = np.zeros_like(Volt1)
    Vbr2 = -inter2/slope2
    print Vbr2
    for i in range(len(Volt1)):
        OV2[i]=Volt1[i]-Vbr2


    print Plot_Gain
    print OV1


    allDCRdatalist = np.load(savepath+str(datafolderpath[a])+'/'+str(alldcrfull[a])+'DCRfromtotalRegr.npy') #[Temp,Vbr,Volt,OVlist,Gain,dDCR,dOCT,Footer,AbsGain])
    allDCR1 = allDCRdatalist[2][0]*fac[a]  #T


    #AbsGain
    wl_bin = 4*2.5     
    wr_bin = 5*2.5
    pulse_geom = 1.3
    conv_factor = (wl_bin+wr_bin)*pulse_geom
    for i in range(len(Gain1)):
        AbsGain1.append(Gain1[i]*1000/conv_factor) 


    #calc error
    Gain_Er = calc_error(Plot_Gain,Plot_Gain2)
    DCR_Er = calc_error(dDCR1,dDCR2)
    OCT_Er = calc_error(dOCT1,dOCT2)
    OV_Er = calc_error(OV1,OV2)    #x axis error?
    Vbr_Er = calc_error(Vbr1,Vbr2)



    #from IPython import embed;embed();1/0



    fig1 = plt.figure(1)
    ax1 = fig1.add_axes([0.1, 0.1, 0.8, 0.8])
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(1e6))
    ax1.yaxis.set_major_formatter(formatter)
    ax1.set_xticks(np.arange(0.,10.,0.5))
    ax1.grid(True)
    #ax1.set_xlim(0,7)
    ax1.set_title('DCR '+str(savestr))
    ax1.set_xlabel('OverVoltage [V]')
    ax1.set_ylabel('DCR [Hz]')
    for i in range(len(Temp1)):
    	ax1.plot(OV1,dDCR1,c=c[i+j],ls='solid')
        ax1.scatter(OV1,dDCR1,c=c[i+j],s=60,label=namefull[j])
        ax1.plot(OV1,allDCR1,c=c[i+j],ls='--')
        ax1.scatter(OV1,allDCR1,c=c[i+j],s=60,label=namefull[j],marker='+')     
        #ax1.fill_between(OV1, dDCR1-DCR_Er, dDCR1+DCR_Er,facecolor=c[i+j],edgecolor=c[i+j],label=namefull[j])
    plt.legend(loc=2)
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()   


    fig6 = plt.figure(6)
    ax6 = fig6.add_axes([0.1, 0.1, 0.8, 0.8])
    ax6.yaxis.set_minor_locator(ticker.MultipleLocator(1e6))
    ax6.yaxis.set_major_formatter(formatter)
    ax6.set_xticks(np.arange(Vbr1+0.,Vbr1+10.,0.5))
    ax6.grid(True)
    #ax6.set_xlim(0,7)
    ax6.set_title('DCR '+str(savestr))
    ax6.set_xlabel('BiasVoltage [V]')
    ax6.set_ylabel('DCR [Hz]')
    for i in range(len(Temp1)):
        ax6.plot(Volt1,dDCR1,c=c[i+j],ls='solid')
        ax6.scatter(Volt1,dDCR1,c=c[i+j],s=60,label=namefull[j])    
        ax6.plot(Volt1,allDCR1,c=c[i+j],ls='--')
        ax6.scatter(Volt1,allDCR1,c=c[i+j],s=60,label=namefull[j],marker='+')    
        
        #ax6.fill_between(Volt1, dDCR1-DCR_Er, dDCR1+DCR_Er,facecolor=c[i+j],edgecolor=c[i+j],label=namefull[j])
    plt.legend(loc=2)
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()   


    fig2= plt.figure(2)
    ax2 = fig2.add_axes([0.1, 0.1, 0.8, 0.8])
    ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax2.grid(True)
    ax2.set_ylim(0,0.9)
    ax2.set_xticks(np.arange(0.,10.,0.5))
    ax2.set_title('OCT '+str(savestr))
    ax2.set_xlabel('OverVoltage [V]', labelpad=-3)
    ax2.set_ylabel('OCT [%]') 
    if j ==0:
        #ax2.plot(LCT5NagOV,LCT5NagOCT,color = 'lightblue',linewidth = 15,label='Nag', zorder=0)
        #ax2.plot(LVRNagOV,LVRNagOCT,color = 'magenta',linewidth = 15,label='Nag', zorder=0)
        ax2.plot(CHECSHAMOV,CHECSHAMOCT,color = 'lightblue',linewidth = 15,label='HPK', zorder=0)
        ax2.plot(CHECSUSOV,CHECSUSOCT,color = 'lightblue',linewidth = 15,label='US', zorder=0)
    for i in range(len(Temp1)):
        #ax2.fill_between(OV1, dOCT1-OCT_Er, dOCT1+OCT_Er,facecolor=c[i+j],edgecolor=c[i+j],label=namefull[j])
        ax2.plot(OV1,dOCT1,c=c[i+j])
        ax2.scatter(OV1,dOCT1,c=c[i+j],s=60,label=namefull[j])


    plt.legend(loc=2)
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()


    fig7= plt.figure(7)
    ax7 = fig7.add_axes([0.1, 0.1, 0.8, 0.8])
    ax7.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax7.grid(True)
    ax7.set_ylim(0,0.9)
    ax7.set_xticks(np.arange(Vbr1+0.,Vbr1+10.,0.5))
    ax7.set_title('OCT '+str(savestr))
    ax7.set_xlabel('BiasVoltage [V]')
    ax7.set_ylabel('OCT [%]')
    plt.legend(loc=2)
    for i in range(len(Temp1)):
        #ax2.fill_between(Volt1, dOCT1-OCT_Er, dOCT1+OCT_Er,facecolor=c[i+j],edgecolor=c[i+j],label=namefull[j])
        ax7.plot(Volt1,dOCT1,c=c[i+j])
        ax7.scatter(Volt1,dOCT1,c=c[i+j],s=60,label=namefull[j])
    if j ==0:
    	ax7.plot(np.asarray(CHECSHAMOV)+64.3,CHECSHAMOCT,color = 'lightblue',linewidth = 15,label='HPK', zorder=0)
        ax7.plot(np.asarray(CHECSUSOV)+64.3,CHECSUSOCT,color = 'lightblue',linewidth = 15,label='US', zorder=0)
   
        #ax7.plot(np.asarray(LCT5NagOV)+51.7,LCT5NagOCT,color = 'lightblue',linewidth = 15,label='Nag', zorder=0)

    plt.legend(loc=2)
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()

    

    fig3= plt.figure(3)
    ax3 = fig3.add_axes([0.1, 0.1, 0.8, 0.8])
    ax3.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    ax3.grid(True)
    ax3.set_xticks(np.arange(0.,10.,0.5))
    #ax3.set_ylim(0,0.3)
    ax3.set_title('Gain '+str(savestr))
    ax3.set_xlabel('OverVoltage [V]', labelpad=-3)
    ax3.set_ylabel('Gain [V*bins]')
    for i in range(len(Temp1)):
        #if j ==5:
            #ax3.scatter(Vop-Vbr1,Vop*slope+inter,marker='+',s=100)
            #ax3.annotate('LCT5 Op Point', xy=(Vop-Vbr1,Vop*slope+inter), xytext=(Vop-Vbr1,(Vop*slope+inter)*0.7),style='italic',bbox={'facecolor':'red', 'alpha':0.5, 'pad':10},arrowprops=dict(facecolor='black', shrink=0.05))
        ax3.plot(x-Vbr1,((x)*slope+inter),c=c[i+j],label='Gain '+namefull[j]+' Vbr'+str('%.2f'%(-inter/slope)),linewidth=2)
        ax3.scatter(OV1,Plot_Gain,c=c[i+j],s=30)
        #ax3.fill_between(OV1, Plot_Gain-Gain_Er, Plot_Gain+Gain_Er,facecolor=c[i+j],edgecolor=c[i+j],label=namefull[j])
        #ax3.text(x[-1],(x)*slope+inter+0.2,c[i+j])
    plt.legend(loc=2)
    #ax3.annotate('HAM Vbr 25deg', xy=(37.2, 0), xytext=(37.2, 0.04),style='italic',bbox={'facecolor':'red', 'alpha':0.5, 'pad':10},arrowprops=dict(facecolor='black', shrink=0.05))
    #ax3.text(67.6,0.,'one regression line:\nFixedGuess3mV520IntWinMPD25_Combined', style='italic',bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()



    fig8= plt.figure(8)
    ax8 = fig8.add_axes([0.1, 0.1, 0.8, 0.8])
    ax8.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    ax8.grid(True)
    ax8.set_xticks(np.arange(Vbr1+0.,Vbr1+10.,0.5))
    #ax8.set_ylim(0,0.3)
    ax8.set_title('Gain '+str(savestr))
    ax8.set_xlabel('BiasVoltage [V]', labelpad=-3)
    ax8.set_ylabel('Gain [V*bins]')
    for i in range(len(Temp1)):
        #if j ==5:
            #ax8.scatter(Vop-Vbr1,Vop*slope+inter,marker='+',s=100)
            #ax8.annotate('LCT5 Op Point', xy=(Vop-Vbr1,Vop*slope+inter), xytext=(Vop-Vbr1,(Vop*slope+inter)*0.7),style='italic',bbox={'facecolor':'red', 'alpha':0.5, 'pad':10},arrowprops=dict(facecolor='black', shrink=0.05))
        ax8.plot(x,((x)*slope+inter),c=c[i+j],label='Gain '+namefull[j]+' Vbr'+str('%.2f'%(-inter/slope)),linewidth=2)
        ax8.scatter(Volt1,Plot_Gain,c=c[i+j],s=30)
        #ax8.fill_between(Volt1, Plot_Gain-Gain_Er, Plot_Gain+Gain_Er,facecolor=c[i+j],edgecolor=c[i+j],label=namefull[j])

        #ax8.text(x[-1],(x)*slope+inter+0.2,c[i+j])

    plt.legend(loc=2)

    #ax8.annotate('HAM Vbr 25deg', xy=(37.2, 0), xytext=(37.2, 0.04),style='italic',bbox={'facecolor':'red', 'alpha':0.5, 'pad':10},arrowprops=dict(facecolor='black', shrink=0.05))
    #ax8.text(67.6,0.,'one regression line:\nFixedGuess3mV520IntWinMPD25_Combined', style='italic',bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()


    if combined:
        fig4= plt.figure(4)
        ax4 = fig4.add_axes([0.1, 0.1, 0.8, 0.8])
        ax4.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
        ax4.grid(True)
        #ax4.set_ylim(0,0.3)
        ax4.set_title('Gain '+str(savestr))
        ax4.set_xlabel('OverVoltage [V]', labelpad=-3)
        ax4.set_ylabel('Gain [V*bins]')
        for i in range(len(Temp1)):
            #if j ==5:
                #ax4.scatter(Vop,Vop*slope+inter,marker='+',s=100)
                #ax4.annotate('LCT5 Op Point', xy=(Vop,Vop*slope+inter), xytext=(Vop,(Vop*slope+inter)*0.7),style='italic',bbox={'facecolor':'red', 'alpha':0.5, 'pad':10},arrowprops=dict(facecolor='black', shrink=0.05))
            ax4.plot(x,((x)*slope+inter),c=c[i+j],label='Gain '+namefull[j]+' Vbr'+str('%.2f'%(-inter/slope)),linewidth=2)
            ax4.scatter(Volt1,Plot_Gain,c=c[i+j],s=30)
            #ax4.text(x[-1],(x)*slope+inter+0.2,c[i+j])
            #ax4.fill_between(Volt1, Plot_Gain-Gain_Er, Plot_Gain+Gain_Er,edgecolor=c[i+j],facecolor=c[i+j])

        plt.legend(loc=2)

        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.draw()


    if combined:
        fig5= plt.figure(5)
        ax5 = fig5.add_axes([0.1, 0.1, 0.8, 0.8])
        ax5.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
        ax5.grid(True)
        ax5.set_xlim(-2,37)
        ax5.set_title('Vbr '+str(savestr))
        ax5.set_xlabel('Temp [$^\circ$C]', labelpad=-3)
        ax5.set_ylabel('BreakDown Voltage [V]')
        for i in range(len(Temp1)):
            #ax5.plot(x,((x)*slope+inter),c=c[i+j],label='Gain '+namefull[j]+' Vbr'+str('%.2f'%(-inter/slope)),linewidth=2)
            #ax5.scatter(Volt1,Plot_Gain,c=c[i+j],s=30)
            #ax5.text(x[-1],(x)*slope+inter+0.2,c[i+j])
            ax5.errorbar(Temp1, Vbr1,yerr=Vbr_Er,color=c[i+j],capsize=7,capthick=2)

        plt.legend(loc=2)

        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.draw()



    if combined:
        fig9= plt.figure(9)
        ax9 = fig9.add_axes([0.1, 0.1, 0.8, 0.8])
        ax9.yaxis.set_minor_locator(ticker.MultipleLocator(1e6))
    	ax9.yaxis.set_major_formatter(formatter)
        ax9.grid(True)
        ax9.set_xlim(0,1)
        ax9.set_title('DCR/OCT '+str(savestr))
        ax9.set_xlabel('OCT [%]', labelpad=-3)
        ax9.set_ylabel('DCR [MHz]')
        for i in range(len(Temp1)):
            ax9.plot(dOCT1,dDCR1,c=c[i+j],label='',linewidth=2)
            ax9.scatter(dOCT1,dDCR1,c=c[i+j],s=30)
         
        plt.legend(loc=2)

        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.draw()








    plt.draw()
    #plt.show()
    colorticker += 1 #=a




    np.save(savepathOC+str(savestr)+'DCR_vs_OV_'+namefull[j],[OV1,dDCR1])
    np.save(savepathOC+str(savestr)+'DCR_vs_Vb_'+namefull[j],[Volt1,dDCR1])
    np.save(savepathOC+str(savestr)+'OCT_vs_OV_'+namefull[j],[OV1,dOCT1])
    np.save(savepathOC+str(savestr)+'OCT_vs_Vb_'+namefull[j],[Volt1,dOCT1])
    np.save(savepathOC+str(savestr)+'Gain_vs_OV_'+namefull[j],[OV1,Plot_Gain])
    np.save(savepathOC+str(savestr)+'Gain_vs_Vb_'+namefull[j],[Volt1,Plot_Gain])




    if not combined:
        plt.show()  
        fig1.savefig(savepathOC+str(savestr)+'DCR_vs_OV_'+namefull[j]+'.pdf',format='pdf', bbox_inches='tight')
        fig2.savefig(savepathOC+str(savestr)+'OCT_vs_OV_'+namefull[j]+'.pdf',format='pdf', bbox_inches='tight')
        fig3.savefig(savepathOC+str(savestr)+'Gain_vs_OV_'+namefull[j]+'.pdf',format='pdf', bbox_inches='tight') 
        fig6.savefig(savepathOC+str(savestr)+'DCR_vs_Vb_'+namefull[j]+'.pdf',format='pdf', bbox_inches='tight')
        fig7.savefig(savepathOC+str(savestr)+'OCT_vs_Vb_'+namefull[j]+'.pdf',format='pdf', bbox_inches='tight')
        fig8.savefig(savepathOC+str(savestr)+'Gain_vs_Vb_'+namefull[j]+'.pdf',format='pdf', bbox_inches='tight') 
        
if combined:
    plt.show()
    fig1.savefig(savepathOC+str(savestr)+'DCR_vs_OV_Combined.pdf',format='pdf', bbox_inches='tight')
    fig2.savefig(savepathOC+str(savestr)+'OCT_vs_OV_Combined.pdf',format='pdf', bbox_inches='tight')
    fig3.savefig(savepathOC+str(savestr)+'Gain_vs_OV_Combined.pdf',format='pdf', bbox_inches='tight')
    fig4.savefig(savepathOC+str(savestr)+'Gain_vs_Vb_Combined.pdf',format='pdf', bbox_inches='tight')
    fig5.savefig(savepathOC+str(savestr)+'Vbr_vs_T_Combined.pdf',format='pdf', bbox_inches='tight')
    fig6.savefig(savepathOC+str(savestr)+'DCR_vs_Vb_Combined.pdf',format='pdf', bbox_inches='tight')
    fig7.savefig(savepathOC+str(savestr)+'OCT_vs_Vb_Combined.pdf',format='pdf', bbox_inches='tight')
    fig8.savefig(savepathOC+str(savestr)+'Gain_vs_VB_Combined.pdf',format='pdf', bbox_inches='tight') 
    fig9.savefig(savepathOC+str(savestr)+'OCT_vs_DCR_Combined.pdf',format='pdf', bbox_inches='tight')
    

print 'saving '+savepathOC


    









#                                                                           Hi Ben, glad you're back ! :)
