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

def tick_func(xticks,Vbr):
    Val = xticks+Vbr
    return ["%.2f"% z for z in Val]
    #return ["%.3f" % z for z in V]


def calc_error(ref,comp):
    mean =(ref+comp)/2
    #error = np.zeros_like(ref)
    error = abs((ref-mean))
    return error





np.set_printoptions(precision=3)
np.set_printoptions(formatter={'float ':lambda x: format(x, '6.3E')})


savestr = 'LCT5_50um_6mm'# 1 'FixedGuess' 2 'Gainx0p25' 3 'FixedValXXmV'
Vop = 54.7

#experiment date


#date = date2 = ['2811162','2111161','1611161','2111162','2811163'] #HPK S12642 combined
date = date2 = ['0212162','0212161','0112161','2811164','0112162'] #LVR 'FixedGuess1p8mV510IntWinMPD25' deltax
#date2 = ['0510161-0','0510161-5','0510161-10','0510161-15','0510161-20','0510161-25','0510161-30','0510161-35'] #LCT5-1 xbar 'FixedGuess2mV55IntWinMPD10'
#date = ['0510162-0','0510162-5','0510162-10','0510162-15','0510162-20','0510162-25','0510162-30','0510162-35'] #LCT5-2 xbar 'FixedGuess2mV55IntWinMPD10'

#compare
#date =date2= ['0510161-25','2811164','1611161']

#folderpath of data .npy
datafolderpath = ['FixedGuess2mV55IntWinMPD10','FixedGuess1p8mV510IntWinMPD25','FixedGuess3mV520IntWinMPD25']
# data .npy name
stepstrfull = ['FixedGuess2mV55IntWinMPD10','FixedGuess1p8mV510IntWinMPD25','FixedGuess3mV520IntWinMPD25']
#regr datafolder path for ONE regr plots
datafolderpathregr = ['FixedGuess2mV55IntWinMPD10','FixedGuess1p8mV510IntWinMPD25','FixedGuess3mV520IntWinMPD25']
#name of regr file
regrnamefull = ['FixedGuess2mV55IntWinMPD10_xbar','FixedGuess1p8mV510IntWinMPD25_xbar','FixedGuess3mV520IntWinMPD25_xbar']
# name of all dcr file
alldcrfull = ['FixedGuess2mV55IntWinMPD10','FixedGuess1p8mV510IntWinMPD25','FixedGuess3mV520IntWinMPD25']
#plot name for label
namefull = ['0','5','10','15','20','25','30','35']
#namefull = ['LCT5_50um_6mm_25','LVR_50um_6mm_25','HPK_S12642_25']
#ownCloud savepath
savepathOC = '/home/gebhardt/ownCloud/HomeWork/LCT5/SNtogetheralsoVB/'



combined =True #T
combined2 = True #also Vb

'''
#Area Height
Date = '1611161'
datafolderpath = ['FixedGuess3mV520IntWinMPD25','HeightFixedGuess3mV520IntWinMPD25']
stepstrfull = ['FixedGuess3mV520IntWinMPD25','HeightFixedGuess3mV520IntWinMPD25']
regrnamefull = ['FixedGuess3mV520IntWinMPD25_Combined','HeightFixedGuess3mV520IntWinMPD25_Combined']
namefull = ['Area','Height']
datafolderpathregr = ['FixedGuess3mV520IntWinMPD25','HeightFixedGuess3mV520IntWinMPD25']
alldcrfull = ['FixedGuess3mV520IntWinMPD25','HeightFixedGuess3mV520IntWinMPD25']
'''




CHECSUSOCT = [32./100,42./100,51./100,60./100]
CHECSUSOV = [2,2.5,3,3.5]
CHECSHAMOCT = [7.5/100,28.75/100,52.5/100,77.5/100]
CHECSHAMOV = [1,2,3,4]
LVRNagOV = [1.8     ,2.8    ,3.8    ,4.8     ,5.8     ,6.8     ,7.8     ,8.8      ,9.8]
LVRNagOCT = [0.7/100,2.9/100,9.3/100,12.9/100,16.3/100,20.2/100,24.2/100,27.97/100,31.3/100]
LCT5NagOV =  [2       ,3      ,4     ,5       ,6        ,7        ,8        ,9        ,10       ,11]
LCT5NagOCT = [0.38/100,2.9/100,6./100,8.22/100,10.75/100,13.24/100,16.24/100,18.91/100,21.55/100,24.8/100]
'''   # COMPARE Analysis variable-setup
savestr = 'Compare_One_Regr_FixedGuess1p5-3mV520IntWinMPD100-63-25-13'# 1 'FixedGuess' 2 'Gainx0p25' 3 'FixedValXXmV'
#experiment date
Date = '1611161'#,'1711161','2111162']
#folderpath of data .npy
datafolderpath = ['FixedGuess3mV520IntWinMPD63','FixedGuess3mV520','FixedGuess3mV520IntWinMPD25','FixedGuess3mV520IntWinMPD13','FixedGuess1p5mV520IntWinMPD25']

# data .npy name
#stepstrfull = ['FixedGuess3mV520IntWinMPD63','FixedGuess3mV520IntWin_Combined','FixedGuess3mV520IntWinMPD25_Combined','FixedGuess3mV520IntWinMPD13_Combined',
#'FixedGuess3mV55IntWinMPD25_Combined']
stepstrfull = ['FixedGuess3mV520IntWinMPD63_Calc_from_one_regr_MPD25','FixedGuess3mV520IntWin_Calc_from_one_regr_MPD25',
'FixedGuess3mV520IntWinMPD25_Calc_from_one_regr_MPD25','FixedGuess3mV520IntWinMPD13_Calc_from_one_regr_MPD25','FixedGuess1p5mV520IntWinMPD25_Calc_from_one_regr_MPD25']

#regr datafolder path for ONE regr plots
datafolderpathregr = ['FixedGuess3mV520IntWinMPD25']
#name of regr file
regrnamefull = ['FixedGuess3mV520IntWinMPD63_Combined','FixedGuess3mV520IntWin_Combined','FixedGuess3mV520IntWinMPD25_Combined',
'FixedGuess3mV520IntWinMPD13_Combined','FixedGuess3mV55IntWinMPD25_Combined']

# name of all dcr file
alldcrfull = ['FixedGuess3mV520IntWinMPD63','FixedGuess3mV520IntWin','FixedGuess3mV520IntWinMPD25','FixedGuess3mV520IntWinMPD13','FixedGuess1p5mV520IntWinMPD25']
#plot name for label
namefull = ['3mV520IntWinMPD63','3mV520IntWinMPD100','3mV520IntWinMPD25','3mV520IntWinMPD13','1p5mV520IntWinMPD25']
'''


linestyle = ['solid','-','--','-.',':']
c = ['cyan','magenta','black','purple','blue','green','red','orange']
#c = ['purple','blue','green','red','orange']

colorticker = 0
#for a,stepstrlong in enumerate(stepstrfull): #one


x = np.arange(50,64)
for a,Date in enumerate(date):  #a = j
    a = 0 #single analysis_step-setup
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
    dDCR1 = datalistregr[3][0]
    dOCT1 = datalistregr[4][0]
    datalistcalc = np.load(savepath+str(datafolderpath[a])+'/'+str(stepstrfull[a])+'AreaNPYDataTable.npy')
    Plot_Gain = datalistcalc[6][0] #6 xbar # 7 deltax # 8 combined

    datalistregr2 = np.load(savepath2+str(datafolderpath[a])+'/'+str(stepstrfull[a])+'AreaNPYDataTableRegr.npy') #[T,Vb,TGainl,TDCRl,TOCTl]))
    Gain2 = datalistregr2[2][0]
    dDCR2 = datalistregr2[3][0]
    dOCT2 = datalistregr2[4][0]
    datalistcalc2 = np.load(savepath2+str(datafolderpath[a])+'/'+str(stepstrfull[a])+'AreaNPYDataTable.npy')
    Plot_Gain2 = datalistcalc2[6][0] #6 xbar # 7 deltax # 8 combined





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



    allDCRdatalist = np.load(savepath+str(datafolderpath[a])+'/'+str(alldcrfull[a])+'DCRfromtotalRegr.npy') #[Temp,Vbr,Volt,OVlist,Gain,dDCR,dOCT,Footer,AbsGain])
    allDCR1 = allDCRdatalist[2][0] #T


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
    ax1.xticks(np.arange(0,10,1))
    ax1.grid(True)
    #ax1.set_xlim(0,7)

    ax1.set_title('DCR '+str(savestr))
    ax1.set_xlabel('OverVoltage [V]')
    ax1.set_ylabel('DCR [Hz]')
    for i in range(len(Temp1)):
        ax1.plot(OV1,dDCR1,c=c[i+j],ls='solid')
        ax1.scatter(OV1,dDCR1,c=c[i+j],s=60,label=namefull[j]+'$^\circ$')    
        #ax1.fill_between(OV1, dDCR1-DCR_Er, dDCR1+DCR_Er,facecolor=c[i+j],edgecolor=c[i+j],label=namefull[j]+'$^\circ$')

    plt.legend(loc=2)

    if not combined2:
        ax1xticks = ax1.get_xticks()
        ax11 = fig1.add_axes(ax1.get_position())
        ax11.patch.set_visible(False)
        ax11.yaxis.set_visible(False)
        ax11.set_xticks(ax1xticks)
        for spinename, spine in ax11.spines.iteritems():
            if spinename != 'bottom':
                spine.set_visible(False)
        ax11.spines['bottom'].set_position(('outward', 30))
        ax11.set_xlim(ax1.get_xlim())
        ax11.set_xlabel('BiasVoltage [V]', labelpad=-3)

        ax11.set_xticklabels(tick_func(ax1xticks,Vbr1))

        #ax1.scatter(OV1,allDCR1,c=c[i+j],s=100,marker='+')
        #ax1.plot(OV1,allDCR1,c=c[i+j],ls='dashed')
        #ax1.text(0,5e6,'allDCR dashed, star = regr', style='italic',bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})
        #ax1.text(Volt1[-1]+0.3,allDCR1[-1],c[i+j], style='italic')
    #ax1.text(66.,4e6,'allDCR dashed , Green visible behind Black', style='italic',bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})


    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()   


    fig2= plt.figure(2)
    ax2 = fig2.add_axes([0.1, 0.1, 0.8, 0.8])
    ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax2.grid(True)
    ax2.set_ylim(0,0.9)

    ax2.set_title('OCT '+str(savestr))
    ax2.set_xlabel('OverVoltage [V]', labelpad=-3)
    ax2.set_ylabel('OCT [%]')

    
    if j ==4:
        ax2.plot(LCT5NagOV,LCT5NagOCT,color = 'lightblue',linewidth = 15,label='Nag', zorder=0)
        #ax2.plot(LVRNagOV,LVRNagOCT,color = 'magenta',linewidth = 15,label='Nag', zorder=0)
        #ax2.plot(CHECSHAMOV,CHECSHAMOCT,color = 'black',linewidth = 15,label='HPK', zorder=0)
        #ax2.plot(CHECSUSOV,CHECSUSOCT,color = 'black',linewidth = 15,label='US', zorder=0)
        

    for i in range(len(Temp1)):
        ax2.fill_between(OV1, dOCT1-OCT_Er, dOCT1+OCT_Er,facecolor=c[i+j],edgecolor=c[i+j],label=namefull[j]+'$^\circ$')
        #ax2.plot(OV1,dOCT1,c=c[i+j])
        #ax2.scatter(OV1,dOCT1,c=c[i+j],s=60,label=namefull[j]+'$^\circ$')


    plt.legend(loc=2)

    if not combined2:
        ax22 = fig2.add_axes(ax2.get_position())
        ax2xticks = ax2.get_xticks()
        ax22.patch.set_visible(False)
        ax22.yaxis.set_visible(False)
        ax22.set_xticks(ax1xticks)
        for spinename, spine in ax22.spines.iteritems():
            if spinename != 'bottom':
                spine.set_visible(False)
        ax22.spines['bottom'].set_position(('outward', 30))
        ax22.set_xlim(ax2.get_xlim())
        ax22.set_xlabel('BiasVoltage [V]', labelpad=-3)
        ax22.set_xticklabels(tick_func(ax1xticks,Vbr1))



    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()


    

    fig3= plt.figure(3)
    ax3 = fig3.add_axes([0.1, 0.1, 0.8, 0.8])
    ax3.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    ax3.grid(True)
    #ax3.set_ylim(0,0.3)
    ax3.set_title('Gain '+str(savestr))
    ax3.set_xlabel('OverVoltage [V]', labelpad=-3)
    ax3.set_ylabel('Gain [V*bins]')
    for i in range(len(Temp1)):
        #if j ==5:
            #ax3.scatter(Vop-Vbr1,Vop*slope+inter,marker='+',s=100)
            #ax3.annotate('LCT5 Op Point', xy=(Vop-Vbr1,Vop*slope+inter), xytext=(Vop-Vbr1,(Vop*slope+inter)*0.7),style='italic',bbox={'facecolor':'red', 'alpha':0.5, 'pad':10},arrowprops=dict(facecolor='black', shrink=0.05))
        ax3.plot(x-Vbr1,((x)*slope+inter),c=c[i+j],label='Gain '+namefull[j]+'$^\circ$'+' Vbr'+str('%.2f'%(-inter/slope)),linewidth=2)
        ax3.scatter(OV1,Plot_Gain,c=c[i+j],s=30)
        #ax3.fill_between(OV1, Plot_Gain-Gain_Er, Plot_Gain+Gain_Er,facecolor=c[i+j],edgecolor=c[i+j],label=namefull[j]+'$^\circ$')

        #ax3.text(x[-1],(x)*slope+inter+0.2,c[i+j])

    plt.legend(loc=2)

    if not combined2:
        ax33 = fig3.add_axes(ax3.get_position())
        ax3xticks = ax3.get_xticks()
        ax33.patch.set_visible(False)
        ax33.yaxis.set_visible(False)
        ax33.set_xticks(ax3xticks)
        for spinename, spine in ax33.spines.iteritems():
            if spinename != 'bottom':
                spine.set_visible(False)
        ax33.spines['bottom'].set_position(('outward', 30))
        ax33.set_xlim(ax3.get_xlim())
        ax33.set_xlabel('BiasVoltage [V]', labelpad=-3)
        ax33.set_xticklabels(tick_func(ax3xticks,Vbr1))
        
    #ax3.annotate('HAM Vbr 25deg', xy=(37.2, 0), xytext=(37.2, 0.04),style='italic',bbox={'facecolor':'red', 'alpha':0.5, 'pad':10},arrowprops=dict(facecolor='black', shrink=0.05))
    #ax3.text(67.6,0.,'one regression line:\nFixedGuess3mV520IntWinMPD25_Combined', style='italic',bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})

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
            ax4.plot(x,((x)*slope+inter),c=c[i+j],label='Gain '+namefull[j]+'$^\circ$'+' Vbr'+str('%.2f'%(-inter/slope)),linewidth=2)
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
            #ax5.plot(x,((x)*slope+inter),c=c[i+j],label='Gain '+namefull[j]+'$^\circ$'+' Vbr'+str('%.2f'%(-inter/slope)),linewidth=2)
            #ax5.scatter(Volt1,Plot_Gain,c=c[i+j],s=30)
            #ax5.text(x[-1],(x)*slope+inter+0.2,c[i+j])
            ax5.errorbar(Temp1, Vbr1,yerr=Vbr_Er,color=c[i+j],capsize=7,capthick=2)

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

        
if combined:
    plt.show()
    fig1.savefig(savepathOC+str(savestr)+'DCR_vs_OV_Combined.pdf',format='pdf', bbox_inches='tight')
    fig2.savefig(savepathOC+str(savestr)+'OCT_vs_OV_Combined.pdf',format='pdf', bbox_inches='tight')
    fig3.savefig(savepathOC+str(savestr)+'Gain_vs_OV_Combined.pdf',format='pdf', bbox_inches='tight')
    fig4.savefig(savepathOC+str(savestr)+'Gain_vs_Vb_Combined.pdf',format='pdf', bbox_inches='tight')
    fig5.savefig(savepathOC+str(savestr)+'Vbr_vs_T_Combined.pdf',format='pdf', bbox_inches='tight')

print 'saving '+savepathOC


    









#                                                                           Hi Ben, glad you're back ! :)
