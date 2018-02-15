#Calculation from Regression Line Data derived Gain

import numpy as np
import os.path
import matplotlib.pyplot as plt
import argparse

datastr = 'FixedGuess3mV520IntWinMPD25'
regr1str =   'FixedGuess3mV520IntWinMPD25_xbar'# 1 'FixedGuess' 2 'Gainx0p25' 3 'FixedValXXmV' 1.1'FixedGuess3mV520IntWin' 1.2 'FixedGuess3mV520IntWinMPD63'
regr2str =   'FixedGuess3mV520IntWinMPD25_Combined'


step1str = 'FixedGuess3mV520IntWinMPD25_xbar'# 1 'FixedGuess' 2 'Gainx0p25' 3 'FixedValXXmV'
step2str = 'FixedGuess3mV520IntWinMPD25_Combined'# 1 'FixedGuess' 2 'Gainx0p25' 3 'FixedValXXmV'
#step2str = 'Gainx0p25'# 1 'FixedGuess' 2 'Gainx0p25' 3 'FixedValXXmV'
#step3str = 'FixedVal2mV'# 1 'FixedGuess' 2 'Gainx0p25' 3 'FixedValXXmV'


savestr = 'Compare_Regr_Combined_Xbar_FixedGuess3mV520IntWinMPD25'# 1 'FixedGuess' 2 'Gainx0p25' 3 'FixedValXXmV'
savefolderpath = 'FixedGuess3mV520IntWinMPD25'# 1 'FixedGuess' 2 'Gainx0p25' 3 'FixedVal'



def GetNames(destT,destV,Date):
    #generate sub-experiment name
    path = '/home/gebhardt/00_SiPM_MPIK/scripts/Data/Date'+str(Date)+'/'+str(destT)+'deg/'+str(destV)+'V/'
    name = 'HAM_T' + str(destT) + '_Vb' + str(destV) + '.trc'
    compl_name = os.path.join(path,name)
    return compl_name


def LoadRunCard(Date):
    a = np.load('/home/gebhardt/00_SiPM_MPIK/scripts/Data/Date'+str(Date)+'/RunCard'+str(Date)+'.npy')
    print 'RunCard loaded'
    print a[0]
    print a[1]
    print a[2]
    return a

def LoadTime(path):

    samplerate = 2.5e9
    length = LecroyBinary.LecroyBinaryWaveform(path).WaveArray()
    time = length/samplerate
    recordtime = LecroyBinary.LecroyBinaryWaveform(path).ReturnRecordTime() # Time Bin
    #print recordtime
    return time

def Plot_All(Date,ignorel,ignorer):
    for i,date in enumerate(Date):
        savepath = '/home/gebhardt/00_SiPM_MPIK/scripts/Data/Date'+str(date)+'/'+str(savefolderpath)+'/'
        #no_regr_1_dataarray = np.load(savepath+str(datastr)+str(date)+'DataPointsPlot.npy')#[Temp,Vbr,Volt,OVlist,Gain,dDCR,dOCT,Footer,AbsGain])
        #no_regr_2_dataarray = np.load(savepath+str(datastr)+str(date)+'DataPointsPlot.npy')#[Temp,Vbr,Volt,OVlist,Gain,dDCR,dOCT,Footer,AbsGain])


        regr_1_dataarray = np.load(savepath+str(step1str)+'AreaNPYDataTableRegr.npy')#[T,Vb,TGainl,TDCRl,TOCTl])
        #regr_2_dataarray = np.load(savepath+str(step2str)+'AreaNPYDataTableRegr.npy')#[T,Vb,TGainl,TDCRl,TOCTl])



        regrfileDeltaX = np.load(savepath+str(regr1str)+'AreaRelGainRegrLineData.npy') #DX
        #regrfileXbar = np.load(savepath+str(regr2str)+'AreaRelGainRegrLineData.npy') #Xbar

        slope1 = regrfileDeltaX[1][0] #T
        inter1 = regrfileDeltaX[2][0] #T
        #slope2 = regrfileXbar[1][0] #T
        #inter2 = regrfileXbar[2][0] #T

        
        T = regr_1_dataarray[0]
        Vb = regr_1_dataarray[1]
        '''
        G0 = regr_0_dataarray[4][0]
        DCR0 = regr_0_dataarray[5]
        OCT0 = regr_0_dataarray[6]
        '''
        
        G1R = regr_1_dataarray[2]
        DCR1R = regr_1_dataarray[3]
        OCT1R = regr_1_dataarray[4]


        #G2R = regr_2_dataarray[2]
        #DCR2R= regr_2_dataarray[3]
        #OCT2R = regr_2_dataarray[4]

        G1 = no_regr_1_dataarray[4]
        DCR1= no_regr_1_dataarray[5]
        OCT1= no_regr_1_dataarray[6]

        
        G2= no_regr_2_dataarray[4]
        DCR2= no_regr_2_dataarray[5]
        OCT2= no_regr_2_dataarray[6]
        
        '''
        G3 = regr_3_dataarray[2]
        DCR3= regr_3_dataarray[3]
        OCT3 = regr_3_dataarray[4]
        '''
        print G1R






        c = ['b','lightblue','green','lightgreen']

        fig1 = plt.figure(1)
        ax1 = fig1.add_axes([0.1, 0.1, 0.8, 0.8])
        ax1.grid(True)
        ax1.set_ylim(-0.05,0.3)
        ax1.set_title('"Gain" vs Vb'+str(savestr))
        ax1.set_xlabel('Bias Voltage [V]')
        ax1.set_ylabel('"Gain" [V*bins]')
        ax1.scatter(64.3,0)
        ax1.scatter(Vb[ignorel:ignorer:],G1R[ignorel:ignorer:],c=c[i],label='Regr '+str(step1str),s=40)
        #ax1.scatter(Vb[ignorel:ignorer:],G1[ignorel:ignorer:],c=c[i+1],label=str(step1str),s=40)
        ax1.scatter(Vb[ignorel:ignorer:],G2R[ignorel:ignorer:],c=c[i+2],label='Regr '+str(step2str),s=40)
        #ax1.scatter(Vb[ignorel:ignorer:],G2[ignorel:ignorer:],c=c[i+3],label=str(step2str),s=40)
        ax1.plot(Vb,(Vb)*slope1+inter1,c=c[i])
        ax1.plot(Vb,(Vb)*slope2+inter2,c=c[i+2])


        plt.legend(loc=2)

        fig2 = plt.figure(2)
        ax2 = fig2.add_axes([0.1, 0.1, 0.8, 0.8])
        ax2.grid(True)
        ax2.set_ylim(-0.05,9e6)

        ax2.set_title('DCR vs Vb'+str(savestr)) 
        ax2.set_xlabel('Bias Voltage [V]')
        ax2.set_ylabel('DCR [Hz]')
        ax2.scatter(Vb[ignorel:ignorer:],DCR1R[ignorel:ignorer:],c=c[i],label='Regr '+str(step1str),s=40)
        #ax2.scatter(Vb[ignorel:ignorer:],DCR1[ignorel:ignorer:],c=c[i+1],label=str(step1str),s=40)
        ax2.scatter(Vb[ignorel:ignorer:],DCR2R[ignorel:ignorer:],c=c[i+2],label='Regr '+str(step2str),s=40)
        #ax2.scatter(Vb[ignorel:ignorer:],DCR2[ignorel:ignorer:],c=c[i+3],label=str(step2str),s=40)


        plt.legend(loc=2)

        fig3 = plt.figure(3)
        ax3 = fig3.add_axes([0.1, 0.1, 0.8, 0.8])
        ax3.grid(True)
        ax3.set_ylim(-0.05,1.1)
        ax3.set_title('OCT vs Vb'+str(savestr))
        ax3.set_xlabel('Bias Voltage [V]')
        ax3.set_ylabel('OCT [%]')
        ax3.scatter(Vb[ignorel:ignorer:],OCT1R[ignorel:ignorer:],c=c[i],label='Regr '+str(step1str),s=40)
        #ax3.scatter(Vb[ignorel:ignorer:],OCT1[ignorel:ignorer:],c=c[i+1],label=str(step1str),s=40)
        ax3.scatter(Vb[ignorel:ignorer:],OCT2R[ignorel:ignorer:],c=c[i+2],label='Regr '+str(step2str),s=40)
        #ax3.scatter(Vb[ignorel:ignorer:],OCT2[ignorel:ignorer:],c=c[i+3],label=str(step2str),s=40)

        plt.legend(loc=2)

        '''
        fig4 = plt.figure(4)
        ax4 = fig4.add_axes([0.1, 0.1, 0.8, 0.8])
        ax4.grid(True)
        ax4.set_title('AbsGain vs Vb'+str(savestr))
        ax4.set_xlabel('Bias Voltage [V]')
        ax4.set_ylabel('AbsGain [mV/p.e.]')
        ax4.scatter(Vb[ignorel:ignorer:],OCT[ignorel:ignorer:],c=c[i],s=30,label=str(T)+'$^\circ$ '+str(date))
        plt.legend(loc=2)
        '''







        plt.draw()

    fig1.savefig(savepath+str(savestr)+'Gain_vs_Vb.pdf',format='pdf', bbox_inches='tight')
    fig2.savefig(savepath+str(savestr)+'DCR_vs_Vb.pdf',format='pdf', bbox_inches='tight')
    fig3.savefig(savepath+str(savestr)+'OCT_vs_Vb.pdf',format='pdf', bbox_inches='tight')
    #fig4.savefig(savepath+str(savestr)+'AbsGain_vs_Vb.pdf',format='pdf', bbox_inches='tight')

    plt.show()




def main():
    Date = ['1611161']#,'1711161','2111162']#,'2508162','2508163']#,'1011161']#,'2508163','0411161']'1011161'
    print 'printing ',str(Date),' for comparison'

    left = 0
    right = 50
    Plot_All(Date,left,right)

if __name__ == '__main__':
    main()

