#Calculation from Regression Line Data derived Gain

import numpy as np
import os.path
import LecroyBinarySlicing
import matplotlib.pyplot as plt
import argparse


#data string
datastr = 'FixedGuess4mV1010IntWinMPD10'# 1 'FixedGuess' 2 'Gainx0p25' 3 'FixedValXXmV'
datastrH = 'FixedGuess4mV1010IntWinMPD10'# 1 'FixedGuess' 2 'Gainx0p25' 3 'FixedValXXmV'


#regrssion line data
step1str = 'FixedGuess4mV1010IntWinMPD10_deltax'
step2str = 'FixedGuess4mV1010IntWinMPD10_xbar'

#path to regression line
savefolderpathregr = 'FixedGuess4mV1010IntWinMPD10'


#savename
savestr =  'FixedGuess4mV1010IntWinMPD10'


#savefolder destination
savefolderpath = 'FixedGuess4mV1010IntWinMPD10'


class RegrCalc(object):

    def __init__(self,iterT,iterV,destT,v,filename,savepathregr):
        self.filename = filename
        self.savepathregr = savepathregr
        self.volt = v
        self.temp = destT
        self.iterT = iterT
        self.iterV = iterV


    def create_data(self):
        # reads in Data from the sub-experiment save dest.
        self.pulse_area_list = np.load(self.filename+str(datastr)+'Area.npy') 



    def Calc(self,time):
        self.create_data()
        #print 'Calculating from regression line'
        #savepathregr = '/home/gebhardt/00_SiPM_MPIK/scripts/Data/Date'+str(date)+'/'+str(savefolderpathregr)+'/'


        regrfiledeltax = np.load(self.savepathregr+str(step1str)+'AreaRelGainRegrLineData.npy')
        slopedeltax = regrfiledeltax[1][self.iterT]
        interdeltax = regrfiledeltax[2][self.iterT]
        print datastr,self.savepathregr,step1str, slopedeltax,interdeltax

        regrfilexbar = np.load(self.savepathregr+str(step2str)+'AreaRelGainRegrLineData.npy')
        slopexbar = regrfilexbar[1][self.iterT]
        interxbar = regrfilexbar[2][self.iterT]


        datalist = np.load(self.savepathregr+str(datastrH)+'AreaNPYDataTable.npy')
        #xbar = datalist[6][self.iterT][self.iterV]
        #deltax = datalist[7][self.iterT][self.iterV]
        #print xbar
        #from IPython import embed;embed();1/0

        tempxbar = (slopexbar*self.volt+interxbar)
        tempdeltax = (slopedeltax*self.volt+interdeltax)
        #gainhalf = tempxbar-tempdeltax/2. # use this for all else
        #gainonehalf = tempxbar+tempdeltax/2.
        gainhalf = tempxbar-tempdeltax/2. 
        gainonehalf = tempxbar+tempdeltax/2. #Lct5 deltax and xbar same, but not at 35deg 



        '''
        self.hist_entr,areaborders=np.histogram(self.pulse_area_list,bins = 1000)
        self.area = (areaborders[:-1]+areaborders[1:])/2

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.set_xlim([0.,0.5])#np.amax(numbin)/2])
        plt.xticks(np.arange(0. ,0.5, 0.1))

        ax.set_ylim([1,5*np.amax(self.hist_entr)])
        ax.set_yscale('log')
        ax.set_ylabel('log$_{10}$ (Events) [#]')
        ax.set_xlabel('(Counts) [V bins]')
        #ax.title('Pulse Area Fit')
        #ax.plot(self.area0,self.hist_entr0,color='lightgrey')
        ax.plot(self.area,self.hist_entr,color='lightgrey')


        ax.plot((xbar,xbar), (0,np.amax(self.hist_entr) +50),'r--',label='xbar',linewidth=2)
        ax.plot((gainhalf,gainhalf), (0,np.amax(self.hist_entr) +50),'g--',label='1/2',linewidth=2)
        ax.plot((gainonehalf,gainonehalf), (0,np.amax(self.hist_entr) +50),'g--',label='1/2',linewidth=2)
        plt.show()
        '''


        pulse_a = np.asarray(self.pulse_area_list)

        pulse_a_array0p5 = pulse_a > gainhalf #True or False
        DCR = (pulse_a_array0p5).sum()/time

        pulse_a_array1p5 = pulse_a > gainonehalf #True or False
        OCT = (float((pulse_a_array1p5).sum()))/float(((pulse_a_array0p5).sum()))    # our way of OCT calc
        #OCT = (float((pulse_a_array1p5).sum()))/float(len(pulse_a))    # nagoya

        all_DCR = float(len(pulse_a))/time

        returngain = tempxbar #Gain
        return returngain,DCR, OCT, all_DCR

def GetNames(destT,destV,Date):
    #generate sub-experiment name (of waveform reduced datafile)
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
    length = LecroyBinarySlicing.LecroyBinaryWaveform(path).WaveArray()
    time = length/samplerate
    #recordtime = LecroyBinary.LecroyBinaryWaveform(path).ReturnRecordTime() # Time Bin empty???
    #print recordtime
    return time

def Plot_All(Date,ignorel,ignorer):
    for i,date in enumerate(Date):
        AbsGain = []
        savepathregr = '/home/gebhardt/00_SiPM_MPIK/scripts/Data/Date'+str(date)+'/'+str(savefolderpathregr)+'/'
        savepath = '/home/gebhardt/00_SiPM_MPIK/scripts/Data/Date'+str(date)+'/'+str(savefolderpath)+'/'

        regr_dataarray = np.load(savepath+str(savestr)+'AreaNPYDataTableRegr.npy')
        #no_regr_dataarray = np.load(savepathregr+str(datastr)+'AreaNPYDataTable.npy')
        regrfile = np.load(savepathregr+str(step2str)+'AreaRelGainRegrLineData.npy') #xbar for LCT5

        slope = regrfile[1][0] #T
        inter = regrfile[2][0] #T
        T = regr_dataarray[0]
        Vb = regr_dataarray[1]
        G = regr_dataarray[2][0]
        DCR = regr_dataarray[3][0]
        OCT = regr_dataarray[4][0]
        all_DCR = regr_dataarray[5][0]

        #deltax = 
        #no_regr_G = no_regr_dataarray[2]

        '''
        wl_bin = 4*2.5
        wr_bin = 5*2.5
        pulse_geom = 1.3
        conv_factor = (wl_bin+wr_bin)*pulse_geom
        for j in range(len(no_regr_G)):
            AbsGain.append(no_regr_G[j]*1000/conv_factor) 
        '''

        c = ['b','g','r']

        x = np.arange(51,63)

        fig1 = plt.figure(1)
        ax1 = fig1.add_axes([0.1, 0.1, 0.8, 0.8])
        ax1.grid(True)
        ax1.set_title('"Gain" vs Vb'+str(savestr))
        ax1.set_xlabel('Bias Voltage [V]')
        ax1.set_ylabel('"Gain" [V]')
        #for i in range(len(T)):
        ax1.scatter(Vb[ignorel:ignorer:],G[ignorel:ignorer:],c=c[i])
        #ax1.scatter(Vb[ignorel:ignorer:],no_regr_G[ignorel:ignorer:],c=c[i],s=30,label=str(T)+'$^\circ$')
        ax1.plot(x,(x)*slope+inter,c=c[i],label=step1str)
        ax1.text((-inter/slope),0,str(-inter/slope))

        plt.legend(loc=2)

        fig2 = plt.figure(2)
        ax2 = fig2.add_axes([0.1, 0.1, 0.8, 0.8])
        ax2.grid(True)
        ax2.set_title('DCR vs Vb'+str(savestr)) 
        ax2.set_xlabel('Bias Voltage [V]')
        ax2.set_ylabel('DCR [Hz]')
        #for i in range(len(T)):
        ax2.scatter(Vb[ignorel:ignorer:],DCR[ignorel:ignorer:],c=c[i],s=30,label=str(T)+'$^\circ$ '+str(date))
        ax2.scatter(Vb[ignorel:ignorer:],all_DCR[ignorel:ignorer:],marker='+',c=c[i],s=30,label=str(T)+'$^\circ$ all_DCR'+str(date))

        plt.legend(loc=2)

        fig3 = plt.figure(3)
        ax3 = fig3.add_axes([0.1, 0.1, 0.8, 0.8])
        ax3.grid(True)
        ax3.set_ylim(0,0.9)

        ax3.set_title('OCT vs Vb'+str(savestr))
        ax3.set_xlabel('Bias Voltage [V]')
        ax3.set_ylabel('OCT [%]')
        #for i in range(len(T)):
        ax3.scatter(Vb[ignorel:ignorer:],np.asarray(OCT[ignorel:ignorer:]),c=c[i],s=30,label=str(T)+'$^\circ$ '+str(date))
        
        plt.legend(loc=2)

        '''
        fig4 = plt.figure(4)
        ax4 = fig4.add_axes([0.1, 0.1, 0.8, 0.8])
        ax4.grid(True)
        ax4.set_title('AbsGain vs Vb'+str(savestr))
        ax4.set_xlabel('Bias Voltage [V]')
        ax4.set_ylabel('AbsGain [mV/p.e.]')
        #for i in range(len(T)):
        ax4.scatter(Vb[ignorel:ignorer:],AbsGain[ignorel:ignorer:],c=c[i],s=30,label=str(T)+'$^\circ$ '+str(date))
        plt.legend(loc=2)
        '''







        plt.draw()

    fig1.savefig(savepath+str(savestr)+'Regr_Gain_vs_Vb.pdf',format='pdf', bbox_inches='tight')
    fig2.savefig(savepath+str(savestr)+'Regr_DCR_vs_Vb.pdf',format='pdf', bbox_inches='tight')
    fig3.savefig(savepath+str(savestr)+'Regr_OCT_vs_Vb.pdf',format='pdf', bbox_inches='tight')
    #fig4.savefig(savepath+str(savestr)+'Regr_AbsGain_vs_Vb.pdf',format='pdf', bbox_inches='tight')

    plt.show()


def Calculate(Date):

    #T = [26.]
    for dAte in Date:
        TGainl = []
        TDCRl = []
        TOCTl = []
        TDCR_all_l = []



        VGainl = []
        VDCRl = []
        VOCTl = []
        VDCR_all_l = []

        date, T, Vb = LoadRunCard(dAte)
        print T
        date = date[0]

        #if date == ('2508162')or('2508163'):
        #   T = [26.]
        #if date == ('1011161')or('1411161'):
        #else:
        #   T = [25.]
        savepathregr = '/home/gebhardt/00_SiPM_MPIK/scripts/Data/Date'+str(date)+'/'+str(savefolderpathregr)+'/'  #overwrite in calc
        for iterT,destT in enumerate(T):
            for iterV,v in enumerate(Vb):
                print destT,' ',v
                compl_name = GetNames(destT,v,dAte)
                #print compl_name
                time = LoadTime(compl_name)
                #print time
                try:
                    Gain,DCR,OCT,all_DCR = RegrCalc(iterT,iterV,destT,v,compl_name,savepathregr).Calc(time)
                except (TypeError,IndexError,ZeroDivisionError,RuntimeError,ValueError) as errorstr:
                    print errorstr
                    Gain,DCR,OCT,all_DCR = 'nan','nan','nan','nan'
                VGainl.append(Gain)
                VDCRl.append(DCR)
                VOCTl.append(OCT)
                VDCR_all_l.append(all_DCR)
            TGainl.append(VGainl)
            TDCRl.append(VDCRl)
            TOCTl.append(VOCTl)
            TDCR_all_l.append(VDCR_all_l)



        T = np.asarray(T,dtype=float)
        Vb = np.asarray(Vb,dtype=float)
        TGainl = np.asarray(TGainl,dtype=float)
        TDCRl = np.asarray(TDCRl,dtype=float)
        TOCTl = np.asarray(TOCTl,dtype=float)
        TDCR_all_l = np.asarray(TDCR_all_l,dtype=float)


        savepath = '/home/gebhardt/00_SiPM_MPIK/scripts/Data/Date'+str(date)+'/'+str(savefolderpath)+'/'

        np.save(savepath+str(savestr)+'AreaNPYDataTableRegr',[T,Vb,TGainl,TDCRl,TOCTl,TDCR_all_l])
        np.save(savepath+str(savestr)+'DCRfromtotalRegr',[T,Vb,TDCR_all_l])

        print 'saved'



parser = argparse.ArgumentParser(description='Decide:')
parser.add_argument('-c','--calc',type=int,help='-cx or --calc x | 0 no, 1 yes')
args = parser.parse_args()






def main():
    date = ['1611161']#[['2811162'],['2111161'],['1611161'],['2111162'],['2811163']]    #todo all other dates!!
    #print 'printing ',str(date),' from regr line'
    if args.calc == 1:
        Calculate(date)
    left = 0
    right = 50
    Plot_All(date,left,right)

if __name__ == '__main__':
    main()


