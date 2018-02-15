import matplotlib                                                              
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os.path

datastr = 'FixedGuess3mV520IntWinMPD63'

step1str = 'FixedGuess3mV520IntWinMPD63Xbar'# 1 'FixedGuess' 2 'Gainx0p25' 3 'FixedValXXmV'
step2str = 'FixedGuess3mV520IntWinMPD63DeltaX'# 1 'FixedGuess' 2 'Gainx0p25' 3 'FixedValXXmV'

save1str = 'FixedGuess3mV520IntWinMPD63DeltaXBar'# 1 'FixedGuess' 2 'Gainx0p25' 3 'FixedValXXmV'
#save2str = ''# 1 'FixedGuess' 2 'Gainx0p25' 3 'FixedValXXmV'
#save3str = 'FixedVal2mV'
#save4str = 'Gainx0p25'# 1 'FixedGuess' 2 'Gainx0p25' 3 'FixedValXXmV'

save5str = 'Compare1.3_520IntWin_MaxhistGain_Gainregr'


class PlotGainArea(object):

    def __init__(self,iterT,iterV,destT,v,filename,savepath):
        
        self.filename = filename
        self.savepath = savepath
        self.volt = v
        self.temp = destT
        self.iterT = iterT
        self.iterV = iterV



    def PlotGainArea(self):
        # reads in Data from the sub-experiment save dest.
        print 'Run iterator: ',self.iterT
        self.pulse_area_list1 = np.load(self.filename+str(datastr)+'Area.npy')
        #self.pulse_area_list2 = np.load(self.filename+str(step2str)+'Area.npy')
        #self.pulse_area_list3 = np.load(self.filename+str(save1str)+'Area.npy')


        BINS = 2000
        self.hist_entr1,areaborders1=np.histogram(self.pulse_area_list1,bins = BINS)
        self.area1 = (areaborders1[:-1]+areaborders1[1:])/2
        #self.hist_entr2,areaborders2=np.histogram(self.pulse_area_list2,bins = BINS)
        #self.area2 = (areaborders2[:-1]+areaborders2[1:])/2
        #self.hist_entr3,areaborders3=np.histogram(self.pulse_area_list3,bins = BINS)
        #self.area3 = (areaborders3[:-1]+areaborders3[1:])/2



        
        vB = self.volt
        regrfile1 = np.load(self.savepath+str(step1str)+'AreaRelGainRegrLineData.npy')
        slope1 = regrfile1[1][self.iterT]
        inter1 = regrfile1[2][self.iterT]
        gain_regr1 = (slope1*vB+inter1)

        regrfile2 = np.load(self.savepath+str(step2str)+'AreaRelGainRegrLineData.npy')
        slope2 = regrfile2[1][self.iterT]
        inter2 = regrfile2[2][self.iterT]
        gain_regr2 = (slope2*vB+inter2)


        datalist = np.load(self.savepath+str(step1str)+'AreaNPYDataTable.npy')
        GainFit = datalist[2][self.iterT][self.iterV] #mix deltaX Xbar
        GainMaxHist = datalist[6][self.iterT][self.iterV]



        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.set_xlim([-0.1,1.])#np.amax(numbin)/2])
        ax.set_ylim([1,5*np.amax(self.hist_entr1)])
        ax.set_yscale('log')
        ax.set_ylabel('log$_{10}$ (Events) [#]')
        ax.set_xlabel('(Counts) [V bins]')
        ax.set_title('gain_regr-0.5gain_regr')

        ax.plot(self.area1,self.hist_entr1,'lightblue',label=step1str)
        #ax.plot(self.area2,self.hist_entr2,'lightgreen',label=step2str)
        #ax.plot(self.area3,self.hist_entr3,'bb',label=save3str)


        ax.plot((gain_regr1,gain_regr1), (0,np.amax(self.hist_entr1) *4.5), 'r--',label='GainRegrXbar')
        ax.plot((gain_regr2,gain_regr2), (0,np.amax(self.hist_entr1) *4.5), 'g--',label='GainRegrDeltaX')

        #ax.plot((GainFit,GainFit), (0,np.amax(self.hist_entr1)*4), 'g--',label='GainFit')
        #ax.plot((Gain,Gain), (0,np.amax(self.hist_entr1) *3), 'r--',label='GainMaxHist+-GainRegr')



        gainhalf1 = gain_regr1-(0.5*gain_regr1)
        gainonehalf1 = gain_regr1 + (0.5*gain_regr1) 
        ax.plot((gainhalf1,gainhalf1), (0,np.amax(self.hist_entr1)*4.5 ), 'r.-')      
        ax.plot((gainonehalf1,gainonehalf1), (0,np.amax(self.hist_entr1) *4.5), 'r.-')

        #gainhalf = GainFit-(0.5*GainFit)
        #gainonehalf = GainFit + (0.5*GainFit) 
        #ax.plot((gainhalf,gainhalf), (0,np.amax(self.hist_entr2) *4), 'g.-')      
        #ax.plot((gainonehalf,gainonehalf), (0,np.amax(self.hist_entr2) *4), 'g.-')

        gainhalf2 = gain_regr2-(0.5*gain_regr2)
        gainonehalf2 = gain_regr2 + (0.5*gain_regr2) 
        ax.plot((gainhalf2,gainhalf2), (0,np.amax(self.hist_entr1)*4.5 ), 'g.-')      
        ax.plot((gainonehalf2,gainonehalf2), (0,np.amax(self.hist_entr1) *4.5), 'g.-')


        plt.xticks(np.arange(0 ,0.9+0.1, 0.1))



        plt.legend()
        print 'saving plot of ',self.filename
        fig.savefig(self.savepath+str(save5str)+str(self.volt)+'V_Regr.pdf',format='pdf', bbox_inches='tight')
        if args.show == 1:plt.show()
        #plt.show()
        
        plt.close()
        #plt.draw()



parser1 = argparse.ArgumentParser(description='Decide:')
parser1.add_argument('-s','--show',type=int,help='-sx or --show x | 1 show last segment')
parser1.add_argument('-d','--date',type=str,help='-dx or --date x | date id of the experiment to reduce')
parser1.add_argument('-r','--regr',type=int,help='-rx or --regr x | 0 =NO regr , 1=fromregr ')




args = parser1.parse_args()
#print args



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




def main():
    print args
    # GLobal vars , appended
   
    datE = '1611161'#'2111162'
    Date, T, Vb = LoadRunCard(datE)
   
    #Vb = [67.4]
    

    for dAte in Date:
        savepath = '/home/gebhardt/00_SiPM_MPIK/scripts/Data/Date'+str(dAte)+'/'
        for iterT,destT in enumerate(T):
            for indv,v in enumerate(Vb):
                compl_name = GetNames(destT,v,dAte)
                print compl_name
                PlotGainArea(iterT,indv,destT,v,compl_name,savepath).PlotGainArea()
          
if __name__ == '__main__':
    main()

