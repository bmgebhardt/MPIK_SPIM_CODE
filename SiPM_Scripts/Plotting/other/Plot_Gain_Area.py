import matplotlib                                                              
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os.path
datastr = 'FixedGuess3mV520IntWinMPD63'# 1 'FixedGuess' 2 'Gainx0p25' 3 'FixedValXXmV'

step1str = 'FixedGuess3mV520IntWinMPD63DeltaX'# 1 'FixedGuess' 2 'Gainx0p25' 3 'FixedValXXmV'
savestr = 'FixedGuess3mV520IntWinMPD63DeltaX'# 1 'FixedGuess' 2 'Gainx0p25' 3 'FixedValXXmV'



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
        self.pulse_area_list = np.load(self.filename+str(datastr)+'Area.npy')

        BINS = 2000
        self.hist_entr,areaborders=np.histogram(self.pulse_area_list,bins = BINS)
        self.area = (areaborders[:-1]+areaborders[1:])/2
        
        vB = self.volt
        regrfile = np.load(self.savepath+str(step1str)+'AreaRelGainRegrLineData.npy')
        slope = regrfile[1][self.iterT]
        inter = regrfile[2][self.iterT]

        gain_regr = (slope*vB+inter)
        datalist = np.load(self.savepath+str(step1str)+'AreaNPYDataTable.npy')
        Gain = datalist[6][self.iterT][self.iterV]




        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.set_xlim([-0.1,1.])#np.amax(numbin)/2])
        ax.set_ylim([1,5*np.amax(self.hist_entr)])
        ax.set_yscale('log')
        ax.set_ylabel('log$_{10}$ (Events) [#]')
        ax.set_xlabel('(Counts) [V bins]')
        ax.set_title('gain_regr-0.5gain_regr')

        ax.plot(self.area,self.hist_entr,'b',label='PA Histogram Data')
        ax.plot((gain_regr,gain_regr), (0,np.amax(self.hist_entr)+20 ), 'b--',label='Gain')
       

        gainhalf = gain_regr-(0.5*gain_regr)
        gainonehalf = gain_regr+(0.5*gain_regr) 
        ax.plot((gainhalf,gainhalf), (0,np.amax(self.hist_entr) ), 'b.-',label='0.5')      
        ax.plot((gainonehalf,gainonehalf), (0,np.amax(self.hist_entr) ), 'b.-',label='1.5')

        plt.xticks(np.arange(0 ,0.9+0.1, 0.1))



        #plt.legend()
        print 'saving plot of ',self.filename
        fig.savefig(self.savepath+str(savestr)+str(self.volt)+'V_gain_regr-0.5gain_regr.pdf',format='pdf', bbox_inches='tight')
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

