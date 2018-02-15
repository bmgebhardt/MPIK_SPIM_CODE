#import locale
#locale.getdefaultlocale()[0]

# scripts : WaveformReduction

import matplotlib                                                              
#matplotlib.use('Agg')  
import WaveformReduction
import math
import scipy
#import numpy as np
import numpy as np
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import asarray as ar,exp
from scipy.optimize import curve_fit
from detect_peaks import detect_peaks
import pdb
import os.path
import time

np.set_printoptions(precision=4)

fromregr = False

def gaus(x,a,x0,sigma):
    #a a:amplitude,x0:mean,sigma:sigma
    return a*exp(-(x-x0)**2/(2*sigma**2))


class GaussFit(object):

    def __init__(self,iterT,destT,v,filename,savepath):
        self.gaintemp = []
        self.sigmatemp = []
        self.amptemp = []
        self.valley = []
        self.bincut = []
        self.histcut= []
        self.gain= []
        self.dcr = []
        self.filename = filename
        self.savepath = savepath
        self.volt = v
        self.temp = destT
        self.peak = []
        self.iterator = iterT

    def create_data(self):
        # reads in Data from the sub-experiment save dest.
        print 'Run iterator: ',self.iterator
        self.pulse_are_list = np.load(self.filename+'Area.npy')
        self.pulse_area_list = np.load(self.filename+'AreaClean.npy')

        self.zero = np.load(self.filename+'ZeroMean.npy')
        self.zerostd = np.std(np.load(self.filename+'ZeroMean.npy'))

        BINS = 2000
        self.hist_entr,areaborders=np.histogram(self.pulse_area_list,bins = BINS)
        #,range=(0.,2.) ) #care_fil you might change DCR and OCT with that
        self.area = (areaborders[:-1]+areaborders[1:])/2
        
        '''
        plt.figure(2)
        #self.hist_entr=np.append(self.hist_entr,self.hist_entr[-1])
        plt.yscale('log')
        plt.hist(self.zero,bins=20,histtype='step')
        plt.plot(self.area,self.hist_entr)
        plt.show()
        '''

        self.zeroentr,zeroborders =np.histogram(self.zero,bins = 10)
        self.zeroarea = (zeroborders[:-1]+zeroborders[1:])/2
        self.zeromax = self.zeroarea[np.argmax(self.zeroentr)]
        print self.zeromax
        print self.area[int(self.zeromax)]    

        self.GetValley()
        print 'valley',self.valley
        if len(self.valley)>1:
            self.bincut1 = self.area[self.valley[0]:self.valley[1]:]
            self.histcut1 = self.hist_entr[self.valley[0]:self.valley[1]:]   
            if len(self.valley)>2:
                self.bincut2 = self.area[self.valley[1]:self.valley[2]:]
                self.histcut2 = self.hist_entr[self.valley[1]:self.valley[2]:] 
                if len(self.valley)>3:
                    self.bincut3 = self.area[self.valley[2]:self.valley[3]:]
                    self.histcut3 = self.hist_entr[self.valley[2]:self.valley[3]:] 


        #if os.path.isfile(self.savepath+'AreaRelGainRegrLineData.npy'):
        if fromregr:
            regrfile = np.load(self.savepath+'AreaRelGainRegrLineData.npy')
            slope = regrfile[1][self.iterator]
            inter = regrfile[2][self.iterator]
            self.gaintemp.append(slope*self.volt+inter)
            self.initvalue='Initial Value (from prev. Run)'
            print 'calculated gain from regr line:',self.gaintemp
        else:
            self.gaintemp.append(self.area[np.argmax(self.hist_entr)])
            self.initvalue='max(hist) as Initial Value'
        self.sigmatemp.append(np.std(self.pulse_area_list,dtype=np.float64)) 

    def GetValley(self):
        #either calculates valley position via regression line or peak detection, if no regr line file is found
        #if os.path.isfile(self.savepath+'AreaRelGainRegrLineData.npy'):
        if fromregr:
            regrfile = np.load(self.savepath+'AreaRelGainRegrLineData.npy')
            slope = regrfile[1][self.iterator]
            inter = regrfile[2][self.iterator]
            self.peak.append(np.argmax(self.area >(slope*self.volt+inter)))
            print 'calculated peak from Regr Line ',self.peak
        else:
            #calculates valley positions to cut on during border creation for gauss fit
            print 'argmax'
            self.peak.append(np.argmax(self.hist_entr))
        #firstvalley = np.where(self.area>0)[0][0]
        print self.area[self.peak[0]],' - ',self.zeromax
        firstvalley = (self.area[self.peak[0]]-self.zeromax)/2
        print firstvalley,' firstvalley'
        firstvalleyindex = np.where(self.area>firstvalley)[0][0]
        print firstvalleyindex,' firstv index'
        #print firstvalley

#-------------------------
        print 'peaks ',self.peak
        dist = self.peak[0]-firstvalleyindex
        self.valley.append(firstvalleyindex)
        self.valley.append(self.peak[0]+dist)
        '''
        if len(self.peak)>1:
            dist = (self.peak[1]-self.peak[0])/2
            for pos in range(len(self.peak)):
                self.valley.append(self.peak[pos]+dist/2)
        else:
            self.valley.append(dist/2 + self.peak[0])
        '''
    def Plot(self,numbin,data,popt,mean,meanfit,zerobin,zero):

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        #ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
        ax.set_xlim([-0.01,0.12])#np.amax(numbin)/2])
        #ax.set_ylim([1,zero[np.argmax(zero)-1]])
        ax.set_ylim([1,1.1*np.amax(zero)])
        ax.set_yscale('log')
        ax.set_ylabel('log$_{10}$ (Events) [#]')
        ax.set_xlabel('(Counts) [V bins]')
        #ax.title('Pulse Area Fit')
        ax.plot(numbin,data,'b',label='PA Histogram Data')
        ax.plot(numbin,gaus(numbin,*popt),'r-',linewidth=2,label='Fit')  
        
        ax.plot(zerobin,zero,'g',label='ZeroPeak')
        #ax.plot((zerobin[np.mean(zero)],zerobin[np.mean(zero)]),(0,5*np.amax(data)),'g--',label='Zero Mean')
        ax.plot((self.zeromax,self.zeromax),(0,np.amax(data)+50),'g')

        ax.plot((mean,mean ), (0,np.amax(data)+20 ), 'b--',label=self.initvalue)
        meanstr = 'Initial: '+str(mean)
        ax.text(mean,10,meanstr,fontsize = 12)

        ax.plot((meanfit,meanfit), (0,np.amax(data) ), 'r--',label='Gain from fit')
        gainstr = 'Fit: '+str(meanfit)
        ax.text(meanfit,15,gainstr,fontsize = 12)

        #ax.plot((0,np.amax(zerobin[np.argmax(zero)])),(mean,np.amax(data)+1000),'k')
        ax.plot((numbin[self.valley[0]],numbin[self.peak[0]]),(np.amax(data)+20,np.amax(data)+20),'k-')
        ax.plot((self.zeromax,numbin[self.peak[0]]),(np.amax(data)+30,np.amax(data)+30),'k-')


        for i in self.valley:
            if i == 0:
                border = self.area[i]
                ax.plot((border,border), (0,np.amax(data)+20 ), 'k--',label='Border')
            else:
               border = self.area[i]
               ax.plot((border,border), (0,np.amax(data)+20 ), 'k--')
        ax.text(border,20,'Border',fontsize = 12)

        plt.legend()
        figure = plt.gcf() # get current figure
        figure.set_size_inches(7, 7)

        print 'saving plot of ',self.filename
        fig.savefig(self.filename+'PulseAreaFit.pdf',format='pdf', bbox_inches='tight')
        plt.show()
        
        plt.close()
        #plt.draw()



    def GetGainDCR(self,time):
        #Main function call
        self.gain = []
        self.create_data()
        try:
            popt,pcov = curve_fit(gaus,self.bincut1,self.histcut1,p0=[100,self.gaintemp[0],self.sigmatemp[0]/20])#p0 = ampl,mean,sigma
            print 200,' ',self.gaintemp[0],' ',self.sigmatemp[0]
            print 'POPT ',popt
            print "PCOV ",pcov
            print 
            perr = np.sqrt(np.diag(pcov))
            #print 'perr ', perr
            self.amptemp.append(popt[0])
            self.gaintemp.append(popt[1])
            self.sigmatemp.append(popt[2])
            self.gain.append(self.gaintemp[1]-self.zeromax)
            self.Plot(self.area,self.hist_entr,popt,self.gaintemp[0],self.gaintemp[1],self.zeroarea,self.zeroentr)

        except RuntimeError: 
            print "First Peak Optimal parameters not found | Rebinning"
            self.histcut1,self.bincut1=np.histogram(self.histcut1,bins=(len(self.bincut1/2)))
            self.bincut1 = (self.bincut1[:-1]+self.bincut1[1:])/2
            popt,pcov = curve_fit(gaus,self.bincut1,self.histcut1,p0=[1,self.gaintemp[0],self.sigmatemp[0]])#p0 = ampl,mean,sigma

            print 'POPT ',popt
            print "PCOV ",pcov
            print 
            perr = np.sqrt(np.diag(pcov))
            #print 'perr ', perr
            self.amptemp.append(popt[0])
            self.gaintemp.append(popt[1])
            self.sigmatemp.append(popt[2])
            self.gain.append(self.gaintemp[1]-self.zeromax)
            #self.Plot(self.area,self.hist_entr,popt,self.gaintemp[0],self.gaintemp[1])
            #self.Plot(self.bincut1,self.histcut1,popt,self.gaintemp[0],self.gaintemp[1],self.zeroarea,self.zeroentr)
            #plt.show()




            '''
            self.gain.append(self.gaintemp[0])
            perr = [1,1,1]
            popt = [0,0,0]
            '''
            #plt.show()

        print 'gaintemp ', self.gaintemp
        if self.gaintemp[1]<1: #skip DCR,OCT calc if gain not realistic(if fit didnt work, take amax of histogramm
            DDCR,CDCR,OCT_d,OCT_c = self.GetDCROCT(time,False)
        else:
            DDCR,CDCR,OCT_d,OCT_c = self.GetDCROCT(time,True)
        #print self.gain

        return self.gain,perr,DDCR,CDCR,OCT_d,OCT_c,self.zerostd


    def GetDCROCT(self,time,skip):
        if skip:self.gain[0] = self.gaintemp[0]
        #calculate DCR, OCT 
        #if len(self.gain)==0:self.gain=self.peak[0]
        gainhalf = self.gain[0]/2.
        gainonehalf = self.gain[0]*3/2    
        pulse_a_d = np.asarray(self.pulse_are_list)
        pulse_a_c = np.asarray(self.pulse_area_list)

        pulse_a_d_array0p5 = pulse_a_d > gainhalf
        pulse_a_c_array0p5 = pulse_a_c > gainhalf       
        dirty_DCR = (pulse_a_d_array0p5).sum()/time
        clean_DCR = (pulse_a_c_array0p5).sum()/time

        pulse_a_d_array1p5 = pulse_a_d > gainonehalf
        pulse_a_c_array1p5 = pulse_a_c > gainonehalf       
        #print (pulse_a_d_array1p5).sum()
        #print (pulse_a_d_array0p5).sum()

        OCT_d = (float((pulse_a_d_array1p5).sum()))/float(((pulse_a_d_array0p5).sum()))
        OCT_c = (float((pulse_a_c_array1p5).sum()))/float(((pulse_a_c_array0p5).sum()))

        return dirty_DCR, clean_DCR, OCT_d, OCT_c



        
        
        

def GetNames(destT,destV,Date):
    #generate sub-experiment name
    path = '/home/gebhardt/00_SiPM_MPIK/scripts/Data/Date'+str(Date)+'/'+str(destT)+'deg/'+str(destV)+'V/'
    name = 'SensL_T' + str(destT) + '_Vb' + str(destV) + '.trc'
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
   
    # GLobal vars , appended
    glT=[]
    glV=[]
    glG=[]
    gldDCR=[]
    glcDCR=[]
    gldOCT=[]
    glcOCT=[]
    glerror = []
    
    datE = ['300616']
    Date, T, Vb = LoadRunCard(datE[0])
    #T = [26.]
    # Date = ['2508162']
    #presets of experiment data
    #T = [25.]
    #Vb = [29.]#[27.,28.,39.,30.,31.,32.]

    #vBlist = np.arange(26.,33.1,0.2)
    #vB = vBlist


    for dAte in Date:
        savepath = '/home/gebhardt/00_SiPM_MPIK/scripts/Data/Date'+str(dAte)+'/'
        # higher order save destination of one experiment
        iterT = 0
        for destT in T:
            gaintemp = []
            errortemp = []
            ddcrtemp = []
            cdcrtemp = []
            cocttemp = []
            docttemp = []
            for v in Vb:
                compl_name = GetNames(destT,v,dAte)
                print compl_name
                time = WaveformReduction.WfRed(compl_name,savepath).LoadTime()
                gainlist,err,dirtydcr,cleandcr,doctalk,coctalk,zerostd = GaussFit(iterT,destT,v,compl_name,savepath).GetGainDCR(time)
                gain = gainlist[0] #its not a mean, since its only the first calculated gain from 0-peak to the 1pe peak
                par_error = err[1]+abs(zerostd)
                #err[1] is one standard deviation error of the parameter 'gain' |
                #zerostd is one standard deviation error of the zeropeak|
                # gain = spe - meanzero -> error(gain) = error(spe)+error(meanzero)
                gaintemp.append(gain)
                errortemp.append(par_error)
                ddcrtemp.append(dirtydcr)
                cdcrtemp.append(cleandcr)
                docttemp.append(doctalk)
                cocttemp.append(coctalk)
                #append for every voltage

            glG.append(gaintemp)
            glerror.append(errortemp)
            gldDCR.append(ddcrtemp)
            glcDCR.append(cdcrtemp)
            gldOCT.append(docttemp)
            glcOCT.append(cocttemp)
            #append for every temperature
            iterT+=1
            
    T = np.asarray(T)
    Vb = np.asarray(Vb)
    glG = np.asarray(glG)
    glerror = np.asarray(glerror)
    gldDCR = np.asarray(gldDCR)
    glcDCR = np.asarray(glcDCR)
    gldOCT = np.asarray(gldOCT)
    glcOCT = np.asarray(glcOCT)
    #print [T,Vb,glG,glerror,gldDCR,glcDCR,gldOCT,glcOCT] 

    np.save(savepath+'AreaNPYDataTable',[T,Vb,glG,glerror,gldDCR,glcDCR,gldOCT,glcOCT]) 


if __name__ == '__main__':
    main()

