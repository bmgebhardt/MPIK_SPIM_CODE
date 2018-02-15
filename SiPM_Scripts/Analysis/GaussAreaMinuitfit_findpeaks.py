#GaussFit Minuit of HAMWaveformreductionSlice.py


#import locale
#locale.getdefaultlocale()[0]


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
from scipy import signal
from detect_peaks import detect_peaks
import pdb
import os.path
import time
import argparse as argp
from iminuit import Minuit


'''
def multigauss(x,*params):
    y = np.zeros_like(x)
    xbar = params[0]
    corr_gain = params[1]
    for i in range(0,len(params)-2,2): # Nrang given to func via parameter-array-size, steps=3
        ampl = params[i+2]
        sig = params[i+3]
        y = y + ampl*exp(-(x-(xbar+(i/4*corr_gain)))**2./(2.*sig**2.)) 
    return y
'''
'''
datastr =    'FixedGuess3mV520IntWinMPD25'# 1 'FixedGuess' 2 'Gainx0p25' 3 'FixedVal'
savestr =    'FixedGuess3mV520IntWinMPD25'# 1 'FixedGuess' 2 'Gainx0p25' 3 'FixedVal'
savefolderpath =   'FixedGuess3mV520IntWinMPD25'# 1 'FixedGuess' 2 'Gainx0p25' 3 'FixedVal'
'''

datastr =    'FixedGuess4mV1010IntWinMPD10'# 1 'FixedGuess' 2 'Gainx0p25' 3 'FixedVal'
savestr =    'FixedGuess4mV1010IntWinMPD10'# 1 'FixedGuess' 2 'Gainx0p25' 3 'FixedVal'
savefolderpath =   'FixedGuess4mV1010IntWinMPD10'# 1 'FixedGuess' 2 'Gainx0p25' 3 'FixedVal'


class multigauss_chi2:
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def multigauss(self,*params):
        f = np.zeros_like(self.x)
        xbar = params[0]
        corr_gain = params[1]
        for i in range(0,len(params)-2,2): # Nrang given to func via parameter-array-size, steps=3
            ampl = params[i+2]
            sig = params[i+3]
            #f = f + ampl*exp(-(self.x-(xbar+(i/2*corr_gain)))**2./(2.*sig**2.))  
            f = f + ampl*exp(-(self.x-(xbar+(i/2*corr_gain)))**2./(2.*sig**2.))  
        return f
    def __call__(self,*params): 
        chi2 = np.sum(np.power(self.y-self.multigauss(*params),2))
        return chi2



np.set_printoptions(precision=4)

fromregr = False
calcfromregr = False

def gaus(x,a,x0,sigma):
    return a*exp(-(x-x0)**2./(2.*sigma**2.))




class GaussFit(object):

    def __init__(self,iterT,destT,v,filename,savepath,prevgain,high_pass):
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
        self.previous_peak = 0
        self.prevgain = prevgain
        self.high_pass = high_pass




    def create_data(self):
        # reads in Data from the sub-experiment save dest.
        print 'Run iterator: ',self.iterator
        self.pulse_area_list = np.load(self.filename+str(datastr)+'Area.npy')



        self.zero = np.load(self.filename+str(datastr)+'ZeroPeak.npy')
        self.zerostd = np.std(np.load(self.filename+str(datastr)+'ZeroPeak.npy'))

        BINS = 2000
        self.hist_entr,areaborders=np.histogram(self.pulse_area_list,bins = BINS)
        '''
        self.hist_entr0,areaborders0=np.histogram(self.pulse_area_list,bins = BINS)
        noiseEND = np.where(areaborders0>=(0.1))[0][0]     
        self.hist_entr = self.hist_entr0[noiseEND:]
        areaborders = areaborders0[noiseEND:]      
        self.area0= (areaborders0[:-1]+areaborders0[1:])/2
        '''
        self.area = (areaborders[:-1]+areaborders[1:])/2

        '''
        plt.figure(2)
        plt.yscale('log')
        plt.hist(self.zero,bins=500,histtype='step')
        plt.plot(self.area,self.hist_entr)
        plt.show()
        plt.close()
        '''

        self.zeroentr,zeroborders =np.histogram(self.zero,bins = 10)
        self.zeroarea = (zeroborders[:-1]+zeroborders[1:])/2
        self.zeromax = self.zeroarea[np.argmax(self.zeroentr)]
        print self.zeromax
        print self.area[int(self.zeromax)]    
        self.GetValley()
        p1= self.valley[0]
        p2= self.valley[1]

        self.bincut1 = self.area[p1:p2:]
        self.histcut1 = self.hist_entr[p1:p2:]


        #if parsargs.regr==1:
        conv_fac = (float(int(4*2.5)+int(5*2.5)))*1.3
        if fromregr:
            vB = self.volt
            #regrfile = np.load(self.savepath+str(datastr)+'AreaRelGainRegrLineData.npy')
            slope = regrfile[1][self.iterator]
            inter = regrfile[2][self.iterator]
            gain_regr = (slope*vB+inter)
            if gain_regr < self.high_pass*conv_fac:
                self.gaintemp.append(self.high_pass*conv_fac)
            else:
                self.gaintemp.append(slope*vB+inter)
            self.initvalue='Initial Value (from prev. Run)'
            print 'calculated gain from regr line:',self.gaintemp
        else:
            #self.gaintemp.append(self.area[np.argmax(self.hist_entr)]) # > XXX
            print 'argmax'
            #maximum = np.maximum(np.argmax(self.hist_entr),np.where(self.area>=(self.high_pass*conv_fac))[0][0])
            #maximum = np.argmax(self.hist_entr) # as to not corrupt lower gains ####
            #print maximum

            print 'argrelmax'
        

            self.gaintemp.append(self.xbar)
            self.initvalue='max(hist) as Initial Value'
        self.sigmatemp.append(np.std(self.pulse_area_list,dtype=np.float64)) 

    def GetValley(self):
        print 'Previous Peak ',self.previous_peak
        #either calculates valley position via regression line or max hist, if no regr line file is found
        #if os.path.isfile(self.savepath+'AreaRelGainRegrLineData.npy'):
        #if parsargs.regr==1:
        if fromregr:
            #parsargs.regr = 1
            vB = self.volt
            conv_fac = (float(int(4*2.5)+int(5*2.5)))*1.3
            #regrfile = np.load(self.savepath+str(datastr)+'AreaRelGainRegrLineData.npy')
            slope = regrfile[1][self.iterator]
            inter = regrfile[2][self.iterator]
            gain_regr = (slope*vB+inter)
            if gain_regr < self.high_pass*conv_fac:
                self.peak.append(np.argmax(self.area >=(self.high_pass*conv_fac)))
            else:
                self.peak.append(np.argmax(self.area >=(slope*vB+inter)))
            print 'calculated peak from Regr Line ',self.peak
        else:
            conv_fac = (float(int(4*2.5)+int(5*2.5)))*1.3
            #calculates valley positions to cut on during border creation for gauss fit
            print 'argrelmax'
            #maximum = np.maximum(np.argmax(self.hist_entr),np.where(self.area>=(self.high_pass*conv_fac))[0][0])
            maximum = np.argmax(self.hist_entr) # as to not corrupt lower gains ####

            #scipy.signal.find_peaks_cwt
            #peak_ind = signal.argrelmax(self.hist_entr,order=2000)
            #print peak_ind
            #if peak_ind[0]<(peak_ind[2]-peak_ind[1]):
            #maximum2 = peak_ind[0][0]






            self.peak.append(maximum) #in bin-number






        #print self.area[self.peak[0]],' - ',self.zeromax
        self.xbar = self.area[self.peak[0]] #for use in multigauss

        firstvalley = (self.area[self.peak[0]]-self.zeromax)*0.5 #makes window narrow 0.5
        #print firstvalley,' firstvalley'
        firstvalleyindex = np.where(self.area>=firstvalley)[0][0]
        #print firstvalleyindex,' firstv index'

        #print 'peaks ',self.peak
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
        self.previous_peak = self.peak
        '''
        plt.figure(1)
        plt.yscale('log')
        plt.plot(self.area,self.hist_entr,c='lightgrey',zorder=0)
        plt.scatter(self.area[maximum],self.hist_entr[maximum],s=60)
        plt.scatter(self.area[self.valley],self.hist_entr[self.valley],c ='r',s=100)
        plt.show(1)
        '''



        #from IPython import embed;embed();1/0




    def Plot(self,numbin,data,popt,zerobin,zero,model,Nrang,xbar,deltax,fitted_gain):



        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.set_xlim([-0.1,1])#np.amax(numbin)/2])
        plt.xticks(np.arange(-0.1 ,1.0, 0.1))

        ax.set_ylim([1,5*np.amax(data)])
        ax.set_yscale('log')
        ax.set_ylabel('log$_{10}$ (Events) [#]')
        ax.set_xlabel('(Counts) [V bins]')
        #ax.title('Pulse Area Fit')
        #ax.plot(self.area0,self.hist_entr0,color='lightgrey')
        ax.plot(self.area,self.hist_entr,color='lightgrey')


        ax.plot(numbin,data,'grey',label='PA Histogram Data')

        ax.plot(numbin,gaus(numbin,*popt),'r-',linewidth=2,label='CurveFit')  
        
        #ax.plot(zerobin,zero,'g',label='ZeroPeak')
        #ax.plot((zerobin[np.mean(zero)],zerobin[np.mean(zero)]),(0,5*np.amax(data)),'g--',label='Zero Mean')
        #ax.plot((self.zeromax,self.zeromax),(0,np.amax(data)+50),'g')


        #ax.plot((mean,mean ), (0,np.amax(data)+20 ), 'b--',label=self.initvalue)
        #meanstr = 'Initial: '+str(mean)
        #ax.text(mean,10,meanstr,fontsize = 12)

        #ax.plot((meanfit,meanfit), (0,np.amax(data) ), 'r--',label='Gain from fit')
        #gainstr = 'Fit: '+str(meanfit)
        #ax.text(meanfit,15,gainstr,fontsize = 12)

        #ax.plot((0,np.amax(zerobin[np.argmax(zero)])),(mean,np.amax(data)+1000),'k')
        #ax.plot((numbin[self.valley[0]],numbin[self.peak[0]]),(np.amax(data)+20,np.amax(data)+20),'k-')
        #ax.plot((self.zeromax,numbin[self.peak[0]]),(np.amax(data)+30,np.amax(data)+30),'k-')

        #ax.plot(numbin,multigauss(numbin,*popt2),'g--',linewidth=2,label='Fit 2')  
        ax.plot(numbin,model,'k--',linewidth=4,label='MinuitFit')




        #ax.plot((xbar+0*xbar,xbar+0*xbar), (0,np.amax(data) +50), 'r--',label='xbar',linewidth=2)
        #ax.plot((xbar+0*deltax,xbar+0*deltax), (0,np.amax(data) +25), 'g--',label='xbar+deltax',linewidth=2)
        #ax.plot((xbar+0*fitted_gain,xbar+0*fitted_gain), (0,np.amax(data) ), 'b--',label='from minuit fit',linewidth=2)

        ax.plot((xbar+(0)*xbar,xbar+(0)*xbar), (0,np.amax(data) +50),'r--',label='Pos xbar+xbar',linewidth=2)
        ax.plot((xbar+(0)*deltax,xbar+(0)*deltax), (0,np.amax(data) +25), 'g--',label='Pos xbar+deltax',linewidth=2)   
        #ax.plot(0,0, 'b--',label='from minuit fit',linewidth=2)

        for i in range(Nrang):
            
            #ax.plot((xbar+(i)*xbar,xbar+(i)*xbar), (np.amax(data) +50*i,np.amax(data) +50*i), 'r.-',linewidth=2)
            #ax.plot((xbar+(i)*deltax,xbar+(i)*deltax), (np.amax(data) +25*i,np.amax(data) +25*i), 'g.-',linewidth=2)


            ax.plot((xbar+(i)*xbar,xbar+(i)*xbar), (0,np.amax(data) +50), 'r--',linewidth=2)
            ax.plot((xbar+(i)*deltax,xbar+(i)*deltax), (0,np.amax(data) +25), 'g--',linewidth=2)
            #ax.plot((xbar+(i+1)*fitted_gain,xbar+(i+1)*fitted_gain), (0,np.amax(data) ), 'b--',linewidth=2)



        '''
        dist = self.xbar - minuitfit
        ax.plot((self.xbar,self.xbar), (0,np.amax(data) ), 'g.-',label='from minuit fit')
        ax.plot((self.xbar+minuitfit,self.xbar+minuitfit), (0,np.amax(data) ), 'g.-',label='from minuit fit')
        ax.plot((self.xbar+2*minuitfit,self.xbar+2*minuitfit), (0,np.amax(data) ), 'g.-',label='from minuit fit')
        '''


        gainhalf = xbar-(0.5*xbar)
        gainonehalf = xbar + (0.5*xbar) 
        ax.plot((gainhalf,gainhalf), (0,np.amax(data) ), 'r.-')      
        ax.plot((gainonehalf,gainonehalf), (0,np.amax(data) ), 'r.-')

        ax.text(gainhalf,np.amax(data)*0.9,'0.5 xbar',fontsize = 12,horizontalalignment='right',color='r')
        ax.text(gainonehalf,np.amax(data)*0.9,'1.5 xbar',fontsize = 12,color='r')

        gainhalf = xbar-(0.5*deltax)
        gainonehalf = xbar + (0.5*deltax) 
        ax.plot((gainhalf,gainhalf), (0,np.amax(data)*1.1 ), 'g.-')      
        ax.plot((gainonehalf,gainonehalf), (0,np.amax(data)*1.1 ), 'g.-')

        ax.text(gainhalf,np.amax(data)*0.6,'0.5 deltax',fontsize = 12,horizontalalignment='right',color='g')
        ax.text(gainonehalf,np.amax(data)*0.6,'1.5 deltax',fontsize = 12,color='g')






        '''
        for i in self.valley:
            if i == 0:
                border = self.area[i]
                ax.plot((border,border), (0,np.amax(data)+20 ), 'k--',label='Border')
            else:
               border = self.area[i]
               ax.plot((border,border), (0,np.amax(data)+20 ), 'k--')
        ax.text(border,20,'Border',fontsize = 12)
        '''
        plt.legend()


        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()   

        print 'saving plot of ',self.savepath+str(savestr)+'_'+str(self.volt)+'V_PulseAreaFit.pdf'
        fig.savefig(self.savepath+str(savestr)+'_'+str(self.volt)+'V_PulseAreaFit.pdf',format='pdf', bbox_inches='tight')
        if parsargs.show == 1:plt.show()
        #plt.show()
        
        plt.close()
        #plt.draw()



    def GetGainDCR(self,time):
        #Main function call
        self.gain = []
        self.create_data()
        #popt,pcov = curve_fit(gaus,self.bincut1,self.histcut1,p0=[self.hist_entr[self.area[self.gaintemp[0]]],self.gaintemp[0],self.sigmatemp[0]/20])#p0 = ampl,mean,sigma

        try:
            popt,pcov = curve_fit(gaus,self.bincut1,self.histcut1,p0=[self.hist_entr[np.where(self.area>=self.gaintemp[0])[0][0]],self.gaintemp[0],self.sigmatemp[0]/10])
            print 'POPT ',popt

            perr = np.sqrt(np.diag(pcov))
            self.amptemp.append(popt[0])
            self.gaintemp.append(popt[1])
            self.sigmatemp.append(popt[2])
            #self.Plot(self.area,self.hist_entr,popt,self.gaintemp[0],self.gaintemp[1],self.zeroarea,self.zeroentr,0)

        except RuntimeError: 
            print "First Peak Optimal parameters not found | Rebinning"
            self.histcut1,self.bincut1=np.histogram(self.histcut1,bins=(len(self.bincut1/2)))
            self.bincut1 = (self.bincut1[:-1]+self.bincut1[1:])/2
            popt,pcov = curve_fit(gaus,self.bincut1,self.histcut1,p0=[1,self.gaintemp[0],self.sigmatemp[0]])#p0 = ampl,mean,sigma
            perr = np.sqrt(np.diag(pcov))
            self.amptemp.append(popt[0])
            self.gaintemp.append(popt[1])
            self.sigmatemp.append(popt[2])
            #self.Plot(self.area,self.hist_entr,popt,self.gaintemp[0],self.gaintemp[1])
            #self.Plot(self.bincut1,self.histcut1,popt,self.gaintemp[0],self.gaintemp[1],self.zeroarea,self.zeroentr)
            #plt.show()





        #----------- correct gain calc here

        fitted_gain,xbar,deltax = self.GetCorrectGain(popt)

        self.gaintemp.append(fitted_gain)
        self.gain.append(fitted_gain)
        print 'GAIN: ',self.gain



        if fitted_gain<1: #skip DCR,OCT calc if gain not realistic(if fit didnt work, take amax of histogramm
            DDCR,CDCR,OCT_d,OCT_c,all_DCR = self.GetDCROCT(time,False)
        else:
            DDCR,CDCR,OCT_d,OCT_c,all_DCR = self.GetDCROCT(time,True)

        return self.gain,perr,DDCR,CDCR,OCT_d,OCT_c,self.zerostd,all_DCR,xbar,deltax



    def GetCorrectGain(self,popt):
        print 'Fitting Multiple Gaussians'
        Nrang = self.GetRange()
        p1 = np.where(self.area>=self.gaintemp[1]*(0.75))[0][0] #left border
        p2 = np.where(self.area>=self.gaintemp[1]*(Nrang+0.3))[0][0] #right border
        self.hist_to_fit = self.hist_entr[p1:p2:]
        self.bin_to_fit = self.area[p1:p2:]

        chi2 = multigauss_chi2(self.bin_to_fit,self.hist_to_fit)
        #define params tuple, initial values, limits, etc
        ptup=()
        kwdarg={}
        ptup=ptup+('xbar',)
        kwdarg['xbar']=self.gaintemp[1]
        kwdarg['error_xbar']=self.gaintemp[1]/10
        kwdarg['limit_xbar']=(self.gaintemp[1]/10,10)
        ptup=ptup+('deltax',)
        kwdarg['deltax']=self.gaintemp[1]
        kwdarg['error_deltax']=self.gaintemp[1]/10
        kwdarg['limit_deltax']=(self.gaintemp[1]/5,10)
        for n in range(Nrang):
            ptup=ptup+('A_'+str(n),)
            kwdarg['A_'+str(n)]=self.hist_entr[np.where(self.area>=self.gaintemp[1]*(n+1))[0][0]]
            kwdarg['error_A_'+str(n)]=(self.hist_entr[np.where(self.area>=self.gaintemp[1]*(n+1))[0][0]])/10
            kwdarg['limit_A_'+str(n)]=(0.,1.e8)
            ptup=ptup+('Sig_'+str(n),)
            kwdarg['Sig_'+str(n)]=abs(self.sigmatemp[1])
            kwdarg['error_Sig_'+str(n)]=abs(self.sigmatemp[1])/2
            kwdarg['limit_Sig_'+str(n)]=(1.e-3,0.5)
            #create minuit object, minimize, return results

        m = Minuit(chi2, forced_parameters=ptup,errordef=1,print_level=1,**kwdarg)
        #print m.values
        #model=chi2.multigauss(*m.args)

        print "now fitting--------------------------"
        fitres=m.migrad()[0]
        model=chi2.multigauss(*m.args)
        '''
        print 'fitres: '
        print fitres
        print 'values: ',m.values
        print 'args: ',m.args[1]
        '''
        #print m.args[1],' ',self.prevgain*0.9
        self.xbar = m.values['xbar']







        if (Nrang<3):     ## was 3 for ChecS and LVR6mm                            
            print 'Range <3 I'
            fitted_gain = m.args[0]#self.gaintemp[1] #xbar

        else:   
            
            if m.args[1]<(self.prevgain*0.8): #
                fitted_gain = m.args[0]
                print 'xbar ,smaller prev gain'

            else:
                #else returns delta-x
            
                fitted_gain = m.args[1]
                print 'delta_x'

        print 'fitted gain is ',fitted_gain
        deltax = m.args[1]
        xbar = m.args[0]    

        try:
            self.Plot(self.bin_to_fit,self.hist_to_fit,popt,self.zeroarea,self.zeroentr,model,Nrang,xbar,deltax,fitted_gain)
        #self.Plot(self.area,self.hist_entr,popt,self.gaintemp[0],self.gaintemp[1],self.zeroarea,self.zeroentr,model,Nrang)
        except ValueError:
            print 'no data'


        return fitted_gain,xbar,deltax

    def GetRange(self):
        i=1

        bl = 10
        br = 10
        #print 'entry ',self.hist_entr[np.where(self.area>=self.gaintemp[0])[0][0]]
        #print 'entry ',self.hist_entr[np.where(self.area>=self.xbar)[0][0]]/100


        thresh=np.maximum(np.amax(self.hist_entr[np.where(self.area>=self.gaintemp[0]*(0.8)*i)[0][0]-bl:np.where(self.area>=self.gaintemp[0]*(0.8)*i)[0][0]+br:])/40.,20)
        #thresh=np.maximum(self.hist_entr[np.where(self.area>=self.xbar)[0][0]]/30.,10)
        print "Threshold ",thresh
        #print self.hist_entr[np.where(self.area>=self.gaintemp[0]*i)[0][0]-bl:np.where(self.area>=self.gaintemp[0]*i)[0][0]+br:]
        #[i-bl:i+br:]
        #np.where(self.area>=self.gaintemp[0]*i)[0][0])-bl:np.where(self.area>=self.gaintemp[0]*i)[0][0])+br:
        
        while np.any((self.hist_entr[np.where(self.area>=self.gaintemp[0]*i)[0][0]-bl:np.where(self.area>=self.gaintemp[0]*i)[0][0]+br:])>(thresh)):

        #while (self.hist_entr[np.where(self.area>=self.gaintemp[0]*i)[0][0]])>(thresh):
            i+=1    
        rang = i-1
        rang = np.minimum(5,rang)
        print 'Range ',rang
        return rang

    def GetDCROCT(self,time,skip):
        if skip:self.gain[0] = self.gaintemp[0]
        #calculate DCR, OCT 
        xbar = self.xbar

        if calcfromregr == False:
            print 'calculating from fit '
            gainhalf = xbar - (self.gain[-1]/2.)
            gainonehalf = xbar + (self.gain[-1]/2.) 
            #print gainhalf
            #print gainonehalf   
        if calcfromregr == True:
            print 'Calculating from regression line'
            #regrfile = np.load(self.savepath+'AreaRelGainRegrLineData.npy')
            slope = regrfile[1][self.iterator]
            inter = regrfile[2][self.iterator]
            tempgain = (slope*self.volt+inter)
            gainhalf = tempgain/2.
            gainonehalf = tempgain*3./2.
            #print gainhalf
            #print gainonehalf
        pulse_a_d = np.asarray(self.pulse_area_list)
        pulse_a_c = np.asarray(self.pulse_area_list)


        pulse_a_d_array0p5 = pulse_a_d > gainhalf
        pulse_a_c_array0p5 = pulse_a_c > gainhalf       
        dirty_DCR = (pulse_a_d_array0p5).sum()/time
        clean_DCR = (pulse_a_c_array0p5).sum()/time

        pulse_a_d_array1p5 = pulse_a_d > gainonehalf
        pulse_a_c_array1p5 = pulse_a_c > gainonehalf       


        OCT_d = (float((pulse_a_d_array1p5).sum()))/float(((pulse_a_d_array0p5).sum()))
        OCT_c = (float((pulse_a_c_array1p5).sum()))/float(((pulse_a_c_array0p5).sum()))

        all_DCR = float(len(pulse_a_d))/time

        return dirty_DCR, clean_DCR, OCT_d, OCT_c,all_DCR


parser1 = argp.ArgumentParser(description='Decide:')
parser1.add_argument('-s','--show',type=int,help='-sx or --show x | 1 show last segment')
parser1.add_argument('-d','--date',type=str,help='-dx or --date x | date id of the experiment to reduce')
parser1.add_argument('-r','--regr',type=int,help='-rx or --regr x | 0 =NO regr , 1=fromregr ')




parsargs = parser1.parse_args()
#print parsargs



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
    print parsargs
    # GLobal vars , appended
    TGainl = []
    TErrorl = []
    TDCRl = []
    TOCTl = []
    TDCR_all_l = []
    T_xbar = []
    T_deltax = []



    if parsargs.date:
        datE = parsargs.date
        print datE
    else:
        datE = '1611161'
        print datE
    Date, T, Vb = LoadRunCard(datE)
    #T = [26.]
    #Vb = [67.8,68.,68.2]
    #Vb = [39.0,39.1,39.2,39.3,39.4,39.5,39.7,39.9,40.,40.2,40.5,41.0,41.2,41.5]
    #Vb = [53.,53.5,54.,54.5,55.,55.5,56.,56.5,57.,57.5,58.,58.5,59.,59.5,60.,60.5,61.]
    #Vb = [67.7,67.5,67.7,68.,68.2,68.4]
    #Vb = [68.,69.,69.2,69.4]
    #Vb = [67.,67.1,67.2,67.3,67.4,67.5]

    #Date = ['2508162']
    #presets of experiment data
    #T = [26.]
    #Vb = [67.5,67.6,67.7,67.8]

    #T = [25.0,30.0,35.0]
    #vBlist = np.arange(56.5,61.1,0.5)
    #Vb = vBlist
    high_pass = 0.003#in V

    for dAte in Date:
        savepath = '/home/gebhardt/00_SiPM_MPIK/scripts/Data/Date'+str(dAte)+'/'+str(savefolderpath)+'/'
        if not os.path.isdir(savepath):
            os.mkdir(savepath)
        # higher order save destination of one experiment
        for iterT,destT in enumerate(T):
            VGainl = []
            VErrorl = []
            VDCRl = []
            VOCTl = []
            VDCR_all_l = []
            V_xbar = []
            V_deltax = []

            prevgain = 0
            for indv,v in enumerate(Vb):
                
                #print indv
                try: # for use overnight
                    compl_name = GetNames(destT,v,dAte)
                    print compl_name
                    time = WaveformReduction.WfRed(compl_name,savepath).LoadTime()
                    print 'time ',time
                    gainlist,error,dirtydcr,cleandcr,dirtyoct,cleanoct,zerostd,all_DCR,xbar,deltax= GaussFit(iterT,destT,v,compl_name,savepath,prevgain,high_pass).GetGainDCR(time)
                    par_error = error[1]+abs(zerostd)
                    gain = gainlist[0]
                    prevgain = gain
                except (TypeError,IndexError,ZeroDivisionError,RuntimeError,ValueError) as errorstr: #add ValueError again
                    gainlist,error,dirtydcr,cleandcr,dirtyoct,cleanoct,zerostd,all_DCR,xbar,deltax= 'nan','nan','nan','nan','nan','nan','nan','nan','nan','nan'
                    print '                      ',errorstr,'                             '
                    gain = gainlist
                    par_error = 'nan'
                VGainl.append(gain)
                VDCRl.append(dirtydcr)
                VOCTl.append(dirtyoct)
                VDCR_all_l.append(all_DCR)
                VErrorl.append(par_error)
                V_xbar.append(xbar)                 
                V_deltax.append(deltax)


            TGainl.append(VGainl)
            TDCRl.append(VDCRl)
            TOCTl.append(VOCTl)
            TDCR_all_l.append(VDCR_all_l)
            TErrorl.append(VErrorl)
            T_xbar.append(V_xbar)                 
            T_deltax.append(V_deltax)
            #append for every temperature


    T = np.asarray(T,dtype=float)
    Vb = np.asarray(Vb,dtype=float)
    TGainl = np.asarray(TGainl,dtype=float)
    TDCRl = np.asarray(TDCRl,dtype=float)
    TOCTl = np.asarray(TOCTl,dtype=float)
    TErrorl = np.asarray(TErrorl,dtype=float)
    T_xbar = np.asarray(T_xbar,dtype=float)
    T_deltax = np.asarray(T_deltax,dtype=float)
    TDCR_all_l = np.asarray(TDCR_all_l,dtype=float)

    print TDCRl
    print TDCR_all_l
    print TOCTl
  
    np.save(savepath+str(savestr)+'AreaNPYDataTable',[T,Vb,TGainl,TErrorl,TDCRl,TOCTl,T_xbar,T_deltax])
    np.save(savepath+str(savestr)+'DCRfromtotal',[T,Vb,TDCR_all_l])

  
if __name__ == '__main__':
    main()

