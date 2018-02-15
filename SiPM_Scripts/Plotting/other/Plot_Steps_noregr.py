
# TODO:
# list -> np.array (faster?)

import matplotlib
#matplotlib.use('Agg')
import sys
sys.path.append('/home/gebhardt/00_SiPM_MPIK/scripts')
import LecroyBinary
import numpy as np
from scipy import signal as sig
from detect_peaks import detect_peaks
from memory_profiler import profile
import argparse
#import matplotlib
import matplotlib.pyplot as plt
import os.path
import numpy.ma as ma
matplotlib.use('Agg')



plt.rcParams['agg.path.chunksize'] = 10000



def log_10_product(x, pos):
    """The two args are the value and tick position.
    Label ticks with the product of the exponentiation"""
    return '{:.3e}'.format(x)



class WfRed(object):

    def __init__(self,data_file_path,savepath):
        self.raw_data = np.zeros(1)
        self.filtered_signal_1 = np.zeros(1)
        self.filtered_signal_2 = np.zeros(1)
        self.ranger = 0
        self.ite = 0
        self.pulse_height = []
        self.pulse_height_clean = []
        self.pulse_area = []
        self.pulse_area_clean = []
        self.samplerate = 2.5e9 
        self.time = np.zeros(1) 
        self.peakless = []
        self.meanzero = [] 
        self.sigmazero = []
        self.data_file_path = data_file_path
        self.savepath = savepath
        self.calcsegmentspro = 100

    def ReportProgress(self,ite,seg):
        pro = ite*100/seg
        if pro%10==0:
            print pro,' % done'


    def GetSegmentCount(self):

        self.ranger = LecroyBinary.LecroyBinaryWaveform(self.data_file_path).SubArray()
        return self.ranger

    def Load_Data(self,it1):
        """
        Loopable through counter in function call
        raw_data from filecontent
        """
        # return all segments 
        self.raw_data,self.time=LecroyBinary.LecroyBinaryWaveform(self.data_file_path).RealGetNextWaveArrayDataAndTime(it1)# in V and s
        #self.raw_data = self.raw_data*1e3 #in mV
        self.time = (self.time-self.time[0])*1e6 #starts at 0, in us
        return self.raw_data

    def RemovePeaksBasedOnRMS3(self,wave_uncentered):
        """
        gets negative of waveform and calculates rms
        uses rms to calculate peakless waveform        
        """   
        mean0 = np.mean(wave_uncentered)
        #print mean0
        wave = wave_uncentered - mean0 #wave at 0
        '''
        mean1 = np.mean(wave)
        print mean1
        indices_negative2 = np.where(wave<mean1)[0] 
        wave_cut_negative2 = np.zeros(wave.size) 
        wave_cut_negative2[indices_negative2] = wave[indices_negative2] 
        mean2 = np.mean(wave_cut_negative2)
        print mean2

        indices_negative = np.where(wave<mean1)[0] 
        # search indices, where wave is below upper limit 0
        wave_cut_negative = np.zeros(wave.size) 
        # create np array of zeros of aquivalent size
        wave_cut_negative[indices_negative] = wave[indices_negative] 
        # fill only those indices, where wave is below upper limit -> peakless signal

        rms = np.std(wave_cut_negative) # get rms of negative signal
        print rms
        #indices = np.where(np.abs(wave)<rms*2)[0] # wave between rms*2, both sides
        indices = np.where(wave<rms)[0] # wave between rms*2, both sides # changed to cutting only top peaks, due to pedestal sub bug
        wave_cut = np.zeros(wave.size) 
        wave_cut[indices] = wave[indices] #fill wave 

        cut = rms #cut-level
        '''

        mean1 = np.mean(wave)  #mean1 ~ 0
        indices_negative2 = np.where(wave<mean1)[0] 
        wave_cut_negative2 = np.zeros(wave.size) 
        wave_cut_negative2[indices_negative2] = wave[indices_negative2] 

        mean = np.mean(wave_cut_negative2)
        indices_negative = np.where(wave<mean1)[0] 
        indices_positive = np.where(wave>mean1)[0]
        wave_cut_negative = np.zeros(wave.size) 
        wave_cut_negative[indices_negative] = wave[indices_negative] 
        wave_cut_negative[indices_positive] = -wave[indices_positive] 

        rms = np.std(wave_cut_negative) # get rms of negative signal
        #print rms
        indices = np.where(np.abs(wave)<rms*3)[0] # wave between rms*2, both sides
        #indices = np.where(wave<rms)[0] # wave between rms*2, both sides # changed to cutting only top peaks, due to pedestal sub bug
        wave_cut = np.zeros(wave.size) 
        wave_cut[indices] = wave[indices] #fill wave 

        cut = rms #cut-level
        #print cut

        '''
        plt.figure()
        plt.title('RemovePeaksBasedOnRMS3')
        plt.xlabel('Time [$\mu$s]')
        plt.ylabel('Voltage [V]')
        plt.plot(self.time,wave,'grey',label='wave')
        plt.plot(self.time,wave_cut,'red',label='wave_cut_negative')
        plt.plot((np.amin(self.time),np.amax(self.time)), (cut,cut), 'k',label='cut level')
#       plt.plot((np.amin(self.time),np.amax(self.time)), (-rms*rmsn,-rms*rmsn), 'b--',label='peak detect level')
#        plt.plot((np.amin(self.time),np.amax(self.time)), (np.mean(filtered_signal_2),np.mean(filtered_signal_2)), 'b--',label='MeanPeakLess') #rmsn from RemovePeaksBasedOnRMS3 = 2
        
        plt.legend()
        #return pulse_pos
        plt.show()
        '''
        return wave_cut+mean0,cut   #add mean0 aggain for script to exactly subtract the pedestal again in slownoise



    def Slownoise_Smooth(self,data,ite):
        """
        generates smoothed signal from peak-less signal and raw_data
        takes: peak-less signal , raw_data
        returns: smoothed signal 1
        """
        
        slownoise,level = self.RemovePeaksBasedOnRMS3(data)
        meanzero = np.mean(slownoise)
        
        self.peakless = slownoise - meanzero #peakless signal - pedestal mean
        meanpeakless = np.mean(self.peakless)
        #print meanpeakless
        #print meanzero, ' ',meanpeakless
        data_at_0 = data - meanzero #subtract mean of pedestal from data 
        # not given that trace is always with same offset // gain-calc only takes into account meanzero of last segment

        nsamples = 700#determines width of smoothing gaussian (/5) = sigma of gauss 
        window = sig.general_gaussian(nsamples, p=1.0, sig=nsamples/5)
        slownoise_smooth = sig.fftconvolve(self.peakless, window, "same")
        slownoise_smooth = (np.average(self.peakless) /np.average(slownoise_smooth)) * slownoise_smooth
        
        self.filtered_signal_1 = data_at_0 - slownoise_smooth

        
        if ite ==0:

            plt.figure()
            plt.plot(self.time,data_at_0)






            fig8 = plt.figure(8)
            #ax8 = fig8.add_axes([0.1, 0.1, 0.8, 0.8])
            #ax8.set_title('Filter Levels')
            #ax8.set_xlabel('Time [$\mu$]')
            #ax8.set_ylabel('Voltage [V]')
            ax81 = fig8.add_subplot(311)
            ax82 = fig8.add_subplot(312)
            ax83 = fig8.add_subplot(313)
            #ax84 = fig8.add_subplot(224)  
            #ax8.set_xlim(0,100)
            ax81.plot(self.time,data,'lightgrey',label='Raw Data')
            ax81.plot((np.amin(self.time),np.amax(self.time)), (meanzero,meanzero), 'b-',label='MeanZero') #rmsn from RemovePeaksBasedOnRMS3 = 2
            ax82.plot(self.time,data_at_0,'grey',label='Raw Data Corrected')
            ax82.plot((np.amin(self.time),np.amax(self.time)), (meanpeakless,meanpeakless), 'g-',label='MeanPeakLess') #rmsn from RemovePeaksBasedOnRMS3 = 2

            ax83.plot(self.time,data,'lightgrey',label='Raw Data')
            ax83.plot(self.time,slownoise,'grey',label='Slownoise')
            ax83.plot((np.amin(self.time),np.amax(self.time)), (meanzero,meanzero), 'b-',label='MeanZero') #rmsn from RemovePeaksBasedOnRMS3 = 2
            ax81.legend()
            ax82.legend()
            ax83.legend()
            fig8.savefig(self.data_file_path+'PeaklessGen.pdf',format='pdf', bbox_inches='tight')

            #rawdatcorr = ax8.plot(self.time,data_at_0,'grey',label='Raw Data Corrected')
            #slown = ax8.plot(self.time,slownoise,'red',label='slow Noise')
            #plt.plot((np.amin(self.time),np.amax(self.time)), (level*2,level*2), 'k--',label='Cut level') #rmsn from RemovePeaksBasedOnRMS3 = 2
            #plt.plot((np.amin(self.time),np.amax(self.time)), (meanzero,meanzero), 'b-',label='MeanZero') #rmsn from RemovePeaksBasedOnRMS3 = 2
            #plt.plot((np.amin(self.time),np.amax(self.time)), (meanpeakless,meanpeakless), 'b--',label='MeanPeakLess') #rmsn from RemovePeaksBasedOnRMS3 = 2
            #ax8.set_title('Smoothed slow Noise')
            #slowns = ax8.plot(self.time,slownoise_smooth,'blue',label='Smoothed slow Noise')
            #ax8.legend()
            plt.show()
            #fig8.savefig(self.data_file_path+'FilterLevels.pdf',format='pdf',bbox_inches='tight')
            #ax8.legend_.remove()
            #fig8.clf()
        
        self.ReportProgress(ite,self.ranger)
        return self.filtered_signal_1


    def Fastnoise_Smooth(self,filtered_signal_1):
        """
        smoothes noise-less signal again
        takes: noise-less signal
        returns: smoothed noise-less signal
       
        peak detection more reliable
        """
        nsamples =50#20
        window = sig.general_gaussian(nsamples, p=1.0, sig=nsamples/5)
        self.filtered_signal_2 = sig.fftconvolve(filtered_signal_1, window, "same")
        self.filtered_signal_2 = (np.average(filtered_signal_1) / np.average(self.filtered_signal_2)) * self.filtered_signal_2

        return self.filtered_signal_2


    def FindPeaks(self,filtered_signal_2,ite,rmsn):
        """
        searches for the peak position in the smoothed noise-less signal
        takes: smoothed noise-less signal
        returns: List of Peak Indeces
        """
        #rms = np.std(self.peakless)
        #print rms
        #print np.std(filtered_signal_2)
        #pulse_pos = detect_peaks(filtered_signal_2, rms*1, mpd=10,show=False)

        #rmsn = 3#1.5
        #print rmsn
        #RMS Calc from only negative part of self.peakless
        indices_negative = np.where(self.peakless<0)[0]  #fs2?
        # search indices, where wave is below upper limit 0, aka take negative
        peakless_cut_negative = np.zeros(len(self.peakless)) 
        # create np array of zeros of aquivalent size
        peakless_cut_negative[indices_negative] = self.peakless[indices_negative] 
        # fill only those indices, where wave is below upper limit -> peakless signal

        
        #additional RMS Calc based on band between RMS from negative peakless
        #still testing
        #rms_negative = np.std(peakless_cut_negative)
        #indices_rmsband = np.where(np.abs(self.peakless)<rms_negative)[0]  #fs2?
        #peakless_cut_rmsband = np.zeros(len(self.peakless)) 
        #peakless_cut_rmsband[indices_rmsband] = self.peakless[indices_rmsband] 
        #rms = np.std(peakless_cut_rmsband)
        #print rms
        #plt.figure()
        #plt.plot(self.time,peakless_cut_rmsband)
        #plt.show()




        #rms = 1
        #rmsn = rmsn/2/17 #for regression
        rms = np.std(self.peakless) # generate rms from peakless signal(noiselevel)
        print rms
        #print np.std(filtered_signal_2)
        #pulse_pos = detect_peaks(filtered_signal_2, rms*rmsn, mpd=30,show=False)
        pulse_pos = detect_peaks(filtered_signal_2, 2*rms, mpd=30,show=False)


        if ite ==0:
            fig9 = plt.figure(9)
            plt.title('signal 2 peak detect')
            plt.xlabel('Time [$\mu$s]')
            plt.ylabel('Voltage [V]')
            #plt.xlim([0,10])
            plt.plot(self.time,filtered_signal_2,'red',label='Filtered2')
            #plt.plot(self.time,peakless_cut,'black',label='Peakless')
            plt.plot((np.amin(self.time),np.amax(self.time)), (rms*rmsn,rms*rmsn), 'b--',label='peak detect level')
            plt.plot((np.amin(self.time),np.amax(self.time)), (-rms*rmsn,-rms*rmsn), 'b--',label='peak detect level')
            plt.plot((np.amin(self.time),np.amax(self.time)), (np.mean(self.peakless),np.mean(self.peakless)), 'g--',label='MeanPeakLess') #rmsn from RemovePeaksBasedOnRMS3 = 2
            
            plt.legend()
            fig9.savefig(self.data_file_path+'PeakDetectLevel.pdf',format='pdf', bbox_inches='tight')
            #plt.show()
            #plt.draw()

        
        return pulse_pos

    def PulseCleanUp(self,pulse_pos):
        """
        Generates List of Peaks without immidiate neighbor
        -> Cleaned up Peak positions
        takes: List of Pulse Positions
        returns: List of cleaned up Pulse positions
        """       
        wpeak = 250
        #is this peak clean?
        pulse_pos_clean = np.zeros(len(pulse_pos),dtype=int)
        peaknumber = np.arange(0,len(pulse_pos))
        look = 0
        while look in range(0,peaknumber.size-1):
            dt_next = abs(pulse_pos[look+1] - pulse_pos[look])
            #time_diff.append(dt_next)
            dt_last = abs(pulse_pos[look] - pulse_pos[look-1])

            if (dt_last >= wpeak) and (dt_next >= wpeak):
                pulse_pos_clean[look] =  pulse_pos[look]
            look +=1

        pulse_pos_clean = pulse_pos_clean[pulse_pos_clean != 0] 
        return pulse_pos_clean

    
    def Pulse(self,data,pulse_pos):
        """
        Generates List of Peak Heights an Peak Areas | Appends
        takes: noise-less signal , List of pulse positions
        returns: List of pulse heights and areas
        """
        wl = 7 # in samples, so 2.5 samples == 1 ns, 10 ns = 25 samples | 2.5GS/s = 0.4ns/S
        wr = 10
        for i, pos in enumerate(pulse_pos):
            start_bin =int(pos - (wl * (self.samplerate / 1e9)))
            if start_bin < 0: start_bin = 0
            end_bin =int(pos + (wr * (self.samplerate / 1e9)))
            if end_bin > data.size: end_bin = data.size 
            
            self.pulse_height.append(self.filtered_signal_1[pos])
            self.pulse_area.append(0)
            for j in range(start_bin, end_bin):
                self.pulse_area[-1] += self.filtered_signal_1[j]

        #return self.pulse_height, self.pulse_area

    def PulseClean(self,data,pulse_pos_clean):
        """
        Generates List of Clean Peak Heights an Peak Areas | Appends
        takes: noise-less signal , List of clean  pulse positions
        returns: List of pulse heights and areas
        """
        wl = 7# in samples, so 2.5 samples == 1 ns, 10 ns = 25 samples 0.4ns/S
        wr = 10
        for i, pos in enumerate(pulse_pos_clean):
            start_bin =int(pos - (wl * (self.samplerate / 1e9)))
            if start_bin < 0: start_bin = 0
            end_bin =int(pos + (wr * (self.samplerate / 1e9)))
            if end_bin > data.size: end_bin = data.size 
            
            self.pulse_height_clean.append(self.filtered_signal_1[pos])
            self.pulse_area_clean.append(0)
            for j in range(start_bin, end_bin):
                self.pulse_area_clean[-1] += self.filtered_signal_1[j] #binvise 'integration' of the pulse area -> in V*bins

        #return self.pulse_height_clean, self.pulse_area_clean




    def Plot(self,pulse_h,pulse_h_c,pulse_a,pulse_a_c,ppc):
        """
        can plot histograms of h,a,clean_h,clean_a
        """
        
        nbins = 5000
        #hist_entr,numbins=np.histogram(pulse_h,bins = nbins)
        #length = len(hist_entr)#-(len(hist_entr)/2) #to cut off the long tail
        #hist_entr= hist_entr[0:length:]
        #numbins= numbins[0:length:]
        fig1 = plt.figure(1)
        ax1 = fig1.add_axes([0.1, 0.1, 0.8, 0.8])
        ax1.set_title('Pulse Height')
        ax1.set_yscale("log")
        ax1.set_xlabel('Pulse Height [V]')
        ax1.set_ylabel('log$_{10}$ (Events) [#]')
        ax1.set_xlim(-0.005,0.1)
        ax1.hist(pulse_h,bins = nbins,color='red',label='Height',histtype='step',log=True)
        ax1.hist(pulse_h_c,bins = nbins,color='green',label='Height Clean',histtype='step',log=True)
        ax1.hist(self.peakless,bins=60,color='black',label='PeaklessZero',histtype='step',log=True)
        #ax1.plot(numbins,hist_entr,'g',label='Histogram Data')
        ax1.hist(self.meanzero,bins=60,label='MeanZero',histtype='step',log=True)
        #ax1.hist(self.filtered_signal_2,bins=nbins,color='black',label='Filtered2Zero',histtype='step',log=True)
        bottom,top=ax1.get_ylim()
        ax1.set_ylim(1,top)
        ax1.legend()

        fig2 = plt.figure(2)
        ax2 = fig2.add_axes([0.1, 0.1, 0.8, 0.8])
        ax2.set_title('Pulse Area')
        ax2.set_yscale("log")
        ax2.set_xlabel('Pulse Area Integral [V $\mu$s]')
        ax2.set_ylabel('log$_{10}$ (Events) [#]')
        ax2.set_xlim(-0.05,1.)
        ax2.hist(pulse_a,bins = nbins,color='red',label='Area',histtype='step',log=True)
        ax2.hist(pulse_a_c,bins = nbins,color='green',label='Area Clean',histtype='step',log=True)
        ax2.hist(self.meanzero,bins=30,label='MeanZero',histtype='step',log=True)
        ax2.hist(self.peakless,bins=30,color='black',label='PeaklessZero',histtype='step',log=True)
        #ax2.hist(self.filtered_signal_2,bins=nbins,color='black',label='Filtered2Zero',histtype='step',log=True)
        bottom,top=ax2.get_ylim()
        ax2.set_ylim(1,top)


        ax2.legend()
        
        fig3 = plt.figure(3)
        rawdat = self.raw_data*1e3 #mV
        ax3 = fig3.add_axes([0.1, 0.1, 0.8, 0.8])
        ax3.set_title('Raw Data')   
        ax3.set_xlabel('Time [$\mu$s]')
        #ax3.set_xlim(0,100)
        ax3.set_ylabel('Voltage [mV]')
        ax3.plot(self.time,rawdat,'grey')
        
        fig4 = plt.figure(4)
        ax4 = fig4.add_axes([0.1, 0.1, 0.8, 0.8])
        ax4.set_xlabel('Pulse Height')
        ax4.set_yscale('log')
        ax4.set_ylabel('log$_{10}$ (Events) [#]')
        ax4.set_title('ZeroPeak Comparison')
        ax4.hist(self.peakless,bins = nbins/50,histtype='step',log=True,label='Peakless Signal Zero-Position')
        ax4.hist(self.filtered_signal_2,bins=nbins,color='black',label='Filtered 2 Zero-Position',histtype='step',log=True)
        ax4.legend()

        fig5 = plt.figure(5)
        ax5 = fig5.add_axes([0.1, 0.1, 0.8, 0.8])
        ax5.set_xlabel('Time [$\mu$s]')
        ax5.set_ylabel('Voltage [V]')
        ax5.set_xlim(0,1)
        ax5.set_title('Raw Data, Filtered 2')
        ax5.plot(self.time,self.raw_data,'grey',label='Raw Data')
        #plt.plot(self.time,self.filtered_signal_1,'green',label='1 Filtered')
        ax5.plot(self.time,self.filtered_signal_2,'red',label='Filtered 2')
        ax5.legend()
        print "SAVING"
        fig1.savefig(self.data_file_path+'PulseHeight.pdf',format='pdf', bbox_inches='tight')
        fig2.savefig(self.data_file_path+'PulseArea.pdf',format='pdf', bbox_inches='tight')
        fig3.savefig(self.data_file_path+'RawData.pdf',format='pdf', bbox_inches='tight')
        fig4.savefig(self.data_file_path+'ZeroPeak.pdf',format='pdf', bbox_inches='tight')
        fig5.savefig(self.data_file_path+'RawFiltered2.pdf',format='pdf', bbox_inches='tight')
        plt.show()
        fig1.clf()
        fig2.clf()
        fig3.clf()
        fig4.clf()
        fig5.clf()

    def GetZeroPeak(self):
        self.meanzero.append(np.mean(self.peakless))#meanzero-list of the segments
        #append whole signal fit with gaussian then compare with 1pe fitted signal
        self.sigmazero.append(np.std(self.peakless))

    def Reduce_Data(self,rmsval):
        segments = self.GetSegmentCount()
        calcseg = int(float(self.calcsegmentspro)/100*segments)
        print 'Calculating ',calcseg,' Segments , ',self.calcsegmentspro,' % of total count'
        for ite in range(0,calcseg): 
            data = self.Load_Data(ite)
            signal1 = self.Slownoise_Smooth(data,ite)
            signal2 = self.Fastnoise_Smooth(signal1)
            pulse_pos = self.FindPeaks(signal2,ite,rmsval)
            pulse_pos_clean = self.PulseCleanUp(pulse_pos)
            self.Pulse(data,pulse_pos)
            self.PulseClean(data,pulse_pos_clean)
            self.GetZeroPeak()
        self.Plot(self.pulse_height,self.pulse_height_clean,self.pulse_area,self.pulse_area_clean,pulse_pos_clean)

        fig7 = plt.figure(7)
        ax72 = fig7.add_subplot(211)
        ax73 = fig7.add_subplot(212)
        
        ax72.set_title('Peak Pos vs Peak Pos Clean')
        ax72.set_xlabel('Time [$\mu$s]')
        ax72.set_ylabel('Voltage [V]')
        ax73.set_xlabel('Time [$\mu$s]')
        ax73.set_ylabel('Voltage [V]')
        leng = np.arange(0,len(self.time))
        pulsecount = str(len(pulse_pos))+' Pulses'
        ax72.text(0.9, 0.8,pulsecount, horizontalalignment='center',
        verticalalignment='center',transform=ax72.transAxes)
        ax72.plot(self.time,signal2,'b',self.time[pulse_pos],signal2[pulse_pos],'rD')
        #ax72.plot(self.time,signal1,'b',self.time[leng],signal1[leng],'bD',self.time[pulse_pos],signal1[pulse_pos],'gD')
        #ax72.set_xlim(0.31788,0.35)
        #ax72.set_ylim(-0.002,0.009)
        pulsecount = str(len(pulse_pos_clean))+' Pulses'
        ax73.text(0.9, -0.4,pulsecount, horizontalalignment='center',
        verticalalignment='center',transform=ax72.transAxes)
        ax73.plot(self.time,signal2,'b',self.time[pulse_pos_clean],signal2[pulse_pos_clean],'rD')
        #ax73.set_xlim(0.07,0.09)
        fig7.savefig(self.data_file_path+'PulsePosComp.pdf',format='pdf', bbox_inches='tight')
        plt.show()
        fig7.clf()
        

def GetName(destT,destV,Date):
    # need to implement date of experiment into data file path
    path = '/home/gebhardt/00_SiPM_MPIK/scripts/Data/Date'+str(Date)+'/'+str(destT)+'deg/'+str(destV)+'V/'
    name = 'HAM_T' + str(destT) + '_Vb' + str(destV) + '.trc'
    compl_name = os.path.join(path,name)
    return compl_name

def GetSavePath(Date):
    savepath = '/home/gebhardt/00_SiPM_MPIK/scripts/Data/Date'+str(Date)+'/'
    return savepath

#@profile       
def main():
    #rmsvallist = [3,2.8,1.6]
    
    Date ='0510162'
    T=[0.]
    Vb=[61.]#,29.0,33.]
    savepath = GetSavePath(Date)
    iterT = 0
    for destT in T:
        print 'Commence Data Reduction T= ',destT
        for v in Vb:
            rmsval = 0.001
            compl_name = GetName(destT,v,Date)
            print 'Saving Plots of ',compl_name 
            WfRed(compl_name,savepath).Reduce_Data(rmsval)
        iterT+=1


    
if __name__ == '__main__':
    main()





