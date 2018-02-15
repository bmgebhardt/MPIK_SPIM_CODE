import sys
sys.path.append('/home/gebhardt/00_SiPM_MPIK/scripts')
import LecroyBinary
import numpy as np
from scipy import signal as sig
from detect_peaks import detect_peaks
from memory_profiler import profile
import argparse
import matplotlib.pyplot as plt
import os.path
import numpy.ma as ma

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
        self.calcsegments = 2

    def ReportProgress(self,ite):
        if (ite % 10) == 0:
            print ite,' % done'


    def GetSegmentCount(self):

        self.ranger = LecroyBinary.LecroyBinaryWaveform(self.data_file_path).SubArray()
        return self.ranger

    def Load_Data(self,it1):
        """
        Loopable through counter in function call
        raw_data from filecontent
        takes: complete self.data_file_path
        returns:  Raw Data
        """
        # return all segments 
        self.raw_data,self.time=LecroyBinary.LecroyBinaryWaveform(self.data_file_path).RealGetNextWaveArrayDataAndTime(it1)# in V and s
        #self.raw_data = self.raw_data*1e3 #in mV
        self.time = (self.time-self.time[0])*1e6 #starts at 0, in us
        return self.raw_data

    
    def RemovePeaksBasedOnRMS(self,wave, nSigma):
        """
        sets Datapoints above a certain rms*nSigma to 0
        original
        """
        rms = np.std(wave)
        mean = np.mean(wave)
        if abs(mean)>1e-2: print "Warning, mean not near 0!!"
        
        cut = nSigma * rms
        print "RMS = ", rms, " Cutting at: ", cut
        
        indices = np.where(np.abs(wave)<cut)[0] 
        # search indices, where wave is below upper limit
        wave_cut = np.zeros(wave.size) 
        # create np array of zeros of aquivalent size
        wave_cut[indices] = wave[indices] 
        # fill only those indices, where wave is below upper limit -> peakless signal

        return wave_cut,cut


    def RemovePeaksBasedOnRMS2(self,wave, nSigma):
        """
        Calculates rms based on negative waveform
        wave_cut is wave<0        
        """   
        indices = np.where(wave<0)[0] 
        # search indices, where wave is below upper limit 0
        wave_cut = np.zeros(wave.size) 
        #print indices
        #print wave[0:15:]
        #print wave_cut[0:15:]
        # create np array of zeros of aquivalent size
        wave_cut[indices] = wave[indices] 
        # fill only those indices, where wave is below upper limit -> peakless signal
        rms = np.std(wave_cut)
        #print rms
        cut = nSigma * rms
        #print cut
        return wave_cut,cut


    def RemovePeaksBasedOnRMS3(self,wave, nSigma):
        """
        gets negative of waveform and calculates rms
        uses rms to calculate peakless waveform        
        """   
        indices = np.where(wave<0)[0] 
        # search indices, where wave is below upper limit 0
        wave_cut_rms = np.zeros(wave.size) 
        #print indices
        #print wave[0:15:]
        #print wave_cut[0:15:]
        # create np array of zeros of aquivalent size
        wave_cut_rms[indices] = wave[indices] 
        # fill only those indices, where wave is below upper limit -> peakless signal

        rms = np.std(wave_cut_rms)
        #combination of 1 and 2
        indices = np.where(np.abs(wave)<rms*2)[0] 
        wave_cut = np.zeros(wave.size) 
        wave_cut[indices] = wave[indices] 


        #print rms
        cut = nSigma * rms
        #print cut
        return wave_cut,cut



    def Slownoise_Smooth(self,data,ite):
        """
        generates smoothed signal from peak-less signal and raw_data
        takes: peak-less signal , raw_data
        returns: smoothed signal 1
        """


        
        rmsn = 2
        slownoise,level = self.RemovePeaksBasedOnRMS3(data,rmsn)
        
        nsamples = 700#determines width of smoothing gaussian (/5) = sigma of gauss 
        window = sig.general_gaussian(nsamples, p=1.0, sig=nsamples/5)
        slownoise_smooth = sig.fftconvolve(slownoise, window, "same")
        #slownoise_smooth = (np.average(slownoise) / np.average(slownoise_smooth)) * slownoise_smooth
        slownoise_smooth = (np.average(slownoise) /np.average(slownoise_smooth)) * slownoise_smooth
        
        self.filtered_signal_1 = data - slownoise_smooth
        self.peakless = slownoise 
        
        if ite ==1:
            fig8 = plt.figure(8)
            ax8 = fig8.add_axes([0.1, 0.1, 0.8, 0.8])
            ax8.set_title('Filter Levels')
            ax8.set_xlabel('Time [us]')
            ax8.set_ylabel('Voltage [V]')
            #ax8.set_xlim(0,100)
            rawdat = ax8.plot(self.time,data,'grey',label='Raw Data')
            slown = ax8.plot(self.time,slownoise,'red',label='slow Noise')
            plt.plot((np.amin(self.time),np.amax(self.time)), (level*rmsn,level*rmsn), 'b--',label='Cut level')
            ax8.set_title('Smoothed slow Noise')
            slowns = ax8.plot(self.time,slownoise_smooth,'blue',label='Smoothed slow Noise')
            ax8.legend()
            #plt.show()
            fig8.savefig(self.data_file_path+'FilterLevels.ps',format='ps',bbox_inches='tight')
            #ax8.legend_.remove()
            #fig8.clf()
        
        self.ReportProgress(ite)
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


    def FindPeaks(self,filtered_signal_2,ite):
        """
        searches for the peak position in the smoothed noise-less signal
        takes: smoothed noise-less signal
        returns: List of Peak Indeces
        """
        #rms = np.std(self.peakless)
        #print rms
        #print np.std(filtered_signal_2)
        #pulse_pos = detect_peaks(filtered_signal_2, rms*1, mpd=10,show=False)

        
        indices = np.where(self.peakless<0)[0] 
        # search indices, where wave is below upper limit
        peakless_cut = np.zeros(len(self.peakless)) 
        # create np array of zeros of aquivalent size
        peakless_cut[indices] = self.peakless[indices] 
        # fill only those indices, where wave is below upper limit -> peakless signal

        rmsn = 3
        rms = np.std(peakless_cut)
        #print rms
        #print np.std(filtered_signal_2)
        pulse_pos = detect_peaks(filtered_signal_2, rms*rmsn, mpd=30,show=False)


        if ite ==1:
            plt.figure()
            plt.title('signal 2 peak detect')
            plt.xlabel('Time [s]')
            plt.ylabel('Voltage [V]')
            #plt.xlim([0,10])
            plt.plot(self.time,filtered_signal_2,'red',label='Filtered2')
            #plt.plot(self.time,peakless_cut,'black',label='Peakless')
            plt.plot((np.amin(self.time),np.amax(self.time)), (rms*rmsn,rms*rmsn), 'b--',label='peak detect level')
            plt.plot((np.amin(self.time),np.amax(self.time)), (-rms*rmsn,-rms*rmsn), 'w--',label='peak detect level')
            plt.legend()
            plt.show()
        
        return pulse_pos

    def PulseCleanUp(self,pulse_pos):
        """
        Generates List of Peaks without immidiate neighbor
        -> Cleaned up Peak positions
        takes: List of Pulse Positions
        returns: List of cleaned up Pulse positions
        """       
        wpeak = 300
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
        wl = 15 # in samples, so 2.5 samples == 1 ns, 10 ns = 25 samples | 2.5GS/s = 0.4ns/S
        wr = 20
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
        wl = 15# in samples, so 2.5 samples == 1 ns, 10 ns = 25 samples 0.4ns/S
        wr = 20
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
        
        nbins = 2000
        hist_entr,numbins=np.histogram(pulse_h,bins = nbins)
        length = len(hist_entr)-(len(hist_entr)/2) #to cut off the long tail
        hist_entr= hist_entr[0:length:]
        numbins= numbins[0:length:]
        fig1 = plt.figure(1)
        ax1 = fig1.add_axes([0.1, 0.1, 0.8, 0.8])
        ax1.set_title('Pulse Height')
        ax1.set_yscale("log")
        ax1.set_xlabel('Pulse Height')
        ax1.set_ylabel('log$_{10}$ (Events) [#]')
        #ax1.hist(pulse_h,bins = nbins,color='red',label='Height',histtype='step',log=True)
        #ax1.hist(pulse_h_c,bins = nbins,color='green',label='Height Clean',histtype='step',log=True)
        #ax1.hist(self.peakless,bins=nbins,color='blue',label='PeaklessZero',histtype='step',log=True)
        ax1.plot(numbins,hist_entr,'g',label='Histogram Data')
        ax1.hist(self.filtered_signal_2,bins=nbins,color='black',label='Filtered2Zero',histtype='step',log=True)


        ax1.legend()

        fig2 = plt.figure(2)
        ax2 = fig2.add_axes([0.1, 0.1, 0.8, 0.8])
        ax2.set_title('Pulse Area')
        ax2.set_yscale("log")
        ax2.set_xlabel('Pulse Area Integral [V s]')
        ax2.set_ylabel('log$_{10}$ (Events) [#]')
        ax2.set_xlim(-1.,3.)
        ax2.hist(pulse_a,bins = nbins,color='red',label='Area',histtype='step',log=True)
        ax2.hist(pulse_a_c,bins = nbins,color='green',label='Area Clean',histtype='step',log=True)
        #ax2.hist(self.peakless,bins=nbins,color='blue',label='PeaklessZero',histtype='step',log=True)
        ax2.hist(self.filtered_signal_2,bins=nbins,color='black',label='Filtered2Zero',histtype='step',log=True)




        ax2.legend()
        
        fig3 = plt.figure(3)
        rawdat = self.raw_data*1e3 #mV
        ax3 = fig3.add_axes([0.1, 0.1, 0.8, 0.8])
        ax3.set_title('Raw Data')   
        ax3.set_xlabel('Time [us]')
        #ax3.set_xlim(0,100)
        ax3.set_ylabel('Voltage [mV]')
        ax3.plot(self.time,rawdat,'grey')
        
        fig4 = plt.figure(4)
        ax4 = fig4.add_axes([0.1, 0.1, 0.8, 0.8])
        ax4.set_xlabel('Pulse Height')
        ax4.set_yscale('log')
        ax4.set_ylabel('log$_{10}$ (Events) [#]')
        ax4.set_title('ZeroPeak')
        ax4.hist(self.peakless,bins = nbins,histtype='step',log=True)
        ax2.hist(self.filtered_signal_2,bins=nbins,color='black',label='Filtered2Zero',histtype='step',log=True)
        
        fig5 = plt.figure(5)
        ax5 = fig5.add_axes([0.1, 0.1, 0.8, 0.8])
        ax5.set_xlabel('Time [us]')
        ax5.set_ylabel('Voltage [V]')
        #ax5.set_xlim(0,100)
        ax5.set_title('Raw, 1Filtered, 2Filtered')
        ax5.plot(self.time,self.raw_data,'grey',label='Raw Data')
        #plt.plot(self.time,self.filtered_signal_1,'green',label='1 Filtered')
        ax5.plot(self.time,self.filtered_signal_2,'red',label='2 Filtered')
        ax5.legend()
        print "SAVING"
        fig1.savefig(self.data_file_path+'PulseHeight.ps',format='ps', bbox_inches='tight')
        fig2.savefig(self.data_file_path+'PulseArea.ps',format='ps', bbox_inches='tight')
        fig3.savefig(self.data_file_path+'RawData.ps',format='ps', bbox_inches='tight')
        fig4.savefig(self.data_file_path+'ZeroPeak.ps',format='ps', bbox_inches='tight')
        fig5.savefig(self.data_file_path+'Raw1Filtered2Filtered.ps',format='ps', bbox_inches='tight')
        plt.show()
        fig1.clf()
        fig2.clf()
        fig3.clf()
        #fig4.clf()
        fig5.clf()

    def GetZeroPeak(self):
        self.meanzero.append(np.mean(self.peakless))
        self.sigmazero.append(np.std(self.peakless))

    def Reduce_Data(self):
        segments = self.GetSegmentCount()
        for ite in range(0,self.calcsegments): #needed just for the last plot option
            data = self.Load_Data(ite)
            signal1 = self.Slownoise_Smooth(data,ite)
            signal2 = self.Fastnoise_Smooth(signal1)
            pulse_pos = self.FindPeaks(signal2,ite)
            pulse_pos_clean = self.PulseCleanUp(pulse_pos)
            self.Pulse(data,pulse_pos)
            self.PulseClean(data,pulse_pos_clean)
            self.GetZeroPeak()
        self.Plot(self.pulse_height,self.pulse_height_clean,self.pulse_area,self.pulse_area_clean,pulse_pos_clean)

        '''
        plt.figure(7)
        plt.title('Peak Pos vs Peak Pos Clean')
        plt.subplot(211)
        plt.xlabel('Time [$\mu$s]')
        plt.ylabel('Voltage [V]')
        plt.plot(self.time,signal2,'b',self.time[pulse_pos],signal2[pulse_pos],'rD')
        #plt.xlim(0.0,10.)
        plt.subplot(212)
        plt.plot(self.time,signal2,'b',self.time[pulse_pos_clean],signal2[pulse_pos_clean],'rD')
        #plt.xlim(0.0,10.0)
        plt.show()
        plt.figure(7).savefig(self.data_file_path+'PulsePosComp.ps',format='ps', bbox_inches='tight')
        plt.figure(7).clf()
        '''


        fig7 = plt.figure(7)
        ax72 = fig7.add_subplot(211)
        ax73 = fig7.add_subplot(212)
        
        ax72.set_title('Peak Pos vs Peak Pos Clean')
        ax72.set_xlabel('Time [$\mu$s]')
        ax72.set_ylabel('Voltage [V]')
        ax73.set_xlabel('Time [$\mu$s]')
        ax73.set_ylabel('Voltage [V]')
        leng = np.arange(0,len(self.time))
        ax72.plot(self.time,signal2,'b',self.time[pulse_pos],signal2[pulse_pos],'rD')
        #ax72.plot(self.time,signal1,'b',self.time[leng],signal1[leng],'bD',self.time[pulse_pos],signal1[pulse_pos],'gD')
        #ax72.set_xlim(0.31788,0.35)
        #ax72.set_ylim(-0.002,0.009)
        ax73.plot(self.time,signal2,'b',self.time[pulse_pos_clean],signal2[pulse_pos_clean],'rD')
        #ax73.set_xlim(0.07,0.09)
        fig7.savefig(self.data_file_path+'PulsePosComp.ps',format='ps', bbox_inches='tight')
        plt.show()
        fig7.clf()
        

'''
parser = argparse.ArgumentParser(description='Decide:')
parser.add_argument('-t','--temp',type=float,help='-tx or --temp x | save plot of one segment at temp, default 25.0')
parser.add_argument('-v','--volt',type=float,help='-vx or --volt x | save plot of one segment at volt, default 28.0')
args = parser.parse_args()
'''

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
    Date ='2909161'
    T=[20.]
    Vb=[60.]#,29.0,33.]
    savepath = GetSavePath(Date)
    for destT in T:
        print 'Commence Data Reduction T= ',destT
        for v in Vb:
            compl_name = GetName(destT,v,Date)
            print 'Saving Plots of ',compl_name 
            WfRed(compl_name,savepath).Reduce_Data()



    
if __name__ == '__main__':
    main()





