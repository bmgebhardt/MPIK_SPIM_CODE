
# TODO:
# list -> np.array (faster?)

#import matplotlib #for remote usage
#matplotlib.use('Agg') #for remote usage
#matplotlib.use('Agg') #for remote usage

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

fromregr = True



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
        self.calcsegmentspro = 3

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
        #print self.time[-1]-self.time[0]
        return self.raw_data

    def RemovePeaksBasedOnRMS3(self,wave):
        """
        gets negative of waveform and calculates rms
        uses rms to calculate peakless waveform        
        """   

        indices_negative2 = np.where(wave<0)[0] 
        wave_cut_negative2 = np.zeros(wave.size) 
        wave_cut_negative2[indices_negative2] = wave[indices_negative2] 
        mean = np.mean(wave_cut_negative2)

        indices_negative = np.where(wave<mean)[0] 
        # search indices, where wave is below upper limit 0
        wave_cut_negative = np.zeros(wave.size) 
        # create np array of zeros of aquivalent size
        wave_cut_negative[indices_negative] = wave[indices_negative] 
        # fill only those indices, where wave is below upper limit -> peakless signal

        rms = np.std(wave_cut_negative) # get rms of negative signal

        #indices = np.where(np.abs(wave)<rms*2)[0] # wave between rms*2, both sides
        indices = np.where(wave<rms)[0] # wave between rms*2, both sides # changed to cutting only top peaks, due to pedestal sub bug
              
        wave_cut = np.zeros(wave.size) 
        wave_cut[indices] = wave[indices] #fill wave 

        cut = rms #cut-level
        
        fig2, (ax1,ax2) = plt.subplots(1,2, sharey=False)
        rawdat = self.raw_data*1e3 #mV
        ax1.set_title('Remove Peaks')   
        ax1.set_xlabel('Time [$\mu$s]')
        #ax1.set_xlim(0,100)
        ax1.set_ylabel('Voltage [mV]')        
        ax1.plot(self.time,wave,'grey',label='Raw Data')
        ax1.plot(self.time,wave_cut,'blue',label='Slownoise')
        ax1.plot((np.amin(self.time),np.amax(self.time)), (cut,cut), 'w--',label='Cut level')

        ax2.set_title('Zoom') 
        ax2.set_xlabel('Time [$\mu$s]')
        #ax2.set_xlim(45,46)
        #ax2.set_ylim(-0.01,0.04)
        ax2.set_ylabel('Voltage [mV]')        
        ax2.plot(self.time,wave,'grey',label='Raw Data')
        ax2.plot(self.time,wave_cut,'blue',label='Slownoise')
        ax2.plot((np.amin(self.time),np.amax(self.time)), (cut,cut), 'k--',label='Cut level')
        ax2.legend()
        fig2.savefig(self.data_file_path+'RemovePeaksBasedOnRMS3Zoom.pdf',format='pdf', bbox_inches='tight')

        plt.show()
        
        return wave_cut,cut



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

        nsamples = 500#determines width of smoothing gaussian (/5) = sigma of gauss 
        window = sig.general_gaussian(nsamples, p=1.0, sig=nsamples/5)
        slownoise_smooth = sig.fftconvolve(self.peakless, window, "same")
        slownoise_smooth = (np.average(self.peakless) /np.average(slownoise_smooth)) * slownoise_smooth
        
        self.filtered_signal_1 = data_at_0 - slownoise_smooth

        
        if ite ==0:


            fig3, (ax1,ax2) = plt.subplots(1,2, sharey=False)
            rawdat = self.raw_data*1e3 #mV
            ax1.set_title('Slow Noise')   
            ax1.set_xlabel('Time [$\mu$s]')
            ax1.set_xlim(0,100)
            ax1.set_ylabel('Voltage [mV]')        
            ax1.plot(self.time,data,'b',label='Raw Data')
            ax1.plot(self.time,slownoise_smooth,'w',label='Slownoise Smooth')
            

            ax2.set_title('Zoom') 
            ax2.set_xlabel('Time [$\mu$s]')
            ax2.set_xlim(45,46)
            ax2.set_ylim(-0.01,0.04)
            ax2.set_ylabel('Voltage [mV]')        
            ax2.plot(self.time,data,'b',label='Raw Data')
            ax2.plot(self.time,slownoise_smooth,'k',label='Slownoise Smooth')            
            ax2.legend()
            fig3.savefig(self.data_file_path+'SlowNoiseSmoothZoom.pdf',format='pdf', bbox_inches='tight')
            #plt.show()
        
        self.ReportProgress(ite,self.ranger)
        return self.filtered_signal_1


    def Fastnoise_Smooth(self,filtered_signal_1):
        """
        smoothes noise-less signal again
        takes: noise-less signal
        returns: smoothed noise-less signal
       
        peak detection more reliable
        """
        nsamples =10#20
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


        '''
        #additional RMS Calc based on band between RMS from negative peakless
        #still testing
        rms_negative = np.std(peakless_cut_negative)
        indices_rmsband = np.where(np.abs(self.peakless)<rms_negative)[0]  #fs2?
        peakless_cut_rmsband = np.zeros(len(self.peakless)) 
        peakless_cut_rmsband[indices_rmsband] = self.peakless[indices_rmsband] 
        rms = np.std(peakless_cut_rmsband)
        ''' 
        rms = 1
        rmsn = (rmsn/2/6) *0.9   #6 for SensL Int Window , 17 for HAM Int Window
        #rms = np.std(peakless_cut_negative) # generate rms from peakless signal(noiselevel)
        #print rms
        #print np.std(filtered_signal_2)
        #pulse_pos = detect_peaks(filtered_signal_2, rms*rmsn, mpd=30,show=False)
        pulse_pos = detect_peaks(filtered_signal_2, rmsn, mpd=10,show=False)


        if ite ==0:



            fig6 , (ax1,ax2) = plt.subplots(1,2, sharey=False)
            ax1.set_title('Filtered 2 Peak Detection')
            ax1.set_xlabel('Time [$\mu$s]')
            ax1.set_ylabel('Voltage [V]')
            ax2.set_title('Zoom')
            #ax1.set_xlim(0,100)
            #ax2.set_xlim(45,46)
            #ax2.set_ylim(-0.01,0.04)
            ax1.plot(self.time,filtered_signal_2,'b',label='Filtered 2')
            ax1.plot((np.amin(self.time),np.amax(self.time)), (rms*rmsn,rms*rmsn),'w--',(np.amin(self.time),np.amax(self.time)), (-rms*rmsn,-rms*rmsn), 'k--',label='peak detect level')
            #ax1.plot((np.amin(self.time),np.amax(self.time)), (-rms*rmsn,-rms*rmsn), 'b--',label='peak detect level')
            ax1.plot((np.amin(self.time),np.amax(self.time)), (np.mean(self.peakless),np.mean(self.peakless)), 'g--',label='Mean') #rmsn from RemovePeaksBasedOnRMS3 = 2
            
            ax2.plot(self.time,filtered_signal_2,'blue',label='Filtered 2')
            ax2.plot((np.amin(self.time),np.amax(self.time)), (rms*rmsn,rms*rmsn), 'k--',(np.amin(self.time),np.amax(self.time)), (-rms*rmsn,-rms*rmsn), 'k--',label='peak detect level')
            #ax2.plot((np.amin(self.time),np.amax(self.time)), (-rms*rmsn,-rms*rmsn), 'b--',label='peak detect level')
            ax2.plot((np.amin(self.time),np.amax(self.time)), (np.mean(self.peakless),np.mean(self.peakless)), 'g--',label='Mean') #rmsn from RemovePeaksBasedOnRMS3 = 2
           
            plt.legend()
            fig6.savefig(self.data_file_path+'PeakDetectLevelZoom.pdf',format='pdf', bbox_inches='tight')
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
        wpeak = 50
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

        #Ham
        #7   
        #10
        #SensL
        #2
        #4

        wl = 2 # in samples, so 2.5 samples == 1 ns, 10 ns = 25 samples | 2.5GS/s = 0.4ns/S
        wr = 4
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
        #Ham
        #7   
        #10
        #SensL
        #2
        #4



        wl = 2# in samples, so 2.5 samples == 1 ns, 10 ns = 25 samples 0.4ns/S
        wr = 4
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
        

        fig1, (ax1,ax2) = plt.subplots(1,2, sharey=False)
        rawdat = self.raw_data#*1e3 #mV
        ax1.set_title('Raw Data')   
        ax1.set_xlabel('Time [$\mu$s]')
        #ax1.set_xlim(0,100)
        ax1.set_ylabel('Voltage [V]')
        ax1.plot(self.time,rawdat,'blue')
        ax2.set_title('Zoom')  
        ax2.set_xlabel('Time [$\mu$s]')
        #ax2.set_xlim(45,46)
        #ax2.set_ylim(-0.01,0.04)
        ax2.plot(self.time,rawdat,'blue')
        figure = plt.gcf() # get current figure
        figure.set_size_inches(7, 7)



        

        fig4 , (ax1,ax2) = plt.subplots(1,2, sharey=False)
        ax1.set_xlabel('Time [$\mu$s]')
        ax1.set_ylabel('Voltage [V]')
        #ax1.set_xlim(0,100)
        ax1.set_title('Raw Data, Filtered 1')
        ax2.set_title('Zoom')
        ax1.plot(self.time,self.raw_data,'grey',label='Raw Data')
        #plt.plot(self.time,self.filtered_signal_1,'green',label='1 Filtered')
        ax1.plot(self.time,self.filtered_signal_1,'b',label='Filtered 1')

        ax2.set_xlabel('Time [$\mu$s]')
        #ax2.set_xlim(45,46)
        #ax2.set_ylim(-0.01,0.04)
        ax2.plot(self.time,self.raw_data,'grey',label='Raw Data')
        #plt.plot(self.time,self.filtered_signal_1,'green',label='1 Filtered')
        ax2.plot(self.time,self.filtered_signal_1,'b',label='Filtered 1')        
        ax2.legend()
        #plt.show()
        

        fig5, (ax1,ax2) = plt.subplots(1,2, sharey=False)
        ax1.set_xlabel('Time [$\mu$s]')
        ax1.set_ylabel('Voltage [V]')
        #ax1.set_xlim(0,100)
        ax1.set_title('Filtered 1, Filtered 2')
        ax2.set_title('Zoom')
        ax1.plot(self.time,self.filtered_signal_1,'grey',label='Filtered 1')
        #plt.plot(self.time,self.filtered_signal_1,'green',label='1 Filtered')
        ax1.plot(self.time,self.filtered_signal_2,'b',label='Filtered 2')

        ax2.set_xlabel('Time [$\mu$s]')
        #ax2.set_xlim(45,46)
        #ax2.set_ylim(-0.01,0.04)
        ax2.plot(self.time,self.filtered_signal_1,'grey',label='Filtered 1')
        #plt.plot(self.time,self.filtered_signal_1,'green',label='1 Filtered')
        ax2.plot(self.time,self.filtered_signal_2,'b',label='Filtered 2')        
        ax2.legend()
        #plt.show()






        print "SAVING"

        fig1.savefig(self.data_file_path+'RawDataZoom.pdf',format='pdf', bbox_inches='tight')
        fig4.savefig(self.data_file_path+'Filtered1Zoom.pdf',format='pdf', bbox_inches='tight')
        fig5.savefig(self.data_file_path+'Filtered2Zoom.pdf',format='pdf', bbox_inches='tight')
        #plt.show()
        #fig1.clf()
        #fig2.clf()
        #fig3.clf()
        #fig4.clf()
        #fig5.clf()

    def GetZeroPeak(self):
        self.meanzero.append(np.mean(self.peakless))#meanzero-list of the segments
        #append whole signal fit with gaussian then compare with 1pe fitted signal
        self.sigmazero.append(np.std(self.peakless))

    def Reduce_Data(self,rmsval):
        segments = self.GetSegmentCount()
        calcseg = int(float(self.calcsegmentspro)/100*segments)-1
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

        fig7 , (ax1,ax2) = plt.subplots(1,2, sharey=False)
                
        ax1.set_title('Peak Pos Clean')
        ax1.set_xlabel('Time [$\mu$s]')
        ax1.set_ylabel('Voltage [V]')
        ax2.set_xlabel('Time [$\mu$s]')
        ax1.set_xlim(0,100)
        ax2.set_xlim(45,46)
        ax2.set_ylim(-0.01,0.04)

        pulsecount = str(len(pulse_pos))+' Pulses'
        ax1.plot(self.time,signal2,'b')
        ax1.plot(self.time[pulse_pos_clean],signal2[pulse_pos_clean],'rD',label=pulsecount)
        ax2.set_title('Zoom')
        pulsecount = str(len(pulse_pos_clean))+' Clean Pulses'
        ax2.plot(self.time,signal2,'b',self.time[pulse_pos_clean],signal2[pulse_pos_clean],'rD',label=pulsecount)
        ax2.legend()
        ax1.legend()
        fig7.savefig(self.data_file_path+'PulsePosCompZoom.pdf',format='pdf', bbox_inches='tight')

        #plt.show()
        #fig7.clf()

        #fig8 , (ax1,ax2) = plt.subplots(1,2, sharey=False)
        fig8 = plt.figure(8)
        ax2 = fig8.add_axes([0.1, 0.1, 0.8, 0.8])




        ax2.set_title('Integration Window')
        #ax1.set_xlabel('Time [$\mu$s]')
        ax2.set_ylabel('Voltage [V]')
        ax2.set_xlabel('Time [$\mu$s]')
        #ax1.set_xlim(45,46)
        ax2.set_xlim(51.35,51.55)
        #ax1.set_ylim(-0.01,0.04)
        ax2.set_ylim(-0.0004,0.004)

        #ax1.plot(self.time,signal2,'b',label='Filtered 2')
        #ax1.plot(self.time[pulse_pos_clean],signal2[pulse_pos_clean],'rD',label='Pulses')
                
        ax2.plot(self.time,signal2,'b')
        
        x1=51.4624-(2*0.4)/1000
        x2=51.4624+(4*0.4)/1000
        y1=0.001107605
        y2=0.001107605
        ax2.set_title('Zoom')
        ax2.plot(self.time[pulse_pos_clean],signal2[pulse_pos_clean],'rD')
        ax2.plot((x1,x2),(y1,y2),'g-',linewidth=5,label='3ns FWHM')

        #ax2.plot((x1,x2),(y1,y2),'g-',(x1,x2),(0,0),'g-',(x1,x1),(0,y2),'g-')
        #ax2.plot((x2,x2),(0,y2),'g-',label='Integration Window')
        ax2.legend()
        ax2.get_xaxis().get_major_formatter().set_useOffset(False)
        figure = plt.gcf() # get current figure
        figure.set_size_inches(7, 7)

        #ax1.legend()
        #fig8.savefig(self.data_file_path+'IntegrationWindowZoom.pdf',format='pdf', bbox_inches='tight')
        fig8.savefig(self.data_file_path+'PulseShape.pdf',format='pdf', bbox_inches='tight')

        plt.show()


        

def GetName(destT,destV,Date):
    # need to implement date of experiment into data file path
    path = '/home/gebhardt/00_SiPM_MPIK/scripts/Data/Date'+str(Date)+'/'+str(destT)+'deg/'+str(destV)+'V/'
    name = 'SensL_T' + str(destT) + '_Vb' + str(destV) + '.trc'
    compl_name = os.path.join(path,name)
    return compl_name

def GetSavePath(Date):
    savepath = '/home/gebhardt/00_SiPM_MPIK/scripts/Data/Date'+str(Date)+'/'
    return savepath

#@profile       
def main():
    rmsvallist = [3,2.8,1.6]
    
    Date ='060716'
    T=[25.]
    Vb=[29.]#,29.0,33.]
    savepath = GetSavePath(Date)
    iterT = 0
    for destT in T:
        print 'Commence Data Reduction T= ',destT
        for v in Vb:
            #if v < 69.:rmsval = rmsvallist[0]
            #else: 
                #if v < 69.7: rmsval = rmsvallist[1]
                #else:rmsval = rmsvallist[1]
            if fromregr:
                regrfile = np.load(savepath+'AreaRelGainRegrLineData.npy')
                slope = regrfile[1][iterT]
                print slope
                inter = regrfile[2][iterT]
                print inter
                rmsval = slope*v+inter
                print rmsval
                #rmsval from regr line
                # to do manually again set rmsval yourself
            else:rmsval = 0.003 #in V
            compl_name = GetName(destT,v,Date)
            print 'Saving Plots of ',compl_name 
            WfRed(compl_name,savepath).Reduce_Data(rmsval)
        iterT+=1


    
if __name__ == '__main__':
    main()





