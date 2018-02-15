#
#     for parameter reference, not usable for SensL Data analysis
#
#



import LecroyBinary
import numpy as np
from scipy import signal as sig
from detect_peaks import detect_peaks
from memory_profiler import profile
import sys
import argparse
import matplotlib.pyplot as plt
import os.path



def log_10_product(x, pos):
    """The two args are the value and tick position.
    Label ticks with the product of the exponentiation"""
    return '{:.3e}'.format(x)



class WfRed(object):

    def __init__(self,data_file_path,savepath):
        self.raw_data = np.zeros(1)
        self.vB = 0.0
        self.T = 0.0
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

    def GetSegmentCount(self):

        self.ranger = LecroyBinary.LecroyBinaryWaveform(self.data_file_path).SubArray()
        return self.ranger

    def LoadPresets(self):
        """
        calcutales measured time, reads vB from filename, raw_data from filecontent
        takes: complete self.data_file_path
        returns: Bias Voltage, Measured Time
        """
        #calculate time from wavearray size
        samplerate = float(2.5e9)
        #ranger = LecroyBinary.LecroyBinaryWaveform(self.data_file_path).SubArray()
        length = LecroyBinary.LecroyBinaryWaveform(self.data_file_path).WaveArray()
   
        time = length/samplerate
        
        #find vB in filename, extract vB
        ind = self.data_file_path.index('Vb')
        volt = self.data_file_path[ind+2:ind+6]
        self.vB = float(volt)
        try:
            #find deg in filename, extract T
            ind = self.data_file_path.index('deg')
            temp = self.data_file_path[ind-3:ind]
            self.T = float(temp)
        except ValueError: sys.exc_clear()#print 'Temp is double digit' 
        try:
            #find deg in filename, extract T
            ind = self.data_file_path.index('deg')
            temp = self.data_file_path[ind-4:ind]
            self.T = float(temp)
        except ValueError: sys.exc_clear()#print 'Temp is single digit' 






        return time,self.vB,self.T
        

    def Load_Data(self,it1):
        """
        Loopable through counter in function call
        raw_data from filecontent
        takes: complete self.data_file_path
        returns:  Raw Data
        """
        # return all segments 
        self.raw_data,self.time=LecroyBinary.LecroyBinaryWaveform(self.data_file_path).RealGetNextWaveArrayDataAndTime(it1)
        return self.raw_data

    
    def RemovePeaksBasedOnRMS(self,wave, nSigma):
        """
        sets Datapoints above a certain rms*nSigma to 0
        takes: raw signal
        returns: peak-less signal
        """
        rms = np.std(wave)
        mean = np.mean(wave)
        if abs(mean)>1e-2: print "Warning, mean not near 0!!"
        
        cut = nSigma * rms
        #print "RMS = ", rms, " Cutting at: ", cut
        
        indices = np.where(np.abs(wave)<cut)[0] 
        # search indices, where wave is below upper limit
        wave_cut = np.zeros(wave.size) 
        # create np array of zeros of aquivalent size
        wave_cut[indices] = wave[indices] 
        # fill only those indices, where wave is below upper limit -> peakless signal

        return wave_cut


    def Slownoise_Smooth(self,data):
        """
        generates smoothed signal from peak-less signal and raw_data
        takes: peak-less signal , raw_data
        returns: smoothed signal 1
        """
        slownoise = self.RemovePeaksBasedOnRMS(data,3)
        nsamples = 500
        window = sig.general_gaussian(nsamples, p=1.0, sig=nsamples/5)
        slownoise_smooth = sig.fftconvolve(slownoise, window, "same")
        slownoise_smooth = (np.average(slownoise) / np.average(slownoise_smooth)) * slownoise_smooth
        self.filtered_signal_1 = data - slownoise_smooth
        self.peakless = slownoise 
        '''
        plt.figure()
        plt.title('slownoise')
        plt.xlabel('Time [s]')
        plt.ylabel('Voltage [V]')
        plt.plot(self.time,data,'grey')
        plt.plot(self.time,slownoise,'red')
        plt.title('slownoise_smooth')
        plt.plot(self.time,slownoise_smooth,'blue')
        #plt.figure()
        plt.xlabel('Time [s]')
        plt.ylabel('Voltage [V]')
        #plt.title('raw data ,filtered signal 1')
        #plt.plot(self.time,data,'grey')
        #plt.plot(self.time,self.filtered_signal_1)
        plt.show()
        '''
        return self.filtered_signal_1


    def Fastnoise_Smooth(self,filtered_signal_1):
        """
        smoothes noise-less signal again
        takes: noise-less signal
        returns: smoothed noise-less signal
       
        peak detection more reliable
        """
        nsamples = 10#20
        window = sig.general_gaussian(nsamples, p=1.0, sig=nsamples/5)
        self.filtered_signal_2 = sig.fftconvolve(filtered_signal_1, window, "same")
        self.filtered_signal_2 = (np.average(filtered_signal_1) / np.average(self.filtered_signal_2)) * self.filtered_signal_2

        return self.filtered_signal_2


    def FindPeaks(self,filtered_signal_2):
        """
        searches for the peak position in the smoothed noise-less signal
        takes: smoothed noise-less signal
        returns: List of Peak Indeces
        """
        '''
        bol = os.path.isfile(self.savepath+'RelGainRegrLineData.csv.npy')
        bol = False
        if bol:
            print
            print 'Regr Line File exists, extracting Gain'+str(bol)
            #rms = np.std(filtered_signal_2)
            slopefile = np.load(self.savepath+'RelGainRegrLineData.csv.npy')
            temp = slopefile[0]
            slope = slopefile[1]
            intercept = slopefile[2]
            index = np.where(temp == self.T)
            slo = slope[index]
            inter = intercept[index]
            gain = float(slo * self.vB + inter)
            gaintenth = gain*0.01
            rms = gaintenth
            print 2*gaintenth
            print np.std(filtered_signal_2)
            pulse_pos = detect_peaks(filtered_signal_2, 2*rms, mpd=10,show=False)
        else:
            print
            print 'no previous Regr Line File'
            rms = np.std(filtered_signal_2)
            print rms
            pulse_pos = detect_peaks(filtered_signal_2, rms*2, mpd=10,show=False)
            '''
        rms = np.std(self.peakless)
        #print rms
        #print np.std(filtered_signal_2)
        pulse_pos = detect_peaks(filtered_signal_2, rms*2, mpd=10,show=False)



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
        wl = 2 # in samples, so 2.5 samples == 1 ns, 10 ns = 25 samples
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
        wl = 2 # in samples, so 2.5 samples == 1 ns, 10 ns = 25 samples
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
        
        nbins = 300
        fig1 = plt.figure(1)
        ax1 = fig1.add_axes([0.1, 0.1, 0.8, 0.8])
        ax1.set_title('Pulse Height')
        ax1.set_yscale("log")
        ax1.hist(pulse_h,bins = nbins,color='grey',label='Height')
        ax1.hist(pulse_h_c,bins = nbins,color='blue',label='Height Clean')
        plt.legend()

        fig2 = plt.figure(2)
        ax2 = fig2.add_axes([0.1, 0.1, 0.8, 0.8])
        ax2.set_title('Pulse Area')
        ax2.set_yscale("log")
        ax2.hist(pulse_a,bins = nbins,color='lightgrey',label='Area')
        ax2.hist(pulse_a_c,bins = nbins,color='blue',label='Area Clean')
        plt.legend()

        fig3 = plt.figure(3)
        ax3 = fig3.add_axes([0.1, 0.1, 0.8, 0.8])
        ax3.set_title('Raw Data')   
        ax3.plot(self.time,self.raw_data,'lightgrey',self.time[ppc],self.filtered_signal_2[ppc],'rD')
        '''
        fig4 = plt.figure(4)
        ax4 = fig4.add_axes([0.1, 0.1, 0.8, 0.8])
        ax4.set_title('ZeroPeak')
        ax4.hist(self.peakless,bins = nbins)
        
        fig5 = plt.figure(5)
        ax5 = fig5.add_axes([0.1, 0.1, 0.8, 0.8])
        ax5.set_title('Raw, 1Filtered, 2Filtered')
        ax5.plot(self.time,self.raw_data,'lightgrey',label='Raw Data')
        #plt.plot(self.time,self.filtered_signal_1,'blue')
        ax5.plot(self.time,self.filtered_signal_2,'red',label='2Filtered')
        '''
        print "SAVING"
        fig1.savefig(self.data_file_path+'PulseHeight.pdf',format='pdf', bbox_inches='tight')
        fig2.savefig(self.data_file_path+'PulseArea.pdf',format='pdf', bbox_inches='tight')
        fig3.savefig(self.data_file_path+'PulsePosClean.pdf',format='pdf', bbox_inches='tight')
        #fig4.savefig(self.data_file_path+'ZeroPeak.pdf',format='pdf', bbox_inches='tight')
        #fig5.savefig(self.data_file_path+'Raw1Filtered2Filtered.pdf',format='pdf', bbox_inches='tight')

        fig1.clf()
        fig2.clf()
        fig3.clf()
        #fig4.clf()
        #fig5.clf()


        if args.plot:
            if args.plot == 3:plt.show()
            if not (args.plot == 2) and not (args.plot == 1) and not (args.plot == 3):raise ValueError('plot not 1,2,3')





    def GetZeroPeak(self):
        self.meanzero.append(np.mean(self.peakless)) # changed to plot the ZeroPeak with GaussFit self.meanzero.append(self.peakless)
        self.sigmazero.append(np.std(self.peakless))

    def Reduce_Data(self):
        self.LoadPresets()
        segments = self.GetSegmentCount()
        for ite in range(0,segments):#-99):    #-99 needed just for the last plot option
            data = self.Load_Data(ite)
            signal1 = self.Slownoise_Smooth(data)
            signal2 = self.Fastnoise_Smooth(signal1)
            pulse_pos = self.FindPeaks(signal2)
            pulse_pos_clean = self.PulseCleanUp(pulse_pos)
            self.Pulse(data,pulse_pos)
            self.PulseClean(data,pulse_pos_clean)
            #self.GetZeroPeak()
            if args.plot: 
                if args.plot == 1:
                    if ite%10 == False:
                        self.Plot(self.pulse_height,self.pulse_height_clean,self.pulse_area,self.pulse_area_clean,pulse_pos_clean)
                if not (args.plot == 2) and not (args.plot == 1) and not (args.plot == 3):raise ValueError('plot not 1,2,3')
        '''
        plt.figure()
        plt.subplot(211)
        plt.xlabel('Time [s]')
        plt.ylabel('Voltage [V]')
        plt.title('Peak Pos vs Peak Pos Clean')
        plt.plot(self.time,signal2,'b',self.time[pulse_pos],signal2[pulse_pos],'rD')
        plt.subplot(212,sharex=plt.subplot(211))
        plt.xlabel('Time [s]')
        plt.ylabel('Voltage [V]')
        plt.plot(self.time,signal2,'b',self.time[pulse_pos_clean],signal2[pulse_pos_clean],'rD')
        plt.show()
        '''
        pulse_h = np.asarray(self.pulse_height)
        pulse_h_c = np.asarray(self.pulse_height_clean)
        pulse_a =np.asarray(self.pulse_area)
        pulse_a_c = np.asarray(self.pulse_area_clean)
        #sigma = np.mean(self.sigmazero)

        if args.plot: 
            if args.plot == 2:self.Plot(pulse_h,pulse_h_c,pulse_a,pulse_a_c,pulse_pos_clean)
            if not (args.plot == 1) and not  (args.plot == 2)and not (args.plot == 3) :raise ValueError('plot not 1,2,3')

        return pulse_h, pulse_h_c , pulse_a, pulse_a_c,self.peakless#self.meanzero


    def Save_Data(self):
        
        pulse_h,pulse_h_c,pulse_a,pulse_a_c,mean = WfRed(self.data_file_path,self.savepath).Reduce_Data()
        data_path = str(self.data_file_path + 'Area')
        data_path_clean = str(self.data_file_path + 'AreaClean')
        data_path_mean = str(self.data_file_path + 'ZeroMean')
        np.save(data_path,pulse_a)
        np.save(data_path_clean,pulse_a_c)
        np.save(data_path_mean,mean)


parser = argparse.ArgumentParser(description='Decide:')
parser.add_argument('-p','--plot',type=int,help='-px or --plot x | 3 show and save, 2 save last segment, 1 save every 10th (so many)')
args = parser.parse_args()


def GetName(destT,destV,Date):
    # need to implement date of experiment into data file path
    path = '/home/gebhardt/00_SiPM_MPIK/scripts/Data/Date'+str(Date)+'/'+str(destT)+'deg/'+str(destV)+'V/'
    name = 'SensL_T' + str(destT) + '_Vb' + str(destV) + '.trc'
    compl_name = os.path.join(path,name)
    return compl_name

def GetSavePath(Date):
    savepath = '/home/gebhardt/00_SiPM_MPIK/scripts/Data/Date'+str(Date)+'/'
    return savepath

def LoadRunCard(Date):
    a = np.load('/home/gebhardt/00_SiPM_MPIK/scripts/Data/Date'+str(Date)+'/RunCard'+str(Date)+'.npy')
    #print a[0]
    #print a[1]
    #print a[2]
    return a





#@profile       
def main():
    exp = LoadRunCard('2809163')
    #print exp

    Date =exp[0]# ['200616']
    T = exp[1]#[5.,10.,15.,23.0,26.0,29.0]
    Vb =exp[2]# []
    for dAte in Date:
        savepath = GetSavePath(dAte)
        for destT in T:
            print 'Commence Data Reduction T= ',destT
            for v in Vb:
                compl_name = GetName(destT,v,dAte)
                print 'Saving ',compl_name 
                WfRed(compl_name,savepath).Save_Data()



    
if __name__ == '__main__':
    main()





