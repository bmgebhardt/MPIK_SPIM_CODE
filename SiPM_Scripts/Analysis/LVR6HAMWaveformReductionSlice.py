# Waveform Reduction of Data for HAM SiPM
#using slicing with LecroyBinarySlicing
#implemented regression with -r1

import LecroyBinarySlicing
import numpy as np
from scipy import signal as sig
from detect_peaks import detect_peaks
from memory_profiler import profile
import sys
import argparse
import matplotlib.pyplot as plt
import os.path

fromregr = False
savestr = 'FixedGuess1p8mV510IntWinMPD25'# 1 'FixedGuess' 2 'Gainx0p25' 3 'FixedVal'

class WfRed(object):

    def __init__(self,data_file_path,savepath,Vb,iterT,iterV,high_pass):
        self.vB = 0.0
        self.T = 0.0
        self.filtered_signal_1 = np.zeros(0)
        self.filtered_signal_2 = np.zeros(0)
        self.ranger = 0
        self.numdata = 0
        self.ite = 0
        self.pulse_height = []
        self.pulse_height_clean = []
        self.pulse_area = []
        self.pulse_area_clean = []
        self.samplerate = 2.5e9
        self.peakless = []
        self.meanzero = []
        self.sigmazero = []
        self.data_file_path = data_file_path
        self.savepath = savepath
        self.calcsegmentspro = 100
        self.V = Vb
        self.iterT = iterT
        self.iterV = iterV
        self.high_pass = high_pass


    def ReportProgress(self,ite,seg):
        pro = ite*100/seg
        #print seg
        #print pro
        if (pro % 10) == 0: # print every 10%
            print pro,' % done'


    def GetSegmentCount(self):
        self.ranger = LecroyBinarySlicing.LecroyBinaryWaveform(self.data_file_path).SubArray()
        return self.ranger

    def GetNumDataPoints(self):

        self.numdata = LecroyBinarySlicing.LecroyBinaryWaveform(self.data_file_path).WaveArray()
        return self.numdata

    def LoadTime(self):

        samplerate = self.samplerate
        length = LecroyBinarySlicing.LecroyBinaryWaveform(self.data_file_path).WaveArray()
        time = length/samplerate
        #recordtime = LecroyBinarySlicing.LecroyBinaryWaveform(self.data_file_path).ReturnRecordTime()
        #print recordtime
        return time

    def Load_Data(self):
        """
        Loopable through counter in function call
        raw_data from filecontent
        """
        # return all segments 
        #self.raw_data,self.time=LecroyBinary.LecroyBinaryWaveform(self.data_file_path).RealGetNextWaveArrayDataAndTime(it1) # memory leak
        #raw_data=LecroyBinary.LecroyBinaryWaveform(self.data_file_path).GetNextWaveArrayData() # reshape outside loop
        #time=LecroyBinary.LecroyBinaryWaveform(self.data_file_path).GetNextWaveArrayTime() #reshape outside loop
        raw_data, time = LecroyBinarySlicing.LecroyBinaryWaveform(self.data_file_path).JustAllGetNextWaveArrayDataAndTime()#raw output from lecroybinary        
        return raw_data,time


    def RemovePeaksBasedOnRMS3(self,wave_uncentered):
        """
        gets negative of waveform and calculates rms
        uses rms to calculate peakless waveform        
        """   


        mean0 = np.mean(wave_uncentered)
        #print mean0
        wave = wave_uncentered - mean0 #wave at 0

        mean1 = np.mean(wave)  #mean1 ~ 0
        indices_negative2 = np.where(wave<mean1)[0] 
        wave_cut_negative2 = np.zeros(wave.size) 
        wave_cut_negative2[indices_negative2] = wave[indices_negative2] 
        
        mean = np.mean(wave_cut_negative2)
        negative_indices = np.where(wave<mean1)[0] 
        positive_indices = np.where(wave>mean1)[0] 

        # search indices, where wave is below upper limit 0
        wave_cut_negative = np.zeros(wave.size) 
        # create np array of zeros of aquivalent size
        wave_cut_negative[negative_indices] = wave[negative_indices] 
        # fill only those indices, where wave is below upper limit -> peakless signal
        wave_cut_negative[positive_indices] = -wave[positive_indices] 
      
        rms = np.std(wave_cut_negative)# get rms of negative signal

        indices = np.where(np.abs(wave)<rms*3)[0] # wave between rms*2, both sides # changed to cutting only top peaks, due to pedestal sub bug
        wave_cut = np.zeros(wave.size) 
        wave_cut[indices] = wave[indices] #fill wave 

        cut = rms#cut-level
        #print cut
        return wave_cut+mean0,cut



    def Slownoise_Smooth(self,data,ite):
        """
        generates smoothed signal from peak-less signal and raw_data
        takes: peak-less signal , raw_data
        returns: smoothed signal 1
        """


        slownoise,level = self.RemovePeaksBasedOnRMS3(data)
        meanzero = np.mean(slownoise)
        self.peakless = slownoise - meanzero #peakless signal - pedestal mean
        data_at_0 = data - meanzero #subtract mean of pedestal from data 
        # not given that trace is always with same offset // gaincalc only takes into account meanzero of last segment
        nsamples = 700 #determines width of smoothing gaussian (/5) = sigma of gauss 
        window = sig.general_gaussian(nsamples, p=1.0, sig=nsamples/5)
        slownoise_smooth = sig.fftconvolve(self.peakless, window, "same")
        slownoise_smooth = (np.average(self.peakless) / np.average(slownoise_smooth)) * slownoise_smooth
        self.filtered_signal_1 = data_at_0 - slownoise_smooth

        self.ReportProgress(ite,int(float(self.calcsegmentspro)/100*self.ranger))
       
        #print 'datamean ',np.mean(data)
        #print 'meanzero ',meanzero
        #print 'data0 ',np.mean(data_at_0)

        return self.filtered_signal_1


    def Fastnoise_Smooth(self,filtered_signal_1):
        """
        smoothes noise-less signal again
        takes: noise-less signal
        returns: smoothed noise-less signal
        peak detection more reliable
        """
        
        sig_smooth_ns = 4 # Depends on pulse shape... 
        win_smooth_ns = sig_smooth_ns * 5 #... possibly FWHM * 1.25
        samples_per_ns = 2.5
        window = sig.general_gaussian(win_smooth_ns*samples_per_ns
                                      , p=1.0, sig=sig_smooth_ns*samples_per_ns)
        self.filtered_signal_2 = sig.fftconvolve(filtered_signal_1, window, "same")
        self.filtered_signal_2 = (np.average(filtered_signal_1) / np.average(self.filtered_signal_2)) * self.filtered_signal_2



        return self.filtered_signal_2


    def FindPeaks(self,filtered_signal_2):
        """
        searches for the peak position in the smoothed noise-less signal
        takes: smoothed noise-less signal
        returns: List of Peak Indeces
        """
        #rmsn = 1.9#rmsval#1.5
        indices = np.where(self.peakless<0)[0] 
        # search indices, where wave is below upper limit
        peakless_cut = np.zeros(len(self.peakless)) 
        # create np array of zeros of aquivalent size
        peakless_cut[indices] = self.peakless[indices] 
        # fill only those indices, where wave is below upper limit -> peakless signal



        if args.regr:
            if args.regr ==1:
                rms = calcregr(self.V,self.savepath,self.iterT)
            if args.regr ==2:
                rms = self.high_pass

        else:
            rmsn = calcrmsn(self.V)
            rms = np.std(self.peakless) *rmsn
        #print rms*2
        #print rms
        if args.plot == 3:
            pulse_pos = detect_peaks(filtered_signal_2, rms, mpd=25,show=True)
        else:
            pulse_pos = detect_peaks(filtered_signal_2, rms, mpd=25,show=False)
 
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
        wl = 5#2 # in ns, so 2.5 samples == 1 ns, 10 ns = 25 samples
        wr = 10#4 
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
        wl = 5#2 # in ns, so 2.5 samples == 1 ns, 10 ns = 25 samples #10?
        wr = 10#4 #15?
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
        fig1 = plt.figure(1)
        ax1 = fig1.add_axes([0.1, 0.1, 0.8, 0.8])
        ax1.set_title('Pulse Height')
        ax1.set_yscale("log")
        #ax1.set_xlim([-0.025,0.2])
        ax1.hist(pulse_h,bins = nbins/4,color='grey',label='Height',histtype='step')
        ax1.hist(pulse_h_c,bins = nbins/4,color='blue',label='Height Clean',histtype='step')
        ax1.hist(self.peakless,bins = nbins/100,color='green',histtype='step')
        plt.legend()


        fig2 = plt.figure(2)
        ax2 = fig2.add_axes([0.1, 0.1, 0.8, 0.8])
        ax2.set_title('Pulse Area')
        ax2.set_yscale("log")
        #ax2.set_xlim([-1,8])
        ax2.hist(pulse_a,bins = nbins,color='grey',label='Area',histtype='step')
        ax2.hist(pulse_a_c,bins = nbins,color='blue',label='Area Clean',histtype='step')
        ax2.hist(self.peakless,bins = nbins/100,color='green',histtype='step')
        plt.xticks(np.arange(0 ,2, 0.1))

        plt.legend()



        print "SAVING"
        fig1.savefig(self.data_file_path+str(savestr)+'PulseHeight.pdf',format='pdf', bbox_inches='tight')
        fig2.savefig(self.data_file_path+str(savestr)+'PulseArea.pdf',format='pdf', bbox_inches='tight')
        if args.show == 1:plt.show()
        fig1.clf()
        fig2.clf()



    def GetZeroPeak(self):#still needed, zeropeak now from filtered2
        #self.meanzero.append(np.mean(self.peakless)) # changed to plot the ZeroPeak with GaussFit before:self.meanzero.append(self.peakless)
        self.meanzero.append(np.mean(self.peakless))
        self.sigmazero.append(np.std(self.peakless))


    def Reduce_Data(self):
        try:
            segments = self.GetSegmentCount()
            NumDataAll = self.GetNumDataPoints()
        except ValueError:
            print 'Data Corrupted'
            segments = 0

        NumPerSeg = NumDataAll/segments

        calcseg = int(float(self.calcsegmentspro)/100*segments)
        print 'Calculating ',calcseg,' Segments , ',self.calcsegmentspro,' % of total count'
        all_data , all_time = self.Load_Data()

        for ite in range(0,calcseg):    
            start = ite*NumPerSeg
            end = (ite+1)*NumPerSeg
            data = all_data[start:end:]
            time = all_time[start:end:]
            self.signal1 = self.Slownoise_Smooth(data,ite)
            signal2 = self.Fastnoise_Smooth(self.signal1)
            pulse_pos = self.FindPeaks(signal2)
            pulse_pos_clean = self.PulseCleanUp(pulse_pos)
            self.Pulse(data,pulse_pos)
            self.PulseClean(data,pulse_pos_clean)
            self.GetZeroPeak()
            if args.plot: 
                if args.plot == 1:
                    if ite%10 == 0:
                        self.Plot(self.pulse_height,self.pulse_height_clean,self.pulse_area,self.pulse_area_clean,pulse_pos_clean)
                if not (args.plot == 2) and not (args.plot == 1) and not (args.plot == 3):raise ValueError('plot not 1,2,3')

        pulse_h = np.asarray(self.pulse_height)
        pulse_h_c = np.asarray(self.pulse_height_clean)
        pulse_a =np.asarray(self.pulse_area)
        pulse_a_c = np.asarray(self.pulse_area_clean)

        if args.plot: 
            if args.plot == 2:
                self.Plot(pulse_h,pulse_h_c,pulse_a,pulse_a_c,pulse_pos_clean)

            if not (args.plot == 1) and not  (args.plot == 2)and not (args.plot == 3) :raise ValueError('plot not 1,2,3')

        return pulse_h, pulse_h_c , pulse_a, pulse_a_c,self.meanzero # was filtered2


    def Save_Data(self):
        
        pulse_h,pulse_h_c,pulse_a,pulse_a_c,zeropeakless = WfRed(self.data_file_path,self.savepath,self.V,self.iterT,self.iterV,self.high_pass).Reduce_Data()
        
        data_path_h = str(self.data_file_path +str(savestr)+ 'Height')
        data_path_h_clean = str(self.data_file_path +str(savestr)+ 'HeightClean')
        np.save(data_path_h,pulse_h)
        np.save(data_path_h_clean,pulse_h_c)

        data_path_a = str(self.data_file_path +str(savestr)+ 'Area')
        data_path_a_clean = str(self.data_file_path +str(savestr)+ 'AreaClean')
        np.save(data_path_a,pulse_a)
        np.save(data_path_a_clean,pulse_a_c)

        data_path_mean = str(self.data_file_path +str(savestr)+ 'ZeroPeak')
        np.save(data_path_mean,zeropeakless)

        



parser = argparse.ArgumentParser(description='Decide:')
parser.add_argument('-p','--plot',type=int,help='-px or --plot x | 2 save last segment, 1 save every 10th (so many)')
parser.add_argument('-s','--show',type=int,help='-sx or --show x | 1 show last segment')
parser.add_argument('-d','--date',type=str,help='-dx or --date x | date id of the experiment to reduce')
parser.add_argument('-r','--regr',type=int,help='-rx or --regr x | 0 =NO regr , 1=fromregr , 2=fixed')
args = parser.parse_args()


def GetName(destT,destV,Date):
    path = '/home/gebhardt/00_SiPM_MPIK/scripts/Data/Date'+str(Date)+'/'+str(destT)+'deg/'+str(destV)+'V/'
    name = 'HAM_T' + str(destT) + '_Vb' + str(destV) + '.trc'
    compl_name = os.path.join(path,name)
    return compl_name

def GetSavePath(Date):
    savepath = '/home/gebhardt/00_SiPM_MPIK/scripts/Data/Date'+str(Date)+'/'
    return savepath

def LoadRunCard(Date):
    a = np.load('/home/gebhardt/00_SiPM_MPIK/scripts/Data/Date'+str(Date)+'/RunCard'+str(Date)+'.npy')
    return a
      
def calcrmsn(Vb):
    1/0
    #calcrmsn still unreliable
    #args.regr = 0
    b = np.asarray([66.0,67.0,68.0,69.0]) # this is SiPM specific (CHEC-S atm)
    a= np.asarray([1.9,1.8,1.7,1.6]) #pixel46 e.g.
    slope, intercept = np.polyfit(b, a, 1)
    rmsn = Vb*slope+intercept
    return rmsn 

def calcregr(Vb,savepath,iterT):
    1/0
    #args.regr = 1
    conv_fac = (float(int(4*2.5)+int(5*2.5)))*1.3
    regrfile = np.load(savepath+'AreaRelGainRegrLineData.npy')
    slope = regrfile[1][iterT]
    inter = regrfile[2][iterT]
    gainregr = slope*Vb+inter #relGain
    '''
    set_gain = 0.0028 #AbsGain
    if gainregr < (set_gain*conv_fac):
        gainregr = set_gain*conv_fac
    '''
    #gainregr = slope*67.4+inter #first Vb
    #print 'Regr Gain: ',gainregr/conv_fac*0.6
    #print gainregr/conv_fac
    return gainregr/conv_fac*0.25
    #still not correct....


#def calcgain(savepath,iterT,iterV):
    #args.regr = 2
    #return 0.0031
    # fixed value
    # could also try with gaincalc from previous run  



    #return (0.130/36.5)*0.75       #1/2  *  relGain/Factor
    #datalist = np.load(savepath+'AreaNPYDataTable.npy')
    #gaincalc = datalist[2][iterT][iterV]
    #print 'Regr Gain: ',gaincalc*0.7/window
    #return gaincalc*0.7/window
    




def main():

    if args.date:
        datE = args.date
        print datE
    else:
        datE = '2811164'#'0411161'          
        print datE
    Date, T, Vb = LoadRunCard(datE)
    high_pass = 0.0018 #peak find threshold
    #Vb = [39.7,39.9]
    #T = [26.]
    #Vb = [66.]#,68.4,69.4]
    #Vb = [39.,39.2,39.4,39.6,39.8,40.,40.2,40.4,40.6,40.8,41.]
    #Vb =[66.,66.2,66.4,66.5,66.6,66.7,66.8,66.9]
    #Vb =[67.,67.1,67.2,67.3,67.4,67.5,67.6,67.7,67.8,68.,68.2,68.4,68.6,68.8,69.]
    print T
    print Vb
    print args
    #rmsvallist = [3,2.8,1.6]
    for dAte in Date:
        savepath = GetSavePath(dAte)
        for iterT,destT in enumerate(T):
            print 'Commence Data Reduction for T= ',destT
            for iterV,v in enumerate(Vb):
                compl_name = GetName(destT,v,dAte)
                print 'Saving ___V___ ',savestr,compl_name 
                WfRed(compl_name,savepath,v,iterT,iterV,high_pass).Save_Data()




    
if __name__ == '__main__':
    main()


