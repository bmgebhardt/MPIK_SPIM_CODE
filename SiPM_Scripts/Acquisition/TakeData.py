#filehandler for Lecroy Scope data

import Oscilloscope
import os
import io
import LecroyBinary
import time
import LoopPlotter
import matplotlib.pyplot as plt
import os.path
import ThermalChamber
from TTi_PLH120P_PSU import TTi_PLH120P_PSU
import numpy as np



class TraceFileHandler():

    def GetNextBinaryWaveformData(self,iterator):
        command = 'TRFL? DISK,HDD,FILE,D:\Waveforms\C1Trace' + iterator + '.trc'
        data = osc.ask(command)
        datastr = str(data)
        return datastr
 
    def SaveTrcToDisc(self,ite,path,file_name='C1Trace.trc'):
        iterator ='{0:05d}'.format(ite)
        #file_name = 'C1Trace'+iterator+'.trc'
        complete_name = os.path.join(path, file_name)
        data = self.GetNextBinaryWaveformData(iterator)
        time.sleep(1)
        f = open(complete_name,'w')
        f.write(data)
        f.close()

    def TakeData(self,acq):
        command = str('python Oscilloscope.py -c1 -a'+str(acq))
        print command
        os.system(command)
        time.sleep(5)


def main():



    date = '1210161'

    # MOLTO IMPORTANTE
    start_TraceID = 8880#next ID to be generated
    TraceID = start_TraceID

    tfh = TraceFileHandler()
    err_code = 0
    temp_increment = 0
    acq = 1
    Volt_Div = 0.050 #for later use
    first = True
    tti =  TTi_PLH120P_PSU('TTI', '/dev/ttyACM0',1)
    Temps =[0.,5.,10.,15.,20.,25.,30.,35.]##[18.,22.,26.,30.]

    #vB = vBlist
    vB = [37.,37.5,38.,38.5,39.,39.2,39.4,39.6,39.8,40.,40.2,40.4,40.6,40.8,41.,41.5,42.,42.5,43.,43.5,44.]
    print vB
    tti.open()
    tti.sendCmd("*IDN?")
    print 'IDN?' 
    IDN = tti.read()
    print "IDN = "+str(IDN)
    tti.switchOutputOFF()
    tti.setI(0.01) #Change appropriately 
    


    for counttemp,destT in enumerate(Temps):#capital loop
        
        tti.setV(0)
        print 'therm?'
        therm = ThermalChamber.ThermalChamber('169.254.166.11',57732)
        therm.Connect(err_code)
        therm.GetStatus()
        err_code, dT = therm.SetTemp(destT,err_code)
        time.sleep(5)
        therm.GoToState(1,err_code)

        if not first: # called after every run except the first, sends previous data set during Temp change
            for j,v in enumerate(vB):
                print TraceID
                file_name = 'HAM_T' + str(Temps[counttemp-1]) + '_Vb' + str(v) + '.trc'
                save_path =  '/home/cta/ben/SiPM_Testing/scripts/Data/Date'+str(date)+'/'+str(Temps[counttemp-1])+'deg/'+str(v)+'V/'
                if not os.path.isdir(save_path):
                    os.makedirs ('/home/cta/ben/SiPM_Testing/scripts/Data/Date'+str(date)+'/'+str(Temps[counttemp-1])+'deg/'+str(v)+'V/')
                complete_name = os.path.join(save_path, file_name)
                if os.path.isfile(complete_name):
                    date = str(int(date)+1)
                    os.makedirs ('/home/cta/ben/SiPM_Testing/scripts/Data/Date'+str(date)+'/'+str(Temps[counttemp-1])+'deg/'+str(v)+'V/')
                    file_name = 'HAM_T' + str(Temps[counttemp-1]) + '_Vb' + str(v) + '.trc'
                    save_path =  '/home/cta/ben/SiPM_Testing/scripts/Data/Date'+str(date)+'/'+str(Temps[counttemp-1])+'deg/'+str(v)+'V/'
                    complete_name = os.path.join(save_path, file_name)

                
                #keep plotting
                cur_sta , cur_temp , set_temp = therm.GetStatus()
                ts = time.time()
                #ask PSU for voltage
                volt = tti.getV()
                plt.scatter(ts,volt,color='red')
                plt.scatter(ts,cur_temp,color='blue')
                plt.pause(0.05)
                print 'saving ', TraceID, ' run as ', file_name, ' in ', save_path
                tfh.SaveTrcToDisc(TraceID,save_path,file_name)
                TraceID = TraceID + 1




        cur_sta , cur_temp , set_temp = therm.GetStatus()
        cur_temp = -999
        finished = 0
        countb = 0
        while finished < 5: 
            cur_sta , cur_temp , set_temp = therm.GetStatus()
            countb += 1
            ts = time.time()
            #ask PSU for voltage
            volt = tti.getV()
            plt.scatter(ts,volt,color='red')
            plt.scatter(ts,cur_temp,color='blue')
            plt.pause(0.05)
            if abs(cur_temp - set_temp) <= 0.2:
                finished += 1
                print 'finished no. ', finished, ' of 5'
            time.sleep(10)
        print 'temp reached!!!!'

        #Vb loop
        plt.ion() #realtimeplotting activated
        plt.figure(1)
        i = 0
        for v in vB:
            tti.setV(v)
            tti.switchOutputON() # Turn on PSU
            time.sleep(1) #Let voltage settle
            ts = time.time()
            plt.scatter(ts,v,color = 'red')
            #ask chamber for temp
            cur_sta , cur_temp , set_temp = therm.GetStatus()
            plt.scatter(ts,cur_temp,color = 'blue')
            plt.pause(0.05)
            tfh.TakeData(acq)
            
            tti.switchOutputOFF() #Turn off PSU
            
            #for ind in range(TraceID,TraceID+acq):
        first = False

    if not first: #Called after the first data set has been taken, sends first data set during Temp change
        for j,v in enumerate(vB):
            print TraceID
            file_name = 'HAM_T' + str(destT) + '_Vb' + str(v) + '.trc'
            save_path =  '/home/cta/ben/SiPM_Testing/scripts/Data/Date'+str(date)+'/'+str(destT)+'deg/'+str(v)+'V/'
            if not os.path.isdir(save_path):
                os.makedirs ('/home/cta/ben/SiPM_Testing/scripts/Data/Date'+str(date)+'/'+str(destT)+'deg/'+str(v)+'V/')
            complete_name = os.path.join(save_path, file_name)

            if os.path.isfile(complete_name):
                date = str(int(date)+1)
                os.makedirs ('/home/cta/ben/SiPM_Testing/scripts/Data/Date'+str(date)+'/'+str(destT)+'deg/'+str(v)+'V/')
                save_path =  '/home/cta/ben/SiPM_Testing/scripts/Data/Date'+str(date)+'/'+str(destT)+'deg/'+str(v)+'V/'
                complete_name = os.path.join(save_path, file_name)

            print date,': saving ', TraceID, ' run as ', file_name, ' in ', save_path
            tfh.SaveTrcToDisc(TraceID,save_path,file_name)
            TraceID = TraceID + 1




    tti.setV(0)
    tti.setI(0)
    tti.switchOutputOFF()
    tti.close()
    print 'done'
    destT = 20
    err_code, dT = therm.SetTemp(destT,err_code)
    print dT
    time.sleep(5)
    #therm.GoToState(1,err_code)
    #while True:
        #plt.pause(0.05)

if __name__ == '__main__':
    osc = Oscilloscope.Oscilloscope(host = '169.254.84.162')
    print 'Osc'
    main()






