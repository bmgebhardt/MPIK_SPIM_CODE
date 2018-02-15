#connect to Thermal Chamber

#states defined by MODE command -1 unknown | 0 standby | 1 constant

#err_code:
# 0 everything okay
# 1 connection failed
# 2 I dont know this thermal chamber
# 3 miscommunication while sending data
# 4 unknown destination state
# 5 destination temperature not set

# Rate of change, updated... tell user how long left
# import matplot lib, and have a drawing option from main...
# python ThermalChamber.py -t 50 -p true

import socket
import pdb
import time
import sys


#self.socket


class ThermalChamber():

    def __init__(self,ip,port):
        self.ip = ip
        self.port = port
        self.socket = ""


    def SendAndRecv(self,command):

        BUFFER_SIZE = 256
        self.socket.send(command+'\r\n')

        err_code = 0
        try:
            self.socket.send(command+'\r\n')
        except self.socket.error as msg:
            state = -1
            err_code = 1
            print 'connection failed', msg
            #sys.exit(1)
        except self.socket.timeout as msg:
            state = -1
            err_code = 1
            print 'timeout ' , msg
            #sys.exit(1)
        data = self.socket.recv(BUFFER_SIZE)
        print data

        #data = ''
        #while data != '\n':
        flush = self.socket.recv(512)
        #    print "Testing for empty:" + data + "--"

        return data , err_code

    def Connect(self,err_code):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        self.socket.settimeout(30)
        try:
            self.socket.connect((self.ip,self.port))
        except socket.error as msg:
            err_code = 1
            print 'connection failed ', msg
            print 'Error ',err_code
            sys.exit(1)
        except socket.timeout as msg:
            err_code = 1
            print 'timeout ' , msg
            print 'Error ',err_code
            sys.exit(1)

        rom, err_code = self.SendAndRecv('ROM?')
        print 'connected to ' + rom + '           = P2LSCCN10.09STD'
       
        return err_code



    def GetStatus(self):
        thcstatus, err_code = self.SendAndRecv('MON?')
        state = -1
        print 'thcstatus ', thcstatus
        thcstat = str(thcstatus)
        print 'thcstat ', thcstat
        if "CONSTANT" in thcstat:
            state = 1
            print 'CONSTANT'
        if "STANDBY" in thcstat:
            state = 0
            print 'STANDBY'
        else:
            err_code = 4
            print err_code, ' check thcstat conversion'

        tempstatus, err_code = self.SendAndRecv('TEMP?')
        #tempstatus, err_code = self.SendAndRecv('TEMP?')
        #tempstatus, err_code = self.SendAndRecv('TEMP?')
        #print "-----------<<<< " + tempstatus
        #pdb.set_trace()
        split_tempstatus = tempstatus.split(',',2)
        #print "I found this many elements: " + str(len(split_tempstatus))
        #cur_temp = float(tempstatus.split(',',1)[0])
        #set_temp = float(tempstatus.split(',',2)[1])
        set_temp = float(split_tempstatus[1])
        cur_temp = float(split_tempstatus[0])
        print 'cur_temp ',cur_temp
        print 'set_temp ',set_temp

        return state , cur_temp , set_temp



    def Stop(self):
        self.GoToState(0,err_code)

    def Disconnect(self):
        self.GoToState(0,err_code)
        self.socket.Shutdown(1)
        self.socket.Close()

    def GoToState(self,sta,err_code):
        if sta == -1:
            err_code = 4
            print 'unknown state'
            self.Stop()
        if sta == 0:
            status = 'MODE,STANDBY'
        if sta == 1:
            status = 'MODE,CONSTANT'

        data, err_code = self.SendAndRecv(status)
        if err_code > 0:
            print ' unable to acces state', sta, ' Error', data
            cur_sta == -1

        print 'going to state ', status

        cur_sta , cur_temp , set_temp = self.GetStatus()
        while cur_sta != sta:
            cur_sta , cur_temp , set_temp = self.GetStatus()
            print cur_sta
            if cur_sta == -1:
                print 'unknown state'
                #sys.exit(1)
            time.sleep(1)

        print 'state reached'

        return err_code

    def SetTemp(self,temp,err_code):
        dT = 0
        tempstr = str(temp)
        tempcommand ='TEMP,S' + tempstr
        #print tempcommand
        self.SendAndRecv(tempcommand)
        time.sleep(0.1)
        cur_sta , cur_temp , set_temp = self.GetStatus()
        if temp == set_temp:
            err_code = 0
        else:
            err_code = 5
            print 'Temp not set'
            #sys.exit(1)
        dT == set_temp - temp

        return err_code , dT


    def RunToTemp(self,temp,err_code):
        #self.GoToState(0,err_code)
        err_code, dT = self.SetTemp(temp,err_code)
        time.sleep(5)
        #if err_code > 0:
        #    print err_code
        #    sys.exit(1)
        self.GoToState(1,err_code)
        #est_time_to_temp = DATASHEET(deg/s)* dT
        cur_sta , cur_temp , set_temp = self.GetStatus()
        cur_temp = -999
        finished = 0
        countb = 0
        while finished < 5:
            cur_sta , cur_temp , set_temp = self.GetStatus()
            countb += 1
            print countb, ' 0.1deg test || * 10 seconds running'
            if abs(cur_temp - set_temp) <= 0.2:
                finished += 1
                print 'finished no. ', finished, ' of 5'
            time.sleep(10)
        print 'temp reached!!!!'

        return err_code


if __name__ == '__main__':
    print 'hi'

    err_code = 0
    thermal_chamber = ThermalChamber('169.254.166.11',57732)
    thermal_chamber.Connect(err_code)
    #print err_code
    thermal_chamber.GetStatus()
    #thermal_chamber.GoToState(1,err_code)
    #thermal_chamber.Stop()
    thermal_chamber.RunToTemp(0.0,err_code)
    #thermal_chamber.GetStatus()


     
