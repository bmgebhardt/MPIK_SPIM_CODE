#
#trying out slicing

#
import numpy as np
import time

def read_timetrace(filename):
  """
  Returns the time trace from the given file. Returns the time and
  voltage array, in that order.
  Both arrays are 1-D.
  """
  bwf = LecroyBinaryWaveform(filename)
  return bwf.WAVE_ARRAY_1_time, bwf.WAVE_ARRAY_1.ravel()
  
from contextlib import contextmanager
@contextmanager
def _open(filename, file_content):
  if file_content is None:
    fh = open(filename, 'rb')
    yield fh
    fh.close()
  else:
    try:
      from cStringIO import StringIO
    except:
      from StringIO import StringIO

    fh = StringIO(file_content)
    yield fh
    fh.close()

class LecroyBinaryWaveform(object):
  """
  Implemented according to specs at:
    http://teledynelecroy.com/doc/docview.aspx?id=5891
  Partially derived from lecroy.py from:
    http://qtwork.tudelft.nl/gitdata/users/guen/qtlabanalysis/analysis_modules/general/lecroy.py
  """

  def __init__(self, inputfilename, file_content=None):
    """
    inputfilename: path to .trc file to read
    file_content: if given, will be used in place of data on disk. Useful when
                  loading data from zips
    """
    super(LecroyBinaryWaveform, self).__init__()

    with _open(inputfilename, file_content) as fh:
      self.fh = fh
      header = self.fh.read(50)
      self.aWAVEDESC = str.find(header, 'WAVEDESC')

      def at(offset):
        return self.aWAVEDESC + offset

      # the lecroy format says COMM_ORDER is an enum, which is a 16 bit
      # value and therefore subject to endianness. However COMM_ORDER
      # dictates the endianness! However, since the possible values are
      # either 0, which is the same in either endianness, or 0x1 or 0x7000
      # in big/small endianness, we can just check for 0. Since all read_*
      # methods needs a define endianness, we can default to 0 and not
      # worry about being wrong because of the preceding argument.

      # XXX The attribute names are important! Any attribute that is all
      # caps and does not start with '_' is considered metadata and will
      # be exported as part of the metadata property. This means it will
      # also be written to file when saving as CSV

      self.COMM_ORDER             = 0

      # We do a double read because after the first read, we will know the
      # correct endianness based on the above argument, and therefore will
      # have the correct value for COMM_ORDER. Otherwise 1 becomes 0x0100
      # iff in little endian mode.
      self.COMM_ORDER             = self.read_enum(at(34))
      self.COMM_ORDER             = self.read_enum(at(34))


      self.TEMPLATE_NAME          = self.read_string(at(16))
      self.COMM_TYPE              = self.read_enum(at(32))
      self._WAVE_DESCRIPTOR_SIZE  = self.read_long(at(36))
      self._USER_TEXT_SIZE        = self.read_long(at(40))
      self._RES_DESC1_SIZE        = self.read_long(at(44))
      self._TRIGTIME_ARRAY_SIZE   = self.read_long(at(48))
      self._RIS_TIME_ARRAY_SIZE   = self.read_long(at(52))
      self._RES_ARRAY1_SIZE       = self.read_long(at(56))
      self._WAVE_ARRAY_1_SIZE     = self.read_long(at(60))

      self.INSTRUMENT_NAME        = self.read_string(at(76))
      self.INSTRUMENT_NUMBER      = self.read_long(at(92))

      self.TRACE_LABEL            = self.read_string(at(96))

      self.TRIG_TIME              = self.read_timestamp(at(296))

      self.RECORD_TYPE            = self.read_record_type(at(316))
      self.PROCESSING_DONE        = self.read_processing_done(at(318))

      self.VERTICAL_GAIN          = self.read_float(at(156))
      self.VERTICAL_OFFSET        = self.read_float(at(160))

      self.HORIZ_INTERVAL         = self.read_float(at(176))
      self.HORIZ_OFFSET           = self.read_double(at(180))



      #added code for sequencing
      self.SEGMENT_INDEX         = self.read_long(at(140))
      self.SUBARRAY_COUNT        = self.read_long(at(144))
      self.WAVE_ARRAY_COUNT      = self.read_long(at(116))
      self.RECORD_TIME           = self.read_long(at(312))





      self.a_WAVE_ARRAY_1             = at(self._WAVE_DESCRIPTOR_SIZE +
                                      self._USER_TEXT_SIZE +
                                      self._TRIGTIME_ARRAY_SIZE)

#      print '_WAVE_DESCRIPTOR_SIZE', self._WAVE_DESCRIPTOR_SIZE
#      print '_USER_TEXT_SIZE', self._USER_TEXT_SIZE
#      print '_RES_DESC1_SIZE', self._RES_DESC1_SIZE
#      print '_TRIGTIME_ARRAY_SIZE', self._TRIGTIME_ARRAY_SIZE
#      print '_RIS_TIME_ARRAY_SIZE', self._RIS_TIME_ARRAY_SIZE
#      print '_RES_ARRAY1_SIZE', self._RES_ARRAY1_SIZE
#      print '_WAVE_ARRAY_1_SIZE', self._WAVE_ARRAY_1_SIZE

#added code for sequencing____________________________
      #print 'Index of transmitted segment SEGMENT_INDEX = ',            self.SEGMENT_INDEX
      #print 'Number of acquired segments in sequencing SUB_ARRAY_COUNT = ',     self.SUBARRAY_COUNT
      #print 'Number of data points in the data array WAVE_ARRAY_COUNT = ',    self.WAVE_ARRAY_COUNT
   






      self._WAVE_ARRAY_1 = self.read_wave_array(self.a_WAVE_ARRAY_1)
      self.Index = 0
     
    self.fh = None


# bin 312 ACQ_DURATION: float
#; duration of the acquisition (in sec)
#; in multi-trigger waveforms.
#; (e.g. sequence, RIS, or averaging)

  def WaveArray(self):
    return self.WAVE_ARRAY_COUNT


  def SubArray(self):
    return self.SUBARRAY_COUNT


  def ReturnWaveform(self):
    return self._WAVE_ARRAY_1


  @property
  def WAVE_ARRAY_1_timeself(self):
    """
    A calculated array of when each sample in wave_form_1 was measured,
    based on HORIZ_OFFSET and HORIZ_INTERVAL.
    """
    tvec = np.arange(0, self._WAVE_ARRAY_1.size)
    return tvec * self.HORIZ_INTERVAL + self.HORIZ_OFFSET
  
  def ReturnRecordTime(self):
    time = self.RECORD_TIME
    print 'Time Bin Empty? ',time
    return time
   
  def RealGetNextWaveArray(self,ind): #memory leak
    #time = self.GetNextWaveArrayTime()[ind]
    data = self.GetNextWaveArrayData()[ind]
    return data#,time

  def RealGetNextWaveArrayDataAndTime(self,ind): #memory leak
    time = self.GetNextWaveArrayTime()[ind]
    data = self.GetNextWaveArrayData()[ind]
    return data,time

  def JustAllGetNextWaveArrayDataAndTime(self): #just return raw
    time = self.WAVE_ARRAY_1_timeself
    data = self._WAVE_ARRAY_1
    return data,time



  def GetNextWaveArrayTime(self):
    #returns the reshaped time of a Waveformarray, 1 entry = 1 Waveform
    #print self.WAVE_ARRAY_1_timeself.shape
    self.watime = self.WAVE_ARRAY_1_timeself.reshape(self.SUBARRAY_COUNT,self.WAVE_ARRAY_COUNT/self.SUBARRAY_COUNT)
    #print self.watime.shape
    return self.watime

  def GetNextWaveArrayData(self):
    #returns a reshaped Waveformarray, 1 entry = 1 Waveform
    #print self._WAVE_ARRAY_1.shape
    self.wa = self._WAVE_ARRAY_1.reshape(self.SUBARRAY_COUNT,self.WAVE_ARRAY_COUNT/self.SUBARRAY_COUNT)
    #print self.wa.shape
    return self.wa


  def SliceNextWaveArray(self,seg_count):

    data = self._WAVE_ARRAY_1[pointspreseg*seg_count:pointspreseg*seg_count+1:]
    time = self.WAVE_ARRAY_1_timeself[pointspreseg*seg_count:pointspreseg*seg_count+1:]
    return data,time










    #return _WAVE_ARRAY_1




# THIS nearly WONT WORK YET___________________________________
  def GetNextWaveArray(self):
    #self.a_WAVE_ARRAY_1 =  self.a_WAVE_ARRAY_1 + self._WAVE_ARRAY_1_SIZE
    #print "Address = " + str(self.a_WAVE_ARRAY_1)
    #wa = self.read_wave_array( self.a_WAVE_ARRAY_1)
    #if wa > 0:
    #self._WAVE_ARRAY_1 = wa
    #print "---> Read " + len(wa)
    #return true
    #else
    #   self._WAVE_ARRAY_1 = {} #set to 0
    #
    #print "bytes per array = " + str(self.bytes_per_wave)
    #start = self.Index*self.bytes_per_wave
    #end = self.Index*self.bytes_per_wave +self.bytes_per_wave
    #print "start " + str(start) + " end " + str(end)

    #wa = self._WAVE_ARRAY_1[0][start:end]

    #print wa

    #self.Index += 1
    #if self.Index > 65000-1: return False
    #return True

  #return false
    print "Segments = " + str(self.SUBARRAY_COUNT)
    ind = self.SUBARRAY_COUNT
    print 'ind ' + str(ind)
    start = 0
    end = 0
    print 'INDEX= ' + str(self.Index)
    #  
    #  start = int(self.Index * 127)
    #  end = start + 127
    #  step = 1
    #  print start
    #  print end
    #  print step
    print "Segments = " + str(self.SUBARRAY_COUNT)
    #print "Shape Test here",self._WAVE_ARRAY_1.shape[1]/127
    #dan: working self._WAVE_ARRAY_1 = self._WAVE_ARRAY_1.reshape(self._WAVE_ARRAY_1.shape[1]/127,127)
    
    self.wa = self._WAVE_ARRAY_1.reshape(self.SUBARRAY_COUNT,self.WAVE_ARRAY_COUNT)
    self.watime = self.WAVE_ARRAY_1_time.reshape(self.SUBARRAY_COUNT,self.WAVE_ARRAY_COUNT) 
    #wa = self._WAVE_ARRAY_1
      
    #print type(self._WAVE_ARRAY_1)
    # print self._WAVE_ARRAY_1.size
    
    for self.Index in range(0,ind):
      
      #print self._WAVE_ARRAY_1[0]
      #print self._WAVE_ARRAY_1[1]
     
      print 'IINDEX= ' + str(self.Index)
      print ' data' 
      print self.wa[self.Index]
      print 'time'
      print self.watime[self.Index]
      #return self._WAVE_ARRAY_1[self.Index]
      
      #time.sleep(1)

    #return self._WAVE_ARRAY_1

   #if self.Index > ind-1: return False
   #return True
   #return false


  @property
  def sampling_frequency(self):
    return 1/self.HORIZ_INTERVAL

  @property
  def LOFIRST(self):
    return not self.HIFIRST

  @property
  def HIFIRST(self):
    return self.COMM_ORDER == 0

  @property
  def WAVE_ARRAY_1(self):
    return self._WAVE_ARRAY_1

  @property
  def WAVE_ARRAY_1_time(self):
    """
    A calculated array of when each sample in wave_form_1 was measured,
    based on HORIZ_OFFSET and HORIZ_INTERVAL.
    """
    tvec = np.arange(0, self._WAVE_ARRAY_1.size)
    return tvec * self.HORIZ_INTERVAL + self.HORIZ_OFFSET

  @property
  def metadata(self):
    """
    Returns a dictionary of metadata information.
    """
    metadict = dict()
    for name, value in vars(self).items():
      if not name.startswith('_') and name.isupper():
        metadict[name] = getattr(self, name)

    return metadict

  @property#_________________________________propably need to look into this, how is WF data(_WAVE_ARRAY_1) and time (_WAVE_ARRAY_1_time) connected
  def mat(self):
    x = np.reshape(self.WAVE_ARRAY_1_time, (-1, 1))
    y = np.reshape(self.WAVE_ARRAY_1, (-1, 1))

    return np.column_stack((x,y))

  @property
  def comments(self):
    keyvaluepairs=list()
    for name, value in self.metadata.items():
      keyvaluepairs.append('%s=%s'%(name, value))
    return keyvaluepairs
  def savecsv(self, csvfname):
    """
    Saves the binary waveform as CSV, with metadata as headers.
    The header line will contain the string
      "LECROY BINARY WAVEFORM EXPORT"
    All headers will be prepended with '#'
    """
    mat = self.mat
    metadata = self.metadata
    jmeta = dict()
    for name, value in metadata.items():
      jmeta[name] = str(value)

    jmeta['EXPORTER'] = 'LECROY.PY'
    jmeta['AUTHOR'] = '@freespace'

    import json
    header = json.dumps(jmeta, sort_keys=True, indent=1)

    np.savetxt(csvfname, mat, delimiter=',', header=header)

  def _make_fmt(self, fmt):
    if self.HIFIRST:
      return '>' + fmt
    else:
      return '<' + fmt

  def _read(self, addr, nbytes, fmt):
    self.fh.seek(addr)
    s = self.fh.read(nbytes)
    fmt = self._make_fmt(fmt)
    return np.fromstring(s, fmt)[0]

  def read_byte(self, addr):
    return self._read(addr, 1, 'u1')

  def read_word(self, addr):
    return self._read(addr, 2, 'i2')

  def read_enum(self, addr):
    return self._read(addr, 2, 'u2')

  def read_long(self, addr):
    return self._read(addr, 4, 'i4')

  def read_float(self, addr):
    return self._read(addr, 4, 'f4')

  def read_double(self, addr):
    return self._read(addr, 8, 'f8')

  def read_string(self, addr, length=16):
    return self._read(addr, length, 'S%d'%(length))

  def read_timestamp(self, addr):
    second  = self.read_double(addr)
    addr += 8 # double is 64 bits = 8 bytes

    minute  = self.read_byte(addr)
    addr   += 1

    hour    = self.read_byte(addr)
    addr   += 1

    day     = self.read_byte(addr)
    addr   += 1

    month   = self.read_byte(addr)
    addr   += 1

    year    = self.read_word(addr)
    addr   += 2

    from datetime import datetime
    s = int(second)
    us = int((second - s) * 1000000)
    return datetime(year, month, day, hour, minute, s, us)

  def read_processing_done(self, addr):
    v = self.read_enum(addr)
    processsing_desc = ['no_processing',
                        'fir_filter',
                        'interpolated',
                        'sparsed',
                        'autoscaled',
                        'no_result',
                        'rolling',
                        'cumulative']
    return processsing_desc[v]

  def read_record_type(self, addr):
    v = self.read_enum(addr)
    record_types = ['single_sweep',
                    'interleaved',
                    'histogram',
                    'graph',
                    'filter_coefficient',
                    'complex',
                    'extrema',
                    'sequence_obsolete',
                    'centered_RIS',
                    'peak_detect']
    return record_types[v]


  # WE NEED TO UPDATE THIS___________________________________________________________-
  def read_wave_array(self, addr):
    self.fh.seek(addr) #______Need to check this is ok... ie when it reads past end of file - what happens
    s = self.fh.read(self._WAVE_ARRAY_1_SIZE) #______only do this is the above command was ok, otherwise return a blank array
    #print 'binary ',len(s)
    nsamples = self._WAVE_ARRAY_1_SIZE
    if self.COMM_TYPE == 0:
      fmt = self._make_fmt('i1')
    else:
      fmt = self._make_fmt('i2')
      # if each sample is a 2 bytes, then we have
      # half as many samples as there are bytes in the wave
      # array
      nsamples /= 2
    #print fmt
    #print nsamples
    dt = np.dtype((fmt, nsamples)) # combines 2 bytes into the correct format
    data1 = np.fromstring(s, dtype=dt) # reads binary data into float

    data = data1[0]

    # as per documentation, the actual value is gain * data - offset

    #from IPython import embed;embed();1/0 

    #print 'data shape ', data.shape
    #print 'data type ',type(data)

    #tvec = np.arange(0, data.size)



    return self.VERTICAL_GAIN * data - self.VERTICAL_OFFSET

if __name__ == '__main__':
  import sys
  fname = sys.argv[1]
  bwf = LecroyBinaryWaveform(fname)
  #raw_data = bwf.GetNextWaveArray()

  #raw_data = bwf.GetNextWaveArray()
  #raw_data = bwf.GetNextWaveArray()
  #raw_data = bwf.GetNextWaveArray()
  #print 'sampling freq=',bwf.sampling_frequency/1e6, 'MHz'

  #bwf.savecsv(sys.argv[2])
