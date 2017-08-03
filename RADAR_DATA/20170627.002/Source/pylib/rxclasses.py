"""
Classes concerned with RX configuration information

"""

import logging
import os
import math
import numpy as np
import numpy.fft as fft
from pylib.Inifile import Inifile

C = 299792458.0
RXCOEFSCALE = 524287.0

def frac(number):
    return number-int(number)
    
def isNumber(s):
    try:
        clean = s.strip()
        n = float(clean)
        return True
    except:
        return False
    
    
class filter:
    def __init__(self,refclk,mcic2,mcic5,mrcf,filepath):
        self.log = logging.getLogger('filter')
        
        self.refClk = refclk
        self.mCic2 = mcic2
        self.mCic5 = mcic5
        self.mRcf = mrcf
        self.filePath = filepath
        
        self.inputSampleRate = self.refClk / self.mCic2 / self.mCic5
        self.outputSampleRate = self.inputSampleRate / self.mRcf
        
        self.loadCoefficientsFromFile(self.filePath)
    
    def loadCoefficientsFromFile(self,filepath):
        f = open(filepath,'r')
        lines = f.readlines()
        self.coefficients = [float(line.strip()) for line in lines if isNumber(line)]
        self.extractParameters()
        
    def extractParameters(self):
        self.nTaps = len(self.coefficients)
        self.log.info('nTaps: %d' % (self.nTaps))

        tclk = 1/self.refClk
        tsamp = tclk
        ad = (self.nTaps*self.mCic2*self.mCic5 + 4*self.mCic2*self.mCic5 - 3*self.mCic2 + 1)/2/self.refClk
        fd = 10*tclk + tsamp*(7+self.mCic2*(7+self.mCic5*(5+self.mRcf))) + self.nTaps*tclk

        self.delay = ad+fd
        self.log.info('Filter delay: %f' % (self.delay))
        self.rangeDelay = self.delay*C/2.0
        self.log.info('Filter range delay: %f' % (self.rangeDelay))

        data = np.zeros(4096,dtype=np.complex64)
        data.real[0:len(self.coefficients)] = self.coefficients
        
        fftdata = fft.fft(data)
        K = self.inputSampleRate/4096
        Bw = np.sum(K * np.absolute(fftdata*np.conjugate(fftdata)))
        self.bandwidth = Bw/2.0
        self.log.info('Filter bandwidth: %f' % (self.bandwidth))        
        

class rxConfig:
    def __init__(self,folder,filename,refclk=50e6):
        self.log = logging.getLogger('rxConfig')
        self.refClk = refclk
        self.filePath = os.path.join(folder,filename)
        self.file = Inifile(self.filePath)
        self.filterPath = os.path.join(folder,self.file.get('filter','file',''))
        self.config = self.parse(self.file)
        
    def parse(self,file):
        res = []
        res.append((int('0x300',16),1)) ## soft reset
        
        # Clear calculation memory
        addr = range(int('0x100',16),int('0x200',16),1)
        value = [0 for n in range(len(addr))]
        both = zip(addr,value)
        res.extend(both)
        
        # NCO control
        ncoc = file.getint('NCO','Bypass',0)
        ncoc = ncoc | (file.getint('NCO','PhaseDither',0) << 1)
        ncoc = ncoc | (file.getint('NCO','AmplitudeDither',0) << 2)
        res.append((int('0x301',16),ncoc))

        # Sync mask
        res.append((int('0x302',16),int('0xffffffff',16)))
        
        # NCO frequency
        ncof = 1e6*file.getfloat('NCO','Frequency',0.0)
        ncotw = int(round(frac(ncof/self.refClk)*math.pow(2,32)))
        res.append((int('0x303',16),ncotw))

        # Phase offset
        po = file.getfloat('NCO','PhaseOffset',0.0);
        pow = int(round(po/360.0*math.pow(2,16)))
        res.append((int('0x304',16),pow))

        # Scic2
        Scic2 = file.getint('Scaling','Scic2',4)
        res.append((int('0x305',16),Scic2))

        #  Mcic2
        Mcic2 = file.getint('Decimation','Mcic2',10)-1
        res.append((int('0x306',16),Mcic2))

        # Scic5
        Scic5 = file.getint('Scaling','Scic5',4)
        res.append((int('0x307',16),Scic5))

        # Mcic5
        Mcic5 = file.getint('Decimation','Mcic5',5)-1
        res.append((int('0x308',16),Mcic5))

        filtersamplerate = self.refClk/(Mcic2+1)/(Mcic5+1)

        # SRcf
        SRcf = file.getint('Scaling','SRcf',4)
        res.append((int('0x309',16),SRcf))

        # MRcf
        MRcf = file.getint('Decimation','MRcf',7)-1
        res.append((int('0x30a',16),MRcf))

        self.sampleRate = self.refClk / (Mcic2+1) / (Mcic5+1) / (MRcf+1)
        self.sampleTime = 1.0/self.sampleRate
        self.sampleSpacing = self.sampleTime*C/2;

        #  Address offset
        addroffset = 0
        res.append((int('0x30b',16),addroffset))

        #  Reserved
        res.append((int('0x30d',16),0))

        #  Mode control
        mc = file.getint('Mode','Input',0);
        mc = mc | (file.getint('Mode','Sync',0) << 1)
        res.append((int('0x300',16),mc))

        # Filter Parameters
        self.filter = filter(self.refClk,Mcic2+1,Mcic5+1,MRcf+1,self.filterPath)
        
        # Filter coefficients
        for addr,coef in enumerate(self.filter.coefficients):
            scaled = coef*RXCOEFSCALE
            rounded = int(round(scaled))
            masked = rounded & int('0x000fffff',16)
            res.append((addr,masked))
        
        # NTaps
        res.append((int('0x30c',16),self.filter.nTaps-1))
        
        res.append((int('0x300',16),0)) ## start running
        
        return res
        
        
