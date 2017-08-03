"""
This module brings in the one instance of the
radac needed various places in the shell program
for access to the RADAC.

History:
    Initial implementation
    Date:       20070115
    Author:     John Jorgensen
    Company:    SRI International
    
"""

develop = 0

import logging
import math
import time as mtime
from pylib.sysutils import Signal

if develop:
    import radacDummy as rd
else:
    import radacDev.release.radacDev as rd
    
#Constants (for now!)
REFCLK=50e6

# Constants
RAMSIZE = 16777216

# Memory definition constants
MEMTU   = 0
MEMREG  = 1
MEMTM   = 2
MEMBC   = 3
MEMBC1  = 4

# Register constants
rcControl       = 0
rcSampleDiv     = 1
rcWrap          = 2
rcOutput        = 3
rcNHeaderWords  = 4
rcPulseCount    = 5
rcVerDate       = 6
rcVerNum        = 7
rcFrameCount    = 8
rcIppMask       = 9
rcIntMask       = 10
rcIntStatus     = 11
rcOutMask       = 12
rcRfAttn        = 14
rcTimeStatus    = 20
rcTimeMsw       = 21
rcTimeLsw       = 22
rcCntlWidth     = 23
rcPulseInc      = 24

REGS = {'control':rcControl,'sampledivider':rcSampleDiv,'wrap':rcWrap,\
        'output':rcOutput,'nheaderwords':rcNHeaderWords,'pulsecount':rcPulseCount,\
        'versiondate':rcVerDate,'versionnumber':rcVerNum,'framecount':rcFrameCount,'cntlwidth':rcCntlWidth,\
        'pulseinc':rcPulseInc,'interruptmask':rcIntMask,'interruptstatus':rcIntStatus,'rxattenuation':rcRfAttn,\
        'timestatus':rcTimeStatus,'timemsw':rcTimeMsw,'timelsw':rcTimeLsw,'ippmask':rcIppMask,'outmask':rcOutMask}

# Control register bit constants
crRunTu                 = 0x00000001
crRadacIntClkSel        = 0x00000002
crRadacIntTrgSel        = 0x00000004
crRadacSyncSel          = 0x00000008
crSampleEnable          = 0x00000010
crTuSwEnable            = 0x00000020
crTuClkEnable           = 0x00000040
crEnablePhaseFlip       = 0x00000080
crBeamCodeWrap          = 0x00000100
crEnableDmaIntr         = 0x00000200
crTestData              = 0x00000400
crEnableTx              = 0x00000800
crControlTr             = 0x00001000
crRfCmd                 = 0x00002000
crSyncRfTr              = 0x00004000
crHardOutputMask        = 0x00008000
crUseSpecialBeamcodes   = 0x00010000
crSyncRx                = 0x10000000
crHeaderEnable          = 0x20000000
crSwapIQ                = 0x40000000
crUseRx                 = 0x80000000

BITS = {'run':crRunTu,'internalclock':crRadacIntClkSel,'internaltrig':crRadacIntTrgSel,\
        'sync':crRadacSyncSel,'sampleenable':crSampleEnable,'specialwordcontrol':crTuSwEnable,'tuclkenable':crTuClkEnable,\
        'phaseflip':crEnablePhaseFlip,'beamcodewrap':crBeamCodeWrap,'dmainterrupt':crEnableDmaIntr,\
        'testdata':crTestData,'enabletx':crEnableTx,'controltr':crControlTr,'rfcmd':crRfCmd,'syncrftr':crSyncRfTr,'syncrx':crSyncRx,'headerenable':crHeaderEnable,\
        'swapiq':crSwapIQ,'userx':crUseRx,'hardoutputmask':crHardOutputMask,'usespecialbeamcodes':crUseSpecialBeamcodes,\
        crRunTu:'run',crRadacIntClkSel:'internalclock',crRadacIntTrgSel:'internaltrig',\
        crRadacSyncSel:'sync',crSampleEnable:'sampleenable',crTuSwEnable:'specialwordcontrol',crTuClkEnable:'tuclkenable',\
        crEnablePhaseFlip:'phaseflip',crBeamCodeWrap:'beamcodewrap',crEnableDmaIntr:'dmainterrupt',\
        crTestData:'testdata',crEnableTx:'enabletx',crControlTr:'controltr',crRfCmd:'rfcmd',crSyncRfTr:'syncrftr',crSyncRx:'syncrx',crHeaderEnable:'headerenable',\
        crSwapIQ:'swapiq',crUseRx:'userx',crHardOutputMask:'hardoutputmask',crUseSpecialBeamcodes:'usespecialbeamcodes'}


# Bit state
on  = True
off = False


# Misc helper functions
def inv32(num):
    return ~num & 0xffffffff
    
class sampleBuffer:
    def __init__(self,headersize=32):
        self.headerSize=headersize
        
        self.headerIn = None
        self.bufferIn = None
        self.headerOut = None
        self.bufferOut = None
        self.size = 0
        
    def setSize(self,size):
        self.headerIn = np.zeros(self.headerSize,dtype=np.uint32)
        self.headerOut = np.zeros(self.headerSize,dtype=np.uint32)
        self.bufferIn = np.zeros(size,dtype=np.complex64)
        self.bufferOut = np.zeros(size,dtype=np.complex64)
        self.size = size
        
    def swapBuffers(self):
        htmp = self.headerIn
        self.headerIn = self.headerOut
        self.headerOut = htmp
        
        btmp = self.bufferIn
        self.bufferIn = self.bufferOut
        self.bufferOut = btmp
        
class radacInterface:
    _txEnabled = False
    _rf = False
    
    def __init__(self,notify=[]):
        ## Create a logger
        self.log = logging.getLogger('radac')
        
        self.rxRefClk = REFCLK
        self.txRefClk = REFCLK
        
        
        # self.notify = notify
        self.notify = Signal()
        for func in notify:
            self.notify.connect(func)
        
        self.info = rd.deviceInfo()
        self.log.info('RADAC version: %x' % (self.info['versionnumber']))
        self.log.info('Version date: %i' % (self.info['versiondate']))
                
        # self.setTxEnabled(False)
        # self.setRf(False)     

    def shutdown(self):
        self.notify.disconnectAll()
        
    def register(self,func):
        self.notify.connect(func)
        # try:
            # names = [n for n,f in self.notify]
            # indx = names.index(name)
            # self.notify[indx] = (name,func)
        # except:
            # self.notify.append((name,func))
            
    def fireEvent(self,event,*lpars,**dpars):
        self.notify(event,*lpars,**dpars)
        # for name,func in self.notify:
            # func(event,*lpars,**dpars)
            
    def control(self,bit,state):
        if isinstance(bit,str):
            try:
                bit = BITS[bit]
            except:
                return 0
        cntl = rd.deviceReadRegisters(MEMREG,rcControl,1)[0]
        if (bit == 0) and state:
            self.fireEvent('lowlevelcheck')
        if state:
            cntl = cntl | bit
        else:
            cntl = cntl & inv32(bit)
        rd.deviceWriteRegisters(MEMREG,rcControl,[cntl])
        return 1
        
    def controlBitsOn(self):
        cntl = rd.deviceReadRegisters(MEMREG,rcControl,1)[0]
        res = []
        for n in range(32):
            v = pow(2,n)
            on = ((cntl & v) == v)
            if on:
                try:
                    res.append(BITS[v])
                except:
                    continue
        return res
                    
    def loadTu(self,buffer):
        self.fireEvent('lowlevelcheck',tuimage=buffer)
        rd.deviceWriteRegisters(MEMTU,0,buffer)
        
    def tuImage(self):
        res = []
        words = 0
        while words < RAMSIZE:
            buf = rd.deviceReadRegisters(MEMTU,words,1024)
            try:
                wrap = buf.index(0x80000000)
                res.extend(buf[0:wrap+1])
                return res
            except:
                msbs = [w for w in buf if ((w & 0x8000000) == 0x80000000)]
                if len(msbs) > 0:
                    return []
            res.extend(buf)
            words += 1024
        return []
        
    def rxEvent(self,addr,value):
        if addr == 0x303:
            f = value * self.rxRefClk / pow(2,32)
            self.fireEvent('rxfrequency',f)
                    
    def loadRx(self,config):
        for addr,value in config:
            rd.deviceWriteRx(addr,[value,0])
            self.rxEvent(addr,value)
        return 1
            
    def rxFrequency(self,value=None):
        if value is not None:
            val = value/self.rxRefClk
            ival = int(val)
            frac = val - ival
            tw = long(pow(2,32) * frac)
            self.loadRx([(0x303,tw)])
        else:
            v0,v1 = rd.deviceReadRx(0x303)
            f = v0 * self.rxRefClk / pow(2,32)
            return f
            
    def loadTx(self,image):
        # loadTx manipulates the whole Tx configuration
        # except full sleep (RF on/off) which is left alone
        
        self.fireEvent('lowlevelcheck',tximage=image)
        
        res = rd.deviceReadTx()
        fullsleep = ((res[0] & 0x00000800) == 0x00000800)
        if fullsleep:
            image[0] |= 0x00000800
        else:
            image[0] &= 0xfffff7ff
        rd.deviceWriteTx(image)
        self.fireEvent('txfrequencies',self.txFrequencies())
        return 1
        
    def txImage(self):
        return rd.deviceReadTx()
        
    def txFrequencies(self,value=None):
        conf = rd.deviceReadTx()
        confbytes = self.TxConfigBytes(conf)
        mul = conf[0] & 0x0000001f
        if mul == 1:
            sysclk = self.txRefClk
        else:
            sysclk = self.txRefClk * mul
        if value is not None:
            tw = [long(v*pow(2,32)/sysclk) for v in value]
            if value[0] >= 0.0:
                confbytes[2:6] = self.BytesFromWord(tw[0])
            if value[1] >= 0.0:
                confbytes[8:12] = self.BytesFromWord(tw[1])
            if value[2] >= 0.0:
                confbytes[14:18] = self.BytesFromWord(tw[2])
            if value[3] >= 0.0:
                confbytes[20:24] = self.BytesFromWord(tw[3])
            confw = self.TxConfigWords(confbytes)
            self.loadTx(confw)
        else:
            tw = [self.WordFromBytes(confbytes[2:6]),\
                  self.WordFromBytes(confbytes[8:12]),\
                  self.WordFromBytes(confbytes[14:18]),\
                  self.WordFromBytes(confbytes[20:24])]
            freq = [w * sysclk / pow(2,32) for w in tw]
            return freq
            
    def txScale(self,value=None):
        conf = rd.deviceReadTx()
        confbytes = self.TxConfigBytes(conf)
        if value is not None:
            b = [int(round(v/math.pow(2,-7))) for v in value]
            if b[0] >= 0.0:
                confbytes[0x07] = b[0]
            if b[1] >= 0.0:
                confbytes[0x0d] = b[1]
            if b[2] >= 0.0:
                confbytes[0x13] = b[2]
            if b[3] >= 0.0:
                confbytes[0x19] = b[3]
            confw = self.TxConfigWords(confbytes)
            self.loadTx(confw)
            
        sv = [confbytes[0x07],confbytes[0x0d],confbytes[0x13],confbytes[0x19]]
        scale = [b*math.pow(2,-7) for b in sv]
        return scale
            
                        
    def loadBeamcodes(self,codes):
        rd.deviceWriteRegisters(MEMBC,0,codes)
        return 1
        
    def loadDynamicBeamcodes(self,codes):
        rd.deviceWriteRegisters(MEMBC1,0,codes)
        return 1
        
    def readRegister(self,reg):
        if isinstance(reg,str):
            try:
                reg = REGS[reg]
                res = rd.deviceReadRegisters(MEMREG,reg,1)[0]
                return res
            except:
                return 0
        
    def writeRegister(self,reg,value):
        if isinstance(reg,str):
            try:
                r = REGS[reg]
                rd.deviceWriteRegisters(MEMREG,r,[value])
                if reg == 'control':
                    run = (value & 1)
                    if run:
                        self.fireEvent('lowlevelcheck')
                if reg == 'rxattenuation':
                    self.fireEvent('rxattenuation',value)
                return 1
            except:
                return 0
                
    def getTime(self,timetag=None):
        if timetag is None:
            ok = False
            cnt = 0
            while not ok and (cnt < 5):
                msw,lsw = rd.deviceReadRegisters(1,21,2)
                chk, = rd.deviceReadRegisters(1,22,1)
                s1 = lsw & 0xfff00000
                s2 = chk & 0xfff00000
                if s1 == s2:
                    ok = True
                cnt += 1
                # msw = self.readRegister('timemsw')
                # lsw = self.readRegister('timelsw')
                # chk = self.readRegister('timemsw')
                # if msw == chk:
                    # ok = True
                # cnt += 1
        else:
            ok = True
            msw = timetag[0]
            lsw = timetag[1]
            
        res = {}
        if ok:
            res['sync'] = ((0xf0000000 & msw) == 0)
            sec = int(0x0fffffff & msw) * 4096.0 + int((0xfff00000 & lsw) >> 20)
            usec = int((0x000fffff) & lsw);
            res['radactime'] = float(sec*1e6+usec)
            res['matlabtime'] = res['radactime']/86400e6+719529
            res['unixtime'] = res['radactime']/1e6
            structtime = mtime.gmtime(res['unixtime'])
            strtime = mtime.strftime('%Y-%m-%d %H:%M:%S',structtime)
            res['radactimestring'] = '%s.%.6i' % (strtime,usec)
        return res

    def setTime(self,usec):
        secs = int(usec/1e6);
        msw = long(secs / 2**32);
        secs = secs-msw*2**32;
        lsw = long(secs);
        self.writeRegister('timemsw',msw)
        self.writeRegister('timelsw',lsw) # when lsw is written the time is set on next 1pps pulse
                                          # or immediately if 1pps is not connected
        
    def setTxEnabled(self,flag):
        self.fireEvent('txenabled',flag)
        self.control(crEnableTx,flag)
        image = rd.deviceReadTx()
        if flag:
            image[0] &= 0xfffff7ff
        else:
            image[0] |= 0x00000800
        rd.deviceWriteTx(image)
        return 1
        
    def isTxEnabled(self):
        image = rd.deviceReadTx()
        rfena = not ((image[0] & 0x00000800) == 0x00000800)
        return rfena
                
    def setRf(self,flag):
        self.control('rfcmd',flag)
        return 1
        
    def isRfOn(self):
        image = rd.deviceReadTx()
        rfena = not ((image[0] & 0x00000800) == 0x00000800)
        cntl = self.readRegister('control')
        rfon = ((cntl & crRfCmd) == crRfCmd)
        return rfena and rfon
        
    def TxConfigBytes(self,config):
        res = range(28)
        for n,w in enumerate(config):
            indx = n*4
            res[indx] = w & 0x000000ff
            res[indx+1] = (w & 0x0000ff00) >> 8
            res[indx+2] = (w & 0x00ff0000) >> 16
            res[indx+3] = (w & 0xff000000) >> 24
        return res
        
    def TxConfigWords(self,bytes):
        words = range(7)
        for n,w in enumerate(words):
            bi = n*4
            words[n] = self.WordFromBytes(bytes[bi:bi+4])
        return words
        
    def WordFromBytes(self,bytes):            
        return bytes[3] * 0x1000000 + bytes[2] * 0x10000 + bytes[1] * 0x100 + bytes[0]
        
    def BytesFromWord(self,word):
        res = range(4)
        res[0] = word & 0x000000ff
        res[1] = (word & 0x0000ff00) >> 8
        res[2] = (word & 0x00ff0000) >> 16
        res[3] = (word & 0xff000000) >> 24
        return res
        

class xmlRadac: 
    def __init__(self):
        # Map functions straight through to radac instance
        self.radac = radacInterface()
        self.control = self.radac.control
        self.isRfOn = self.radac.isRfOn
        self.setRf = self.radac.setRf
        self.isTxEnabled = self.radac.isTxEnabled
        self.setTxEnabled = self.radac.setTxEnabled
        self.loadBeamcodes = self.radac.loadBeamcodes
        
            
        

        
        
