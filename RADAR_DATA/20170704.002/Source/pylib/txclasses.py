"""
Classes concerned with TX configuration information

"""

import logging
import math


# Misc helper functions
def inv32(num):
    return ~num & 0xffffffff
        

class txConfig:
    def __init__(self,file,refclk=50e6):
        self.log = logging.getLogger('txConfig')
        self.file = file
        self.refClk = refclk
        self.image = [0,0,0,0,0,0,0]
        self.mem = [0 for n in range(28)]
        
        self.parse(file)
        
    def loadImageFromMem(self):
        indx = 0
        for n,v in enumerate(self.image):
            self.image[n] = self.mem[indx+3]*0x1000000+self.mem[indx+2]*0x10000+ \
                            self.mem[indx+1]*0x100 + self.mem[indx]
            indx += 4
            
    def setMem(self,addr,mask,value):
        # find least significat set bit in mask
        for n in range(8):
            chk = int(math.pow(2,n))
            if (chk & mask) == chk:
                break
        self.mem[addr] &= inv32(mask) ## Clear masked bits
        self.mem[addr] |= (value << n) & mask  ## or in new ones
        
    def setFreq(self,profile,freq):
        addr = [0x02,0x08,0x0e,0x14][profile]
        mul = self.mem[0] & 0x1f
        if mul == 1:
            sysclk = self.refClk
        else:
            sysclk = self.refClk * mul
        tw = long(freq*math.pow(2,32)/sysclk)
        self.setMem(addr,0xff,(tw & 0xff))
        self.setMem(addr+1,0xff,(tw & 0xff00)>>8)
        self.setMem(addr+2,0xff,(tw & 0xff0000)>>16)
        self.setMem(addr+3,0xff,(tw & 0xff000000)>>24)
        
    def setScale(self,profile,scale):
        addr = [0x07,0x0d,0x13,0x19][profile]
        t = scale
        if t > 1.9922:
            t = 1.9922
        elif t < math.pow(2,-7):
            t = math.pow(2,-7)

        by = int(round(t/math.pow(2,-7)))
        self.setMem(addr,0xff,by)        
        
    def parse(self,file):
        fm = file.getint('Common','SysClkMultiplier',4)
        self.setMem(0x00,0x1f,fm)
        
        pll = file.getint('Common','PllLockControl',1)
        self.setMem(0x00,0x20,pll)
        
        mode = file.getint('Common','Mode',0)
        self.setMem(0x01,0x03,mode)
        
        apd = file.getint('Common','AutoPwrDown',0)
        self.setMem(0x01,0x04,apd)
        
        fullsleep = file.getint('Common','SleepMode',0)
        self.setMem(0x01,0x08,fullsleep)
        
        invsinc = file.getint('Common','InvSincBypass',1)
        self.setMem(0x01,0x40,invsinc)
        
        cicclr = file.getint('Common','CicClear',0)
        self.setMem(0x01,0x80,cicclr)

        for i in range(4):
            p = 'Profile%d' % (i)
            freq = file.getfloat(p,'TuningFreq',0.0)
            self.setFreq(i,freq)
            
            scale = file.getfloat(p,'ScaleFactor',1.0)
            self.setScale(i,scale)
            
            cicrate = file.getint(p,'CicIntRate',1)
            self.setMem([0x06,0x0c,0x12,0x18][i],0xfc,cicrate)

            si = file.getint(p,'SpectralInv',0)
            self.setMem([0x06,0x0c,0x12,0x18][i],0x02,si)
            
            icb = file.getint(p,'InvCicBypass',1)
            self.setMem([0x06,0x0c,0x12,0x18][i],0x01,icb)
        self.loadImageFromMem()

        
if __name__ == '__main__':
    import sys
    sys.path.append('//dtc0/radac/pydaq/pylib')
    
    from Inifile import Inifile
    
    f = Inifile('//dtc0/radac/daq/setup/john/pyshellspeed/dtc0.txc')
    txc = txConfig(f)
    
    print [hex(n) for n in txc.image]
