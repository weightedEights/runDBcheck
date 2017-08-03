"""
system check module
"""

from pylib.radac import *
from pylib.exceptionclasses import systemCheckError

radac = radacInterface()

# Default low level system limits
LIMITS = {'amisr':{'dutycycle':0.1,'maxpulsewidth':5000,'minipp':500,'trenveloperf':1,'beamcodetriggertotr':400},
          'sondrestrom':{'dutycycle':0.03,'maxpulsewidth':500,'minipp':1000,'trontorfon':10,'rfofftotroff':0}}
          
class tuImage:
    def __init__(self,image):
        self.image = image
        
        self.output = []
        self.specialword = []
        
        n = 0
        time = 0
        while n < len(self.image):
            delay = self.image[n]
            n += 1
            output = self.image[n]
            n += 1
            if self.bitSet(30,output):
                specialword = self.image[n]
                n += 1
                self.output.append((time,output))
                self.specialword.append((time,specialword))
            else:   
                self.output.append((time,output))
            time += delay
        self.frametime = time
                    
    def loaded(self):
        return (self.image <> [])
                
    def bitSet(self,bit,word):
        w = 2**bit
        return ((word & w) == w)
        
    def bitOn(self,bit):
        res = []
        st = 0
        for t,o in self.output:
            if self.bitSet(bit,o):
                if st == 0:
                    st = t
            else:
                if st <> 0:
                    res.append((st,t-st))
                    st = 0
        return res
        
    def specialWord(self,addr=0):
        res = []
        for t,sw in self.specialword:
            swa = (sw & 0x3fff0000) >> 16
            swv = (sw & 0x0000ffff)
            if swa == addr:
                res.append((t,swv))
        return res
                
        
        
    def bitUsed(self,bit):
        return (len([t for t,o in self.output if (self.bitSet(bit,o))]) > 0)
        
    def duty(self,bit=0):
        on = self.bitOn(0)
        ontime = 0.0
        for s,d in on:
            ontime += d
        return ontime/float(self.frametime)
        
        
        
        
class txImage:
    def __init__(self,image):
        self.image = image
        
        self.bytes = self.txConfigBytes(image)
        
    def txConfigBytes(self,config):
        res = range(28)
        for n,w in enumerate(config):
            indx = n*4
            res[indx] = w & 0x000000ff
            res[indx+1] = (w & 0x0000ff00) >> 8
            res[indx+2] = (w & 0x00ff0000) >> 16
            res[indx+3] = (w & 0xff000000) >> 24
        return res
        
    def wordFromBytes(self,bytes):            
        return bytes[3] * 0x1000000 + bytes[2] * 0x10000 + bytes[1] * 0x100 + bytes[0]
        
    def mode(self):
        return (self.bytes[0] & 3)
        
    def onProfiles(self):
        res = []
        addr = [2,8,14,20]
        for p in range(4):
            a = addr[p]
            tw = self.wordFromBytes(self.bytes[a:a+4])
            if tw > 0:
                res.append(p)
        return res
    
    def offProfiles(self):
        res = []
        addr = [2,8,14,20]
        for p in range(4):
            a = addr[p]
            tw = self.wordFromBytes(self.bytes[a:a+4])
            if tw == 0:
                res.append(p)
        return res
        
        
# Helper functions

def pulse(on,off):
    res = []
    for t in on:
        o = [et for et in off if et > t][0]
        d = o-t
        res.append((t,d))
    return res

def findDelay(time,times,frametime=0):
    for t in times:
        if t >= time:
            return t-time
    raise systemCheckError('findDelay: failed to find bigger time in list')
    
def extractRf(tu,tx):
    onp = tx.onProfiles()
    # if onp == []:
        # raise systemCheckError('extractRf: No on profiles in TX')
    offp = tx.offProfiles()
    if offp == []:
        raise systemCheckError('extractRf: No off profiles in TX')
    sw = tu.specialWord(0)
    prof = [None,None,None,None]
    for p in range(4):
        prof[p] = [t for t,v in sw if v == p]
    on = []
    off = []
    for n,p in enumerate(prof):
        if p <> []:
            # p.append(p[0]+tu.frametime)  #wrap
            for t in p:
                if n in onp:
                    on.append(t)
                elif n in offp:
                    off.append(t)
    on.sort()
    off.sort()
    
    rf = pulse(on,off)
    if len(rf) > 0:
        if rf[-1][0] > tu.frametime: # remove wrap
            del rf[-1]
    return rf
    
def overlap(start,stop,times):
    indx = 0
    while indx < len(times):
        st = times[indx][0]
        et = times[indx][0]+times[indx][1]
        if (st >= start) and (st <= stop):
            return indx
        elif (et > start) and (et <= stop):
            return indx
        elif (st <= start) and (et >= stop):
            return indx
        indx += 1
    return -1
            

def systemCheck(tuimage=None,tximage=None,limits={}):
    tuRun = (tuimage is None) and (tximage is None)
    
    if tuimage is None:
        tuimage = radac.tuImage()
   
    if tximage is None:
        tximage = radac.txImage()
    
    # Perform various checks
    
    # Tu file loaded
    if tuRun and tuimage == []:
        raise systemCheckError('No timing unit file loaded! Can not start system')
        
    tu = tuImage(tuimage)
    
    tx = txImage(tximage)
        
    if (tuRun and tu.bitUsed(5) and (tx.mode() <> 0)):
        raise systemCheckError('To use phase flipping TX needs to be in mode 0')
        
    if tu.loaded():
        # Tu related checks
        if limits.has_key('dutycycle'):
            dc = tu.duty(bit=0)
            limit = limits['dutycycle']
            if dc > limit:
                raise systemCheckError('TR dutycycle: %7.6f exceeded, limit: %7.6f' % (dc,limit))
                
        if limits.has_key('maxpulsewidth'):
            b0 = tu.bitOn(0)
            du = [d for s,d in b0]
            if len(du) > 0:
                mpw = max(du)
                limit = limits['maxpulsewidth']
                if mpw > limit:
                    raise systemCheckError('Pulsewidth: %d exceeded, limit: %d' % (mpw,limit))
                
        if limits.has_key('minipp'):
            b0 = tu.bitOn(0)
            st = [s for s,d in b0]
            if len(st) > 1:
                n = 1
                ipps = []
                while n < len(st):
                    d = st[n]-st[n-1]
                    ipps.append(d)
                    n += 1
                ipp = min(ipps)
            else:
                ipp = tu.frametime
            limit = limits['minipp']
            if ipp < limit:
                raise systemCheckError('Min IPP: %d exceeded, limit: %d' % (ipp,limit))
                
        if limits.has_key('beamcodetriggertotr'):
            limit = limits['beamcodetriggertotr']
            bct = tu.bitOn(12)
            if bct == []:
                raise systemCheckError('Beamcode trigger to TR: No beamcode triggers in TU file')
            b0 = tu.bitOn(0)
            if b0 == []:
                raise systemCheckError('Beamcode trigger to TR: No TR pulses in TU file')
            times = [s for s,d in b0]
            times.append(times[0]+tu.frametime) # wrap 
            for s,d in bct:
                delay = findDelay(s,times)
                if delay < limit:
                    raise systemCheckError('Min time between beamcode trigger and TR: %d violated, limit: %d' % (delay,limit))
        
        if limits.has_key('trenveloperf'):
            limit = limits['trenveloperf']
            rf = extractRf(tu,tx)
            tr = tu.bitOn(0)
            for s,d in tr:
                trs = s
                tre = s+d
                indx = overlap(trs,tre,rf)
                if indx <> -1:
                    rfs = rf[indx][0]-limit
                    rfe = rf[indx][0]+rf[indx][1]+limit
                    if rfs < trs:
                        raise systemCheckError('RF on time violated')
                    if rfe > tre:
                        raise systemCheckError('RF off time violated') 

        if limits.has_key('trontorfon'):
            limit = limits['trontorfon']
            rf = extractRf(tu,tx)
            tr = tu.bitOn(0)
            for s,d in tr:
                trs = s
                tre = s+d
                indx = overlap(trs,tre,rf)
                if indx <> -1:
                    rfs = rf[indx][0]-limit
                    if rfs < trs:
                        raise systemCheckError('RF on time violated')
                        
        if limits.has_key('rfofftotroff'):
            limit = limits['rfofftotroff']
            rf = extractRf(tu,tx)
            tr = tu.bitOn(0)
            for s,d in tr:
                trs = s
                tre = s+d
                indx = overlap(trs,tre,rf)
                if indx <> -1:
                    rfe = rf[indx][0]+rf[indx][1]+limit
                    if rfe > tre:
                        raise systemCheckError('RF off time violated') 
                        
        
        
if __name__ == '__main__':
    lim = LIMITS['amisr']
    # lim['maxpulsewidth'] = 60
    # lim['dutycycle'] = 0.01
    #lim['minipp'] = 15000
    #lim['beamcodetriggertotr'] = 500
    # lim['trontorfon'] = 1
    # lim['rfofftotroff'] = 1
    systemCheck(None,None,lim)
    print 'Done'
    