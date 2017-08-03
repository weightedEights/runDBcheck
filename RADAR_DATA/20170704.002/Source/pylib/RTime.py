#-----------------------------------------------------------------------------
# Name:        RTime.py
# Purpose:     Radar time handling routines 
#
# Author:      John Jorgensen
#
# Created:     2005/01/06
# RCS-ID:      $Id: module1.py $
# Copyright:   SRI International
#-----------------------------------------------------------------------------


from time import *
from threading import Lock

class BTime:
    def __init__(self,time=None):
        self._time = None
        self._lock = Lock()
        self._clk = clock()
        self.__call__(time)
        self.Months = ['Non','Jan','Feb','Mar','Apr','May','Jun',
                       'Jul','Aug','Sep','Oct','Nov','Dec']
    def __str__(self):
        return self.Str()
    
    def __call__(self,time=None):
        self._clk = clock()
        if time:
            self._time = time
        else:
            self._time = self.Now()

        
    def __add__(self,other):
        if isinstance(other,BTime):
            return self._time+other._time
        else:
            return self._time+other
    
    def __sub__(self,other):
        if isinstance(other,BTime):
            return self._time-other._time
        else:
            return self._time-other
            
    def __float__(self):
        return self._time
            
    def Lock(self):
        self._lock.acquire()
        
    def Release(self):
        self._lock.release()
        
    def Now(self):
        t = self.LinearTime(localtime())
        return t
    
    def Elapsed(self,ret='Seconds'):
        if ret == 'Days':
            return (clock()-self._clk)/86400.0
        elif ret == 'Hours':
            return (clock()-self._clk)/3600.0
        elif ret == 'Minutes':
            return (clock()-self._clk)/60.0
        elif ret == 'Seconds':
            return (clock()-self._clk)
        
    def LinearTime(self,time=[]):
        if len(time) == 2:        #Radar time input
            adate,atime = time
            td = adate
            cent = td / 1000000L
            td -= cent * 1000000L
            year = td / 10000L
            td -= year * 10000L
            month = td / 100L
            td -= month * 100L
            day = td
            
            tt = atime
            hour = tt / 100000L
            tt -= hour * 100000L
            min = tt / 1000L
            tt -= min * 1000L
            sec = tt / 10L
            tt -= sec * 10L
            sec10 = tt
        elif len(time) == 9:      #Unix time struct
            y = time[0]
            cent = y / 100L
            year = y-cent*100L
            month = time[1]
            day = time[2]
            hour = time[3]
            min = time[4]
            sec = time[5]
            sec10 = 0
        
        if (month > 2): 
            month -= 3 
        else: 
            month += 9 
            if (year): 
                year -= 1 
            else: 
                year = 99 
                cent  += 1 
        ldays = ((146097L * cent)    / 4L + 
              (1461L   * year)       / 4L + 
              (153L    * month + 2L) / 5L + 
              day   + 1721119L) 
        days = ldays
        
        fracday = ((hour*36000.0)+(min*600.0)+(sec*10.0)+(sec10*1.0)) / 864000.0
        return days+fracday 
    
    def RadarTime(self,lintime):
        dt = lintime
        days = long(dt)
        fracday = dt - (days * 1.0)
        tenths = 864000 * fracday
        
        days   -= 1721119L 
        cent = (4L * days - 1L) / 146097L 
        days    =  4L * days - 1L  - 146097L * cent 
        day     =  days / 4L 
        
        year    = (4L * day + 3L) / 1461L 
        day     =  4L * day + 3L  - 1461L * year 
        day     = (day + 4L) / 4L 
        
        month   = (5L * day - 3L) / 153L 
        day     =  5L * day - 3   - 153L * month 
        day     = (day + 5L) / 5L 
        
        if (month < 10): 
            month += 3 
        else: 
            month -= 9; 
            year += 1
            if (year == 99): 
                year = 0 
                cent += 1 
        
        hour = long(tenths) / 36000L
        tenths -= hour * 36000L
        min = long(tenths) / 600L
        tenths -= min * 600L
        sec = long(tenths) / 10L
        tenths -= sec * 10L
        sec10 = long(tenths)
        
        date = long((cent * 1000000L)+(year * 10000L)+(month * 100L)+(day))
        time = long((hour * 100000L)+(min * 1000L)+(sec * 10L)+(sec10)) 
        return [date,time]
               
    def Decode(self,atime=None):
        if not atime is None:
            t = atime
        else:
            t = self.Time
        date,time = self.RadarTime(t)
        y = date / 10000L
        date -= y * 10000L
        m = date / 100L
        date -= m * 100L
        d = date
        
        h = time / 100000L
        time -= h * 100000L
        n = time / 1000L
        time -= n * 1000L
        s = time / 10L
        time -= s * 10L;
        s10 = time
        return [y,m,d,h,n,s,s10]
    
    def Str(self,time=None,fmt='%y4%m2%d2 %h2:%n2:%s2.%f'):
        if not time is None:
            t = time
        else:
            t = self._time
        [y,m,d,h,n,sec,s10] = self.Decode(t)
        f = fmt.split('%')
        res = ''
        for s in f:
            if s == '':
                continue
            e = s[2::]
            if s[0] == 'y':
                if s[1] == '4':
                    res += str(y)
                elif s[1] == '2':
                    c = y/100L
                    y -= c*100L
                    sy = str(y)
                    if len(sy) == 1:
                        sy = '0'+sy
                    res += sy
            elif s[0] == 'm':
                if s[1] == '2':
                    sm = str(m)
                    if len(sm) == 1:
                        sm = '0'+sm
                    res += sm
                elif s[1] == '3':
                    res += self.Months[m]
            elif s[0] == 'd':
                if s[1] == '2':
                    sd = str(d)
                    if len(sd) == 1:
                        sd = '0'+sd
                    res += sd
            elif s[0] == 'h':
                if s[1] == '2':
                    sh = str(h)
                    if len(sh) == 1:
                        sh = '0'+sh
                    res += sh
            elif s[0] == 'n':
                if s[1] == '2':
                    sn = str(n)
                    if len(sn) == 1:
                        sn = '0'+sn
                    res += sn
            elif s[0] == 's':
                if s[1] == '2':
                    ss = str(sec)
                    if len(ss) == 1:
                        ss = '0'+ss
                    res += ss
            elif s[0] == 'f':
                res += str(s10)
            res += e
        return res

    def LoadTime(self,t,fmt='hhnnss'):
        it = int(t)
        if fmt == 'hhnnss':
            h = int(it / 10000)
            it = it - h * 10000
            n = int(it / 100)
            s = it - n * 100
            self._time = (float(h) / 24.0)+(float(n) / 1440) + (float(s) / 86400.0)
            
            
                
                
       
class M2Time(BTime):
    def __init__(self,time):
        self.BoiTime = self.LinearTime(time[0:2])
        self.EoiTime = self.LinearTime(time[2:4])
        self.IntegrationTime = self.EoiTime-self.BoiTime
        BTime.__init__(self,self.BoiTime)
        
class LTime(BTime):
    def __init__(self,time=None):
        BTime.__init__(self,time)
        
class RTime(BTime):
    def __init__(self,time):
        BTime.__init__(self,self.LinearTime(time))
        
        
if __name__=='__main__':
    t = LTime()
    print t
    
    
    
        
        