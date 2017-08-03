#-----------------------------------------------------------------------------
# Name:        Timer.py
# Purpose:     
#
# Author:      John Jorgensen
#
# Created:     2005/08/06
# RCS-ID:      $Id: Timer.py $
# Copyright:   SRI International
#-----------------------------------------------------------------------------

import threading
from pylib.RTime import LTime
from pylib.sysutils import Signal
from time import sleep

class Timer(threading.Thread):
    def __init__(self,owner=None,callback=None,interval=1,enabled=True):
        threading.Thread.__init__(self)
        self.Lock = threading.Lock()
        self._run = True
        self.Owner = owner
        self.Time = LTime()
        self.OnTimeout = Signal()
        self.OnTimeout.connect(callback)
        self.Interval = interval
        self.Enabled = False
        self.Enable(enabled)
        self.start()
        
    def Kill(self):
        self.OnTimeout.disconnectAll()
        self.Lock.acquire()
        self._run = False
        self.Enabled = False
        self.Lock.release()
        
    def Enable(self,enable=True):
        self.Lock.acquire()
        self.Enabled = enable
        if enable:
            self.Time() #Reset time
        self.Lock.release()
            
    def SetInterval(self,interval):
        self.Lock.acquire()
        self.Interval = interval
        self.Lock.release()
        
    def run(self):
        _run_=self._run
        try:
            while _run_:
                self.Lock.acquire()
                ena = self.Enabled
                self.Lock.release()
                while ena:
                    self.Lock.acquire()
                    elapsedtime = self.Time.Elapsed()
                    interval = self.Interval
                    self.Lock.release()
                    if elapsedtime >= interval:
                        self.OnTimeout(self)
                        self.Lock.acquire()
                        self.Time() #resets time
                        self.Lock.release()
                    else:
                        sleep(0)
                    self.Lock.acquire()
                    ena = self.Enabled
                    self.Lock.release()
                sleep(0)
                self.Lock.acquire()
                _run_ = self._run
                self.Lock.release()
        except:
            raise
            
if __name__ == '__main__':

    def TimeOut(timer):
        print 'Timer timed out'
        
    t = Timer(None,TimeOut,3,True)
    t.Enable()
    while True:
        try:
            sleep(0)
        except:
            print 'Killing timer'
            t.Kill()
            break
            
        

    
     
