"""
Job handler.

Queue up jobs and excecute them asynchronously. 
Used to make xmlrpc calls asynchronous.

"""
import logging
from threading import Thread, Lock
from Queue import Queue,Empty
from pylib.sysutils import LockableSet,LockedDict
from time import sleep
from pylib.uuid import uuid4 as uuid

class asyncJobs(Thread):
    def __init__(self):
        self.log = logging.getLogger('jobs')
        self.jobs = LockedDict()
        self.state = LockableSet()
        Thread.__init__(self)
        self.jobQueue = Queue()  
        
        self.start()

    def shutdown(self):
        self.state.Include('shutdown')
        while self.state.HasSignal('running'):
            sleep(0.5)
        self.log.warning('Terminated')
        
    def run(self):
        self.state.Include('running')
        while not self.state.HasSignal('shutdown'):
            if self.state.HasSignal('pause'):
                sleep(0)
                continue
                
            try:
                id,cls,func,args,kwargs = self.jobQueue.get(False)                
                self.jobs[id] = 'executing'
                func(cls,*args,**kwargs)
                del self.jobs[id]
            except Empty:
                if len(self.jobs) == 0:
                    self.state.Exclude('busy')                   
                continue
            except Exception,inst:
                self.log.exception(inst)
                self.state.Include('shutdown')
        self.state.Exclude('running')
        
    def add(self,cls,callback,*args,**kwargs):
        id = str(uuid())
        # add to job list before busy is set because of the way busy is cleared in run function
        self.jobs[id] = 'queued'
        self.state.Include('busy')
        self.log.debug('Id: %s' % (id))
        self.jobQueue.put((id,cls,callback,args,kwargs))
        return id
        
    def pause(self,on):
        if on:
            self.state.Include('pause')
        else:
            self.state.Exclude('pause')
        
    def busy(self):
        return self.state.HasSignal('busy')
        
    def status(self,id):
        try:
            res = self.jobs[id]
            if res is None:
                return 'done'
            else:
                return res
        except:
            return 'done'
        
        
# Test routines

def testJobs():
    from sysutils import printRefCounts
    from msvcrt import kbhit
    
    class job:
        def func(self,msg):
            print msg
            
    def job1(msg):
        print msg
        print 'Going to sleep for 5 secs'
        sleep(5)
        
    def job2(n):
        print n
        
    jh = asyncJobs()
    while not kbhit():
        jh.pause(True)
        for n in range(10):
            id = jh.add(job,job().func,'hello world')
            print jh.status(id)
            jh.add(None,job2,n)
        sleep(2)
        jh.pause(False)
        while jh.busy():
            sleep(0)
        # printRefCounts()
        sleep(2)
        
    id = jh.add(job1,'Time consuming task')
    print jh.busy()
        
    while jh.busy():
        print 'waiting...'
        print jh.status(id)
        sleep(1)
    jh.shutdown()
    
    
    
        
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    
    testJobs() #Not working because cls was added to the parameters (JJ 20071108)
