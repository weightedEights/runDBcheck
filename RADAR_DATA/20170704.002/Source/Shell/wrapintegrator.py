"""
This is a python wrapper class around the 
integrator extension.

"""

import logging
from Queue import Queue,Empty
import thread
from threading import Event

INTVER='release'

class wrapIntegrator:
    def __init__(self,exp):
        self.log = logging.getLogger('integrator')
        
        self.ext = __import__('integrator.%s.extIntegrator' % (INTVER),globals(),locals(),['extIntegrator'])
        self.ext.initialize(self,exp,logging.getLogger('extIntegrator'))
        
        # map some methods straight through to extension
        self.busy = self.ext.busy
        self.start = self.ext.start
        self.stop = self.ext.stop
        
        # event handling
        self.events = []
        self.threadShutdown = Event()
        self.eventQueue = Queue()
        thread.start_new_thread(self.eventThread,())
              
        self.config = {}
        
    def shutdown(self):
        self.threadShutdown.set()
        if self.config <> {}:
            self.log.info('shutdown')
            # shutdown extension. This should be done before modes are terminated if an integration is in progress!! 
            self.log.debug('Ext shutdown')
            self.ext.shutdown()
            # shutdown all modes
            self.log.info('Mode shutdown')
            for m in self.config['modes']:
                m.shutdown()
        self.config.clear() 
        
    def registerEvent(self,event,callback):
        self.events.append((event,callback))
        
    def eventThread(self):
        while not self.threadShutdown.isSet():
            try:
                event = self.eventQueue.get(True,0.1)
                for ev,func in self.events:
                    if ev == event:
                        func(event)
                    elif ev == event[1:]:
                        func(event)
            except Empty:
                continue
            except Exception,inst:
                self.log.exception(inst)
        self.log.info('eventThread shutdown')        
    
        
    def integratorEvent(self,event):
        self.eventQueue.put(event)

    def resetModes(self):
        if self.config.has_key('modes'):
            for m in self.config['modes']:
                try:
                    m.shutdown()
                    m = None
                except Exception,inst:
                    self.log.exception(inst)
            del self.config['modes']
            
    def configure(self,config={}):
        self.ext.configure(config)
        self.config = config
        
        
        