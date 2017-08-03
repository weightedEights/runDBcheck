"""
base proxy
All proxies should inherit from this.

"""

import logging
from xmlrpclib import ServerProxy

class directory(ServerProxy):
    def __init__(self,url):
        ServerProxy.__init__(self,url)
        self.log = logging.getLogger('directory')
        while True:  ## Hang around until we can talk to directory service
            try:
                self.ident()
                break;
            except:
                self.log.info('Waiting for directory service')
                sleep(5)
        
    def connect(self,service):
        url = self.get(service,'url')
        hold = self.get(service,'hold')
        sv = ServerProxy(url)

        while hold:
            try:
                sv.ident()
                break
            except:
                self.log.info('Waiting for service: %s' % (service))
                sleep(5)              
        return sv 

class baseProxy:
    def __init__(self,name,experiment):
        self.log = logging.getLogger(name)
        
        self.name = name
        self.exp = experiment
        
        self.log.info('Base initialized')
        
    def setup(self,n):
        self._setup = n
        
        
proxy = baseProxy
