"""
shell proxies object for handling communication to external data sources

"""

import os
import logging
import re
import siteconfig


class proxies(list):
    def __init__(self,experiment):
        list.__init__(self)
        
        self.log = logging.getLogger('proxies')
        
        hosts = experiment.experimentConfig.vars('hosts')
        
        # load proxies
        
        # load site proxy if any
        p = self.proxy(siteconfig.sitename,experiment)
        if p is not None:
            self.append(p)
            
        # load host proxies         
        for n,v in hosts.items():
            p = self.proxy(n,experiment)
            if p is not None:
                self.append(p)
                
    def setup(self,number):
        for p in self:
            p.setup(number)
        
    def storeData(self,h5buffer,starttime,endtime,vars):
        for p in self:
            p.storeData(h5buffer,starttime,endtime,vars)
        
    def proxyName(self,name):
        pat = re.compile('[a-zA-Z]*')
        prx = pat.match(name).group()
        return prx.lower()
        
    def proxy(self,name,experiment):
        folder = siteconfig.sourceFolder('shell','proxies')
        files = os.listdir(folder)
        prxfiles = [n for n in files if n.endswith('.py')]
        modules = [os.path.splitext(m)[0] for m in prxfiles]
        srv = self.proxyName(name)
        self.log.debug('Looking for proxy: %s' % (srv))
        for m in modules:
            if srv in m.lower():
                try:
                    self.log.info('Loading %s for: %s' % (m,name))
                    try:
                        mod = __import__('shell.proxies.%s' % (m),{},{},['proxy'])
                    except Exception,inst:
                        self.log.error(inst)
                    prx = mod.proxy(experiment)
                    return prx
                except Exception,inst:
                    self.log.exception(inst)
                    return None
                