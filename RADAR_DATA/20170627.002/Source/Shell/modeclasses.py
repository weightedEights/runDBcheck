"""
Mode handling classes

"""

import logging
import os
import numpy as np

EXTVERSION = ['release','Release','debug','Debug']

class mode:
    def __init__(self,h5buffer,pars):
        self.h5Buffer = h5buffer
        self.pars = pars
        self.index = -1 ## Set in mode extension  >= 0
        
        ## various attributes
        self.rootPath = '' ## Will be set in mode extension
        
        ## Get logger
        self.log = logging.getLogger('%s/%s' % (pars['mode'],pars['name']))
        self.pars['log'] = self.log ## Make sure it gets passed down to mode extension
        try:
            # Deal with common parameters
            
            # pulsewidth and txbaud
            pw = 0.0
            if self.pars.has_key('pulsewidth'):
                pw = float(self.pars['pulsewidth'])
                # self.log.info('Pulsewidth: %i' % (pw))
                pw *= 1e-6 # pulsewidth in seconds
                nppw = np.array(pw,dtype=np.float32)
                self.log.info('Pulsewidth: %f' % (pw))
            else:
                self.log.error('Pulsewidth not specified')
            txb = pw
            if self.pars.has_key('baud'):
                txb = pw/float(self.pars['baud'])
            elif self.pars.has_key('codelength'):
                txb = pw/float(self.pars['codelength'])
            elif self.pars.has_key('txbaud'):
                txb = float(self.pars['txbaud'])
                txb *= 1e-6
                
            self.pars['txbaud'] = str(txb*1e6) # value in uSec to mode dlls.
            nptxb = np.array(txb,dtype=np.float32)
            self.log.info('TxBaud: %f' % (txb))
                                  
                
            m = self.pars['mode']
            for extver in EXTVERSION:
                modepath = '%s.%s.%s.%s' % ('modes',m,extver,m)
                try:
                    self.ext = __import__(modepath,globals(),locals(),[m])
                    break
                except:
                    continue
                    
            self.dllpath = self.ext.__file__
            self.ext.configure(self)
            self.log.debug('Mode configured. Index=%d' % (self.index))
            
            # Write common datasets and attributes            
            self.h5Write('static','/'.join([self.rootPath,'Pulsewidth']),nppw)
            self.h5Write('attribute','/'.join([self.rootPath,'Pulsewidth','Unit']),'s')
            self.h5Write('static','/'.join([self.rootPath,'TxBaud']),nptxb)
            self.h5Write('attribute','/'.join([self.rootPath,'TxBaud','Unit']),'s')
            #Ambiguity pointer
            if self.pars.has_key('ambiguity'):
                self.h5Write('static','/'.join([self.rootPath,'Ambiguity']),self.pars['ambiguity'])
                
            # Check for display exclude
            if self.pars.get('displayexclude',0):
                self.arrayExclude('/%s' % ('/'.join([self.pars['mode'],self.pars['name']])))
            
        except Exception,value:
            self.log.exception(value)
            self.ext = None
            
    def shutdown(self):
        if self.index >= 0:
            self.ext.shutdown(self.index)
            
        self.h5Buffer.removeBranch(os.path.dirname(self.rootPath))    
            
        # Break circular references
        self.h5Buffer = None
        self.dllpath = None
        
        # Remove ext
        del self.pars
        del self.ext
        del self.log
                
    def h5Write(self,type,path,data=None):
        t = type.lower()
        if t == 'group':
            self.h5Buffer.setGroup(path)
        elif t == 'attribute':
            self.h5Buffer.setAttribute(path,data)
        elif t == 'static':
            self.h5Buffer.setStatic(path,data)
        elif t == 'dynamic':
            self.h5Buffer.setDynamic(path,data)
        else:
            self.log.error('Type: %s not supported' % (type))
            
    def arrayExclude(self,path):
        self.log.info('Excluding data tree: %s' % (path))
        self.h5Buffer.excludeArray(path)
            
        
class modeConfig:
    def __init__(self,file,dtc,id,mode):
        self.log = logging.getLogger('modeConfig')
        
        # Read mode main and sub sections from exp file
        commonsec = 'Common Mode:%d' % (id)
        mainsec = '%s Mode:%d' % (dtc,id)
        subsec = '%s,%s' % (mainsec,mode)
        
        # Read all relevant sections
        pars = {}
        if file.has_section(commonsec):
            pars.update(self.lowercaseKeys(file.vars(commonsec)))
        pars.update(self.lowercaseKeys(file.vars(mainsec)))
        pars.update(self.lowercaseKeys(file.vars(subsec)))
        self.pars = pars
            
    def lowercaseKeys(self,d):
        res = {}
        for vn,vv in d.items():
            res[vn.lower()] = vv
        return res
        
        
        
        
        
        
