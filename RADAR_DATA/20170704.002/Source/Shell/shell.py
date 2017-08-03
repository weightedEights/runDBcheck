"""
This program is a member of the new python based DAQ system.
It collects samples from the RADAC, call mode specific integration 
routines and write the output to hdf5 formatted data files.

The interface to the program is based on the xmlrpc protocol.

History:
    Initial implementation
    Date:       20070212
    Author:     John Jorgensen
    Company:    SRI International
    
"""

from release import version

import sys,os
# Fix to allow slave dtcs to import python packages from master dtc
# packages = os.path.normpath(os.environ['pydaqpackages'])
# if not packages.endswith('\\'):
    # packages += '\\'
# if packages not in sys.path:    
    # sys.path.append(packages)

import logging
from pylib.logger import logger
logging.setLoggerClass(logger)

from urlparse import urlparse
from SimpleXMLRPCServer import SimpleXMLRPCServer
from SocketServer import ThreadingMixIn
from xmlrpclib import ServerProxy
from pylib.radac import xmlRadac
from experiment import experiment
from pylib.exceptionclasses import *
from pylib.baseClasses import baseClass
from pylib.sysutils import urlExtract,getRefCounts
import numpy as np

class shell(baseClass):
    def __init__(self):
        name = os.environ['computername']
        baseClass.__init__(self,name)
        
        ## Create a logger
        # self.log = logging.getLogger(app)
        # self.log.setLevel(logging.DEBUG)
        
        self.exp = None
        self.err = None
        
        # Set various attributes
        # self.name = os.environ['COMPUTERNAME']
        # self.config['name'] = self.name
        # self.config['ident'] = self.ident()
        
        ## Exported functions
        self.register_function(self.loadExperiment)
        self.register_function(self.endExperiment)
        self.register_function(self.startIntegrator)
        self.register_function(self.stopIntegrator)
        self.register_function(self.startIntegration)
        self.register_function(self.saveRecord)
        self.register_function(self.outputFolder)
        self.register_function(self.setFilename)
        self.register_function(self.setFilenumber)
        self.register_function(self.lastError)
        self.register_function(self.loadBeamcodes)
        self.register_function(self.configureSetup)
        self.register_function(self.experimentStatus)
        self.register_function(self.setRecord)
        self.register_function(self.setDisplayRecord)
        self.register_function(self.setStatic)
        self.register_function(self.setAsync)
        self.register_function(self.settings)
        self.register_function(self.setIppMask)
        self.register_function(self.setTrackInfo)
        self.register_function(self.referenceCounts)
        
        #Direct mapping to Radac specific functions
        self.register_instance(xmlRadac())
                    
    # XmlRPc exported functions
    def status(self):
        return 1
        
    def ident(self):
        return 'Shell: %s, version: %s' % (self.name,version)
        
    def loadExperiment(self,filename='',expid='',options={}):
        try:
            if self.exp is not None: ## Kill old experiment if any
                self.exp.shutdown()
            self.exp = experiment(filename,expid,self.ident(),options)
            return 1
        except Exception,self.err:
            self.log.exception(self.err)
            return 0
            
    def endExperiment(self):
        try:
            self.exp.shutdown()
            self.exp = None
            return 1
        except Exception,self.err:
            self.log.exception(self.err)
            return 0
            
    def startIntegrator(self,timetag=0):
        try:
            self.exp.startIntegrator(timetag)
            return 1
        except Exception,self.err:
            self.log.exception(self.err)
            return 0;
                        
    def stopIntegrator(self,timetag=0):
        try:
            self.exp.stopIntegrator(timetag)
            return 1
        except Exception,self.err:
            self.log.exception(self.err)
            return 0;
                        
    def startIntegration(self,vars={}):
        try:
            self.exp.startIntegration(vars)
            return 1
        except integratorInfo,e:
            self.err = e.message
            return 0
        except Exception,self.err:
            self.log.exception(self.err)
            return 0;
                        
    def saveRecord(self,vars={}):
        try:
            self.exp.saveRecord(vars)
            return 1
        except Exception,self.err:
            self.log.exception(self.err)
            return 0

    def outputFolder(self,outputfolder):
        try:
            self.exp.outputFolder = outputfolder
            return 1
        except Exception,self.err:
            self.log.exception(self.err)
            return 0


            
    def setFilename(self,filename,delete=False):
        try:
            self.exp.setFilename(filename,delete)
            return 1
        except Exception,self.err:
            self.log.exception(self.err)
            return 0 
            
    def setFilenumber(self,filenumber):
        try:
            self.exp.setFilenumber(filenumber)
            return 1
        except Exception,self.err:
            self.log.exception(self.err)
            return 0 
            
    def setRecord(self,flag):
        try:
            self.exp.setRecord(flag)
            return 1
        except Exception,self.err:
            self.log.exception(self.err)
            return 0 
            
    def setDisplayRecord(self,flag):
        try:
            self.exp.setDisplayRecord(flag)
            return 1
        except Exception,self.err:
            self.log.exception(self.err)
            return 0 
            
    def setRf(self,flag):
        try:
            self.exp.setRf(flag)
            return 1
        except Exception,self.err:
            self.log.exception(self.err)
            return 0 
            
    def lastError(self):
        return self.err
        
    def experimentStatus(self):
        try:
            return self.exp.status()
        except AttributeError:
            return {}
        except Exception,self.err:
            self.log.exception(self.err)
            return {} 
        
    def loadBeamcodes(self,filename):
        try:
            self.exp.loadBeamcodes(filename)
            return 1
        except Exception,self.err:
            self.log.exception(self.err)
            return 0 
            
    def configureSetup(self,setup):
        try:
            self.exp.configureSetup(setup)
            return 1
        except Exception,self.err:
            self.log.exception(self.err)
            return 0

    def setStatic(self,path,data,attr={}):
        try:
            self.exp.h5Buffer.setStatic(path,np.array(data))
            for an,av in attr.items():
                self.exp.h5Buffer.setAttribute(an,av)
            return 1
        except Exception,self.err:
            self.log.exception(self.err)
            return 0
            
            
            
    def setAsync(self,path,varnames,data,attr={}):
        try:
            self.exp.h5Buffer.setAsync(path,varnames,data,attr)
            return 1
        except Exception,self.err:
            self.log.exception(self.err)
            return 0
            
    def settings(self,settings):
        try:
            self.exp.safeSettings(settings)
            return 1
        except Exception,self.err:
            self.log.exception(self.err)
            return 0
            
    def setIppMask(self,mask):
        try:
            self.exp.setIppMask(mask)
            return 1
        except:
            return 0
            
    def setTrackInfo(self,track):
        try:
            self.exp.setTrackInfo(track)
            return 1
        except:
            return 0
        
    def referenceCounts(self):
        pairs = getRefCounts()
        res = [(cnt,str(cls)) for cnt,cls in pairs]
        return res

        
        
if __name__ == '__main__':
    from traceback import print_exc
    try:
        sh = shell()
        sh.serve_forever()
    except Exception,inst:
        print_exc()
    raw_input('Hit any key to terminate')
   
    
    
    
    