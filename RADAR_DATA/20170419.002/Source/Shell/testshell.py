"""
Shell test script

"""

import xmlrpclib as xl
from time import sleep
import os

loops = 1
integrate = 1
ints = 1
expfolder = '//dtc0/radac/pydaq/setup/3PosS'

def doIntegration(dtc):
    while not dtc.startIntegration():
        sleep(0.1)
        
if __name__ == '__main__':
    try:
        dtc = xl.ServerProxy('http://dtc0:8080')
        print dtc.ident()
        
        if not dtc.loadExperiment(os.path.join(expfolder,'3PosS.exp')):
            raise Exception(dtc.lastError())
        if not dtc.setFilename('c:/tmp/test0001.dt0.h5',True):
            raise Exception(dtc.lastError())
            
        if not dtc.setRecord(True):
            raise Exception(dtc.lastError())
            
        if not dtc.setRf(True):
            raise Exception(dtc.lastError())
            
        for l in range(loops):    
            dtc.loadBeamcodes('beamcodes.bco')
            dtc.setFilename('c:/tmp/test0001.dt0.h5',True)
            dtc.configureSetup(0)
            for i in range(ints):
                doIntegration(dtc)
                
            while dtc.experimentStatus()['integratorbusy']:
                print 'waiting for integrator'
                sleep(1)
                
            dtc.loadBeamcodes('beamcodes1.bco')
            dtc.setFilename('c:/tmp/test0002.dt0.h5',True)
            dtc.configureSetup(0)
            
            print 'starting second integration'
            for i in range(ints):
                doIntegration(dtc)
            
            while dtc.experimentStatus()['integratorbusy']:
                print 'waiting for integrator'
                sleep(1)
                
    finally:
        dtc.endExperiment()
        
    