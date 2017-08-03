"""
Acc proxy.
This proxy handles all communication between the ACC and the 
shell programs

"""

from shell.proxies.baseProxy import baseProxy
import os
import time
from xmlrpclib import ServerProxy
import siteconfig

timediscrepancy=0.8

class accProxy(baseProxy):
    def __init__(self,experiment):
        baseProxy.__init__(self,'Acc',experiment)
        
        accurl = siteconfig.url('acc')
        self.remote = ServerProxy(accurl)
        
        h5buffer = experiment.h5Buffer
                       
        try:
            h5buffer.setDynamic('/Antenna/Azimuth',[0.0,0.0])
            h5buffer.setDynamic('/Antenna/Elevation',[0.0,0.0])
            h5buffer.setDynamic('/Antenna/DesiredAzimuth',[0.0,0.0])
            h5buffer.setDynamic('/Antenna/DesiredElevation',[0.0,0.0])
            h5buffer.setDynamic('/Antenna/Mode',[0,0])
            h5buffer.setDynamic('/Antenna/Event',[0,0])
            h5buffer.setDynamic('/Antenna/EventCount',[0,0])
            h5buffer.setDynamic('/Antenna/Offset',[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
            h5buffer.setDynamic('/Antenna/OffsetType',[0,0])
            h5buffer.setDynamic('/Antenna/Filename',[' '*80,' '*80])
            h5buffer.setDynamic('/Antenna/UserNumbers',[0]*16)
            h5buffer.setDynamic('/Antenna/UserString',[' '*256,' '*256])
        except Exception,inst:
            self.log.exception(inst)
        self.log.info('Initialized')
        
    def storeData(self,h5buffer,starttime,endtime,vars):
        try:
            tsub = time.clock()
            state = self.remote.getState([starttime,endtime])
            esub = time.clock()-tsub
            self.log.info('Gathering acc info: %3.2f [secs]' % (esub))
            
            # set boi info
            h5buffer.h5Dynamic.setOutBoi('/Antenna/Azimuth',state[0]['azimuth'])
            h5buffer.h5Dynamic.setOutBoi('/Antenna/Elevation',state[0]['elevation'])
            h5buffer.h5Dynamic.setOutBoi('/Antenna/DesiredAzimuth',state[0]['antstate']['azd'])
            h5buffer.h5Dynamic.setOutBoi('/Antenna/DesiredElevation',state[0]['antstate']['eld'])
            h5buffer.h5Dynamic.setOutBoi('/Antenna/Mode',state[0]['antennamode'])
            h5buffer.h5Dynamic.setOutBoi('/Antenna/Event',state[0]['antennaevent'])
            h5buffer.h5Dynamic.setOutBoi('/Antenna/EventCount',state[0]['antennaeventcount'])
            h5buffer.h5Dynamic.setOutBoi('/Antenna/Offset',[state[0]['antennaoffset1'],state[0]['antennaoffset2'],
                                                            state[0]['antennaoffset3'],state[0]['antennaoffset4']])
            h5buffer.h5Dynamic.setOutBoi('/Antenna/OffsetType',state[0]['antennaoffsettype'])
            h5buffer.h5Dynamic.setOutBoi('/Antenna/Filename',state[0]['antennafilename'])
            h5buffer.h5Dynamic.setOutBoi('/Antenna/UserNumbers',state[0]['usernumbers'])
            h5buffer.h5Dynamic.setOutBoi('/Antenna/UserString',state[0]['userstring'])
            if state[0]['_timediff'] > timediscrepancy:
                self.log.error('Antenna boi info off in time by: %f secs' % (state[0]['_timediff']))
            
            # set eoi info
            h5buffer.h5Dynamic.setOutEoi('/Antenna/Azimuth',state[1]['azimuth'])
            h5buffer.h5Dynamic.setOutEoi('/Antenna/Elevation',state[1]['elevation'])
            h5buffer.h5Dynamic.setOutEoi('/Antenna/DesiredAzimuth',state[1]['antstate']['azd'])
            h5buffer.h5Dynamic.setOutEoi('/Antenna/DesiredElevation',state[1]['antstate']['eld'])
            h5buffer.h5Dynamic.setOutEoi('/Antenna/Mode',state[1]['antennamode'])
            h5buffer.h5Dynamic.setOutEoi('/Antenna/Event',state[1]['antennaevent'])
            h5buffer.h5Dynamic.setOutEoi('/Antenna/EventCount',state[1]['antennaeventcount'])
            h5buffer.h5Dynamic.setOutEoi('/Antenna/Offset',[state[1]['antennaoffset1'],state[1]['antennaoffset2'],
                                                            state[1]['antennaoffset3'],state[1]['antennaoffset4']])
            h5buffer.h5Dynamic.setOutEoi('/Antenna/OffsetType',state[1]['antennaoffsettype'])
            h5buffer.h5Dynamic.setOutEoi('/Antenna/Filename',state[1]['antennafilename'])
            h5buffer.h5Dynamic.setOutEoi('/Antenna/UserNumbers',state[1]['usernumbers'])
            h5buffer.h5Dynamic.setOutEoi('/Antenna/UserString',state[1]['userstring'])
            if state[1]['_timediff'] > timediscrepancy:
                self.log.error('Antenna eoi info off in time by: %f secs' % (state[1]['_timediff']))
        except Exception,inst:
            self.log.exception(inst)
        
        
                
        
        
proxy = accProxy
