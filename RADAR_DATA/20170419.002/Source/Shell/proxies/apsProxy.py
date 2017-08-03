""""
Amisr array specific proxy.

This proxy handles all the variables from the 
synchronizer that needs to go in the data files
"""


from shell.proxies.baseProxy import baseProxy, directory
import os,re
import siteconfig
from xmlrpclib import ServerProxy

class apsProxy(baseProxy):
    def __init__(self,experiment):
        baseProxy.__init__(self,experiment.dtc,experiment)
        
        syncurl = siteconfig.url('sync')
        self.remote = ServerProxy(syncurl)
           
        # Connect to array servers 
        dirurl = self.exp.systemConfig.get('servers','directory','')
        self.directoryServer = directory(dirurl)
        self.log.info('Connected to: %s' % (self.directoryServer.ident()))
        self.configServer = self.directoryServer.connect('config')
        self.log.info('Connected to: %s' % (self.configServer.ident()))
        arrayid = self.configServer.get('faces')
        self.arrayServer = self.directoryServer.connect(arrayid)
        self.log.info('Connected to: %s' % (self.arrayServer.ident()))        
        rackid = self.arrayServer.face_info('rfrack')
        self.rfRackServer = self.directoryServer.connect(rackid)
        self.log.info('Connected to: %s' % (self.rfRackServer.ident()))
        self.beamcodeServer = self.directoryServer.connect('beamcodes')
        self.log.info('Connected to: %s' % (self.beamcodeServer.ident()))
        
        h5buffer = self.exp.h5Buffer  
        
        try:
            # Receiver variables
            h5buffer.setDynamic('/Rx/TuningFrequency',[0.0,0.0])
            h5buffer.setAttribute('/Rx/TuningFrequency/Unit','Hz')
            h5buffer.setAttribute('/Rx/TuningFrequency/Description','RADAC RX tuning frequency')
            h5buffer.setDynamic('/Rx/Frequency',[0.0,0.0])
            h5buffer.setAttribute('/Rx/Frequency/Unit','Hz')
            h5buffer.setAttribute('/Rx/Frequency/Description','RX frequency')
            
            # Transmitter variables
            h5buffer.setDynamic('/Tx/TuningFrequency',[0.0,0.0])
            h5buffer.setAttribute('/Tx/TuningFrequency/Unit','Hz')
            h5buffer.setAttribute('/Tx/TuningFrequency/Description','TX LO frequency')
            h5buffer.setDynamic('/Tx/Frequency',[0.0,0.0])
            h5buffer.setAttribute('/Tx/Frequency/Unit','Hz')
            h5buffer.setAttribute('/Tx/Frequency/Description','TX frequency')
            
            # Transmitter power and AEU status
            h5buffer.setDynamic('/Tx/Power',[0.0,0.0])
            h5buffer.setAttribute('/Tx/Power/Unit','W')
            h5buffer.setAttribute('/Tx/Power/Description','TX Power')
            h5buffer.setDynamic('/Tx/AeuTx',[0,0])
            h5buffer.setAttribute('/Tx/AeuTx/Description','Transmitting AEUs')
            h5buffer.setDynamic('/Rx/AeuRx',[0,0])
            h5buffer.setAttribute('/Rx/AeuRx/Description','Receiving AEUs')
            h5buffer.setDynamic('/Site/AeuTotal',[0,0])
            h5buffer.setAttribute('/Site/AeuTotal/Description','Total number of AEUs')           
        except Exception,inst:
            self.log.exception(inst)
                    
        # load beamcode map
        self.loadPositions(h5buffer)
            
        self.log.info('Initialized')
        
        
    def setup(self,n):
        baseProxy.setup(self,n)
        
        dtc = self.name
        exp = self.exp
        conf = self.exp.experimentConfig
        id = self.exp.id
        
        rxinfo = self.exp.getRxChannelInfo(self.name,n)        
        self.rxLoName = rxinfo['loname']
        self.rxBandName = rxinfo['bandname']
        self.rxChannel = rxinfo['frequencyalgorithm']
        
        txinfo = self.exp.getTxChannelInfo(self.name,n)
        self.txFreqName = txinfo['frequencyname']
        self.txLoName = txinfo['loname']
        self.txBandName = txinfo['bandname']
        self.txChannel = txinfo['frequencyalgorithm']
        
        section = '%s mode:%d' % (dtc,n)
                
        syncvars = {}    
        txenabled = conf.getint(section,'txenabled',0)
        syncvars['tx%denabled' % (id)] = txenabled
        txband = conf.getfloat(section,'txband','450')
        # if txenabled then configure upconverter
        if txenabled:
            syncvars['tx%dband' % (id)] = txband
            uc = self.arrayServer.dtc_info(dtc,'upconverter')
            self.rfRackServer.setFrequencyBand(uc,txband)
        
        rxband = conf.getfloat(section,'rxband','')
        syncvars[self.rxBandName] = rxband
        # Configure downconverter
        dc = self.arrayServer.dtc_info(dtc,'downconverter')
        self.rfRackServer.setFrequencyBand(dc,rxband)
        
        # Send relevant variables to sync
        self.remote.variables(syncvars)
        
        
        
    def loadPositions(self,h5buffer):
        try:
            self.log.debug('loadPositions')
            data = self.beamcodeServer.getPositions()
            attributes = self.beamcodeServer.getHeaders()
            
            res = []
            for bc,cdata in data.items():
                row = []
                nbc = int(bc)
                nbc |= 0x8000
                row.append(nbc)
                row.extend(cdata)
                res.append(row)
            
            h5buffer.setStatic('/Setup/BeamcodeMap',res)
            
            for an,av in attributes.items():
                h5buffer.setAttribute('/Setup/BeamcodeMap/%s' % (an),av)
        except Exception,inst:
            self.log.exception(inst)
                        
        
    def storeData(self,h5buffer,starttime,endtime,vars):
        if vars.has_key(self.rxLoName):
            h5buffer.h5Dynamic.setOutBoth('/Rx/TuningFrequency',vars[self.rxLoName])            
            rxf = eval(self.rxChannel,globals(),vars)
            h5buffer.h5Dynamic.setOutBoth('/Rx/Frequency',rxf)
        
        if vars.has_key(self.txLoName):
            h5buffer.h5Dynamic.setOutBoth('/Tx/TuningFrequency',vars[self.txLoName])            
            txf = eval(self.txChannel,globals(),vars)
            h5buffer.h5Dynamic.setOutBoth('/Tx/Frequency',txf)
                            
        h5buffer.h5Dynamic.setOutBoth('/Tx/Power',vars.get('peak',0))        
        h5buffer.h5Dynamic.setOutBoth('/Tx/AeuTx',vars.get('numtx',0))
        h5buffer.h5Dynamic.setOutBoth('/Rx/AeuRx',vars.get('numrx',0))
        h5buffer.h5Dynamic.setOutBoth('/Site/AeuTotal',vars.get('total',0))
                
        
                
        
        
proxy = apsProxy