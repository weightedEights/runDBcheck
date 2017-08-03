""""
Sondrestrom specific proxy. Loaded based on siteconfig.sitename.

This proxy handles all the variables from the 
synchronizer that needs to go in the data files
"""


from shell.proxies.baseProxy import baseProxy
import os,re
import siteconfig
from xmlrpclib import ServerProxy

class sondrestromProxy(baseProxy):
    def __init__(self,experiment):
        baseProxy.__init__(self,experiment.dtc,experiment)
        
        syncurl = siteconfig.url('sync')
        self.remote = ServerProxy(syncurl)
                       
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
            
        except Exception,inst:
            self.log.exception(inst)
        self.log.info('Initialized')
        
    def setup(self,n):
        baseProxy.setup(self,n)
        
        rxinfo = self.exp.getRxChannelInfo(self.name,n)        
        self.rxLoName = rxinfo['loname']
        self.rxChannel = rxinfo['frequencyalgorithm']
        
        txinfo = self.exp.getTxChannelInfo(self.name,n)
        self.txFreqName = txinfo['frequencyname']
        self.txLoName = txinfo['loname']
        self.txChannel = txinfo['frequencyalgorithm']
        
        
    def storeData(self,h5buffer,starttime,endtime,vars):        
        if vars.has_key(self.rxLoName):
            h5buffer.h5Dynamic.setOutBoth('/Rx/TuningFrequency',vars[self.rxLoName])            
            rxf = eval(self.rxChannel,globals(),vars)
            h5buffer.h5Dynamic.setOutBoth('/Rx/Frequency',rxf)
        
        if vars.has_key(self.txLoName):
            h5buffer.h5Dynamic.setOutBoth('/Tx/TuningFrequency',vars[self.txLoName])            
            txf = eval(self.txChannel,globals(),vars)
            h5buffer.h5Dynamic.setOutBoth('/Tx/Frequency',txf)
                
        
                
        
        
proxy = sondrestromProxy