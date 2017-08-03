"""
Txc proxy.
This proxy handles all communication between the Txc and the 
shell programs

"""

from shell.proxies.baseProxy import baseProxy
import os,re
import time
from xmlrpclib import ServerProxy
import siteconfig

class txcProxy(baseProxy):
    def __init__(self,experiment):
        baseProxy.__init__(self,'txc',experiment)
        
        txcurl = siteconfig.url('txc')
        self.remote = ServerProxy(txcurl)
                       
        h5buffer = self.exp.h5Buffer        
        try:
            h5buffer.setDynamic('/Tx/Power',[0.0,0.0])
            h5buffer.setAttribute('/Tx/Power/Unit','W')
            h5buffer.setAttribute('/Tx/Power/Description','Transmitted power')
            
        except Exception,inst:
            self.log.exception(inst)
        self.log.info('Initialized')
        
    def setup(self,n):
        baseProxy.setup(self,n)
        
        section = '%s mode:%d' % (self.exp.dtc,n)
        txf = self.exp.experimentConfig.get(section,'txfrequency',0)
        
        try:
            # sondrestrom legacy 
            f = int(txf)
            self.powerName = 'txcfrequency%dpower' % (f)
        except:
            # amisr and new sondrestrom way. The freq is named directly in the exp file
            mo = re.match('tx([c0-9]*)frequency([0-9]*)',txf,re.I)
            txid = mo.group(1)
            txlo = mo.group(2)
            self.powerName = 'tx%sfrequency%spower' % (txid,txlo)
        
    def storeData(self,h5buffer,starttime,endtime,vars):
        tsub = time.clock()
        state = self.remote.getState([starttime,endtime])
        esub = time.clock()-tsub
        self.log.info('Gathering txc info: %3.2f [secs]' % (esub))
        
        # set boi info
        if state[0]['_timediff'] < 60:
            h5buffer.h5Dynamic.setOutBoi('/Tx/Power',state[0][self.powerName])
        else:
            self.log.error('Txc boi info of in time by: %f secs' % (state[0]['_timediff']))
        
        # set eoi info
        if state[1]['_timediff'] < 60:
            h5buffer.h5Dynamic.setOutEoi('/Tx/Power',state[1][self.powerName])
        else:
            self.log.error('Txc eoi info of in time by: %f secs' % (state[1]['_timediff']))
        
        
        
                
        
        
proxy = txcProxy