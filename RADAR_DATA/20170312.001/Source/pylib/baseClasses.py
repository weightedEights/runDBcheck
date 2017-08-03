"""
pyDAQ base class. 
All the programs in the pyDAQ suite should decend from
this class

History:
    Date:           20070802
    Description:    Initial implementation
    Author:         JJ
    
"""
from pylib.logger import logger
import logging
logging.setLoggerClass(logger)
import logging.handlers
import sys,os,re

from SimpleXMLRPCServer import SimpleXMLRPCServer
from SocketServer import ThreadingMixIn

import siteconfig
from pylib.Inifile import Inifile
from pylib.sysutils import urlExtract
from expbuild.classes import binConfig

loglevels = {'debug':logging.DEBUG,'info':logging.INFO,'warning':logging.WARNING,'error':logging.ERROR,'critical':logging.CRITICAL}

class baseClass(ThreadingMixIn,SimpleXMLRPCServer):

    allow_reuse_address = 1
    
    def __init__(self,name='',rootlogger=False):
        self.name = name
        
        self.systemConfig = siteconfig.inifile('all')
        
        url = self.systemConfig.get('servers',self.name,'')
        port = urlExtract('port',url)
        addr = ('',port)
        SimpleXMLRPCServer.__init__(self,addr,logRequests=False)
                
        logurl = self.systemConfig.get('servers','log.tcp','dtc0:9020')
        logaddr = urlExtract('address',logurl)
        logport = urlExtract('port',logurl)
        
        locallevel = siteconfig.get('logging','locallevel','info').lower()
        globallevel = siteconfig.get('logging','globallevel','info').lower()
        try:
            llevel = loglevels[locallevel]
        except:
            llevel = logging.INFO
        try:
            glevel = loglevels[globallevel]
        except:
            glevel = logging.INFO
        
        if rootlogger:
            fmt = logging.Formatter('%(asctime)s %(server)-6s %(application)-12s %(name)-12s %(levelname)-6s %(message)s',datefmt='%H:%M:%S')        
        else:
            fmt = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s',datefmt='%H:%M:%S')
            socketloghandler = logging.handlers.SocketHandler(logaddr,logport)
            socketloghandler.setLevel(glevel)
            logging.getLogger('').addHandler(socketloghandler)
            

        console = logging.StreamHandler(sys.stdout)
        console.setLevel(llevel)
        console.setFormatter(fmt)
        logging.getLogger('').addHandler(console)
        logging.root.setLevel(llevel) 
            
        ## Internal system functions
        self.register_introspection_functions()
        
        ## Common functions
        self.register_function(self.ident)
        self.register_function(self.status)
        
        self.log = logging.getLogger('main')
        self.log.info('Running on port: %d',port)

class baseExperiment:
    def __init__(self,exppath):
        self.log = logging.getLogger('experiment')
        self.log.info('Experiment starting...')
        
        self.application = os.path.basename(sys.argv[0]).split('.')[0]
        self.dtc = self.name = self.computerName = os.environ['COMPUTERNAME']
        
        self.id = int(self.dtc[3:]) # extract dtc number
        self.log.debug('Id: %d' % (self.id))
        
        
        # load binary config file
        root,ext = os.path.splitext(exppath)
        expconfig = '.'.join([root,'h5'])
        self.binConfig = binConfig(expconfig)
        
        # Load system config file
        self.systemConfig = self.binConfig.inifile('system')
        self.systemConfigFilename = self.systemConfig.Filename
        
        # Load experiment file
        self.experimentConfigFilename = exppath
        self.experimentConfigPath = os.path.dirname(exppath)
        self.experimentBasename = os.path.basename(exppath)
        self.experimentConfig = self.binConfig.inifile(self.experimentBasename)
        
    def reloadConfig(self):
        root,ext = os.path.splitext(self.experimentConfigFilename)
        expconfig = '.'.join([root,'h5'])
        self.binConfig = binConfig(expconfig)
        return self.binConfig
        
    def getRxChannelInfo(self,dtc,setup):
        info = {}
        id = int(dtc[3:])
        section = '%s mode:%d' % (dtc,setup)
        
        rxloname = 'rx%dlo0' % (id)
        rxbandname = 'rx%dband' % (id)
        rxch = self.experimentConfig.get(section,'rxchannel','ionline')
        rxchannel = siteconfig.get('rx channels',rxch,'')
        if rxchannel:
            rxchannel = rxchannel.replace('nco',rxloname)
            rxchannel = rxchannel.replace('rxband',rxbandname)
        else:
            raise Exception('Rx channel not specified')
            
        tuningmethod = self.experimentConfig.getint(section,'tuningmethod',0)
        if tuningmethod:
            tuningsection = 'tuning method %d' % (tuningmethod)
            method = siteconfig.vars(tuningsection)
            if self.experimentConfig.has_section(tuningsection):
                method = self.experimentConfig.vars(tuningsection)
            if method:
                tuningalgorithm = method
            else:
                raise Exception('Not tuning method specified for method: %d' % (tuningmethod))
        else:
            tuningalgorithm = {}
                    
        info['channelname'] = rxch
        info['loname'] = rxloname
        info['bandname'] = rxbandname
        info['frequencyalgorithm'] = rxchannel
        info['tuningmethod'] = tuningmethod
        info['tuningalgorithm'] = tuningalgorithm
        
        return info
        
    def getTxChannelInfo(self,dtc,setup):
        info = {}
        section = '%s mode:%d' % (dtc,setup)
        
        txf = self.experimentConfig.get(section,'txfrequency',0)
        txch = self.experimentConfig.get(section,'txchannel','normal')
        txchannel = siteconfig.get('tx channels',txch,'')
                  
        # amisr and new sondrestrom way. The freq is named directly in the exp file
        mo = re.match('tx([c0-9]*)frequency([0-9]*)',txf,re.I)
        if not mo is None:
            txid = mo.group(1)
            txlo = mo.group(2)
            txfreqname = txf
            txloname = 'tx%slo%s' % (txid,txlo)
            txbandname = 'tx%sband' % (txid)
        else:
            raise Exception('Can not interpret txfrequency')
                
        # substitute config "dso" with real txlo name        
        if txchannel:
            txchannel = txchannel.replace('dso',txloname)
            txchannel = txchannel.replace('txband',txbandname)
        else:
            raise Exception('Tx channel not specified')
            
        info['channelname'] = txch
        info['loname'] = txloname
        info['bandname'] = txbandname
        info['frequencyname'] = txfreqname
        info['frequencyalgorithm'] = txchannel
        
        return info
        
        
        
        