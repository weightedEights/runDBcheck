"""
Experiment class
Handles configuration and execution of an experiment

History:
    Date:       20070309
    Author:     John Jorgensen
    Company:    SRI International
    Initial implementation
    
"""
import time
import logging
import os,sys
from xmlrpclib import ServerProxy
import siteconfig
from pylib.baseClasses import baseExperiment
from pylib.exceptionclasses import *
from pylib.sysutils import urlExtract,LockableSet
from h5StorageBoth import h5Storage,h5Buffer
from pylib.exceptionclasses import integratorInfo
from setup import setup
from pylib.radac import *
from modeclasses import mode
from wrapintegrator import wrapIntegrator
import numpy as np
import systemCheck
from jjlib.Utils.sysutils import lowercaseKeysDict

from Queue import Queue,Empty
import thread
from threading import Event
from proxies.proxies import proxies

achar = [chr(c) for c in range(ord('a'),ord('z')+1,1)]
achar.extend([chr(c) for c in range(ord('A'),ord('Z')+1,1)])
ALPHA = ''.join(achar)

def dataPath(sys,exp,id,setup,dtc,options={}):
    datapath = '/tmp'
    section = '%s mode:%s' % (dtc,setup)
        
    # backwards compatibility 
    try:
        if options['testing']:
            datapath = sys.get('testing','datafolder')
        else:    
            datapath = sys.get('experiment','folder','')
        commonlocalpath = exp.get('common parameters','localdata','')
        localdata = exp.get(section,'localdata',commonlocalpath)
        if localdata:
            datapath = localdata
    except:
        # now we count on that the new way is successfull below
        pass
    
    # new way override old way above   
    try:
        places = lowercaseKeysDict(sys.vars('storage places'))
        defaults = lowercaseKeysDict(sys.vars('storage defaults'))
        
        if options['testing']:
            datapath = places['test']
        else:           
            dtcdefault = defaults.get(dtc,'main')
            if dtcdefault in places:
                datapath = places[dtcdefault]
            else:
                # full path specified
                datapath = dtcdefault
            
        # check experiment file and override defaults if datafolder is specified
        commondefault = exp.get('common parameters','datafolder',datapath)
        if commondefault in places:
            datapath = places[commondefault]
        else:
            datapath = commondefault
            
        # and last check for setup specific override
        setupoverride = exp.get(section,'datafolder','')
        if setupoverride:
            if setupoverride in places:
                datapath = places[setupoverride]
            else:
                datapath = setupoverride
    except:
        # assumes old way if we get here
        pass
            
    datapath = os.path.join(datapath,id)            
    return datapath

class experiment(baseExperiment):
    def __init__(self,exppath='',expid='',identity='',options={}):
        baseExperiment.__init__(self,exppath)  
        self.expId = expid
        self.identity = identity
        self.options = options
        self.log.info('Loading experiment: %s' % (os.path.basename(exppath)))
        
        bconf = self.binConfig
        
        self.state = LockableSet()
                
        self.name = os.environ['computername']
        
        self._filenumber = 0
        
        self.intQueue = Queue()
        self.threadShutdown = Event()
        thread.start_new_thread(self.intFinishThread,())
        
        self.syncSettings = Queue()
        
        self.radac = radacInterface()
        # set radac event recipient
        self.radac.register(self.radacEvent)
                
        url = self.systemConfig.get('servers',self.name,'')
        syncurl = self.systemConfig.get('servers','sync','')
        syncaddr = urlExtract('address',syncurl)
        if url <> '':
            proxyport = urlExtract('proxyport',url)
            proxyurl = 'http://%s:%d' % (syncaddr,proxyport)
            self.sync = ServerProxy(proxyurl)
            self.log.info('Proxy url: %s' % (self.sync._ServerProxy__host))
        else:
            self.log.error('Could not find url for %s' % (self.name))
            
        # Dtc name and id
        self.dtc = self.name
        try:
            self.id = int(self.dtc.strip(ALPHA))
            self.log.debug('Id: %d' % (self.id))
        except:
            self.id = 0
            self.log.critical('Could not extract id number from dtc name: %s' % (self.dtc))
            
        # Create buffer to hold data going to disk
        self.log.info('Starting storage thread')
        self.h5Buffer = h5Buffer()
        
        # Data storage thread
        self.dataStorage = h5Storage('DataStorage',self.h5Buffer)
        
        # Display storage thread
        # self.displayStorage = h5Storage('DisplayStorage',self.h5Buffer,options=['lock'])
        
        # Initialize integrator
        self.integrator = wrapIntegrator(self)
        self.integrator.registerEvent('integrating',self.integrating)
                
        # low level system limits
        defaultlim = self.systemConfig.get('limits','default','')
        if defaultlim <> '':
            try:
                limits = systemCheck.LIMITS[defaultlim]
            except:
                self.log.error('No default limits for: %s\nGoing to use AMISR limits' % (defaultlim))
                limits = systemCheck.LIMITS['amisr']
        else:
            limits = self.systemConfig.vars('limits')
        self.log.info('Low level limits: %s' % (str(limits)))
        # radac.setLimits(limits)
        self.limits = limits
        
        self.datapath = '/tmp' # if datapath configuration fails completely it will try to store in /tmp
        
        # Get RecordsPerFile from system ini and possibly overloaded in the experiment file
        self.recordsPerFile = self.systemConfig.getint('Data','RecordsPerFile',250)
        self.recordsPerFile = self.experimentConfig.getint('Common Parameters','RecordsPerFile',self.recordsPerFile)
        self.log.info('RecordsPerFile: %d' % (self.recordsPerFile))
                
        #Display record filename
        try:
            defaultDisplayfilename = 'c:/tmp/%s.h5' % (self.dtc)
            systemDisplayfilename = self.systemConfig.get(self.dtc,'displayfilename',defaultDisplayfilename)
            expDisplayfilename = self.experimentConfig.get('common parameters','displayfilename',systemDisplayfilename)
            displayfilename = self.experimentConfig.get(self.dtc,'displayfilename',expDisplayfilename)
            self.dataStorage.setDisplayFilename(displayfilename)
            self.log.info('Display filename: %s' % (displayfilename))
        except Exception,inst:
            self.log.exception(inst)

                
        # Load setups
        
        # gather external setup stuff that is needed
        extconf = {'nradacheaderwords':self.radac.readRegister('nheaderwords')}
        self.log.info('Loading setups')
        self.setups = []
        modes = self.experimentConfig.options('Modes')
        for mode in modes:
            self.setups.append(setup(self,mode,extconf))
        
        self._setup = 0
        # self.configureSetup(self._setup) ## Always start with setup 0
        
        # Write experiment relevant info to h5buffer
        
        # Handle [include data] section of experiment file
        if self.experimentConfig.has_section('include data'):
            include = self.experimentConfig.vars('include data')
            forme = [v for v in include.items() if v[0].lower().find(self.dtc.lower()) <> -1]
            forall = [v for v in include.items() if v[0].lower().find('all') <> -1]
            forme.extend(forall)
            for dst,fn in forme:
                sdst = dst.split(':')
                h5path = sdst[1]
                if not h5path.startswith('/'):
                    h5path = '/'+h5path
                fp = '\\'.join([self.experimentConfigPath,fn])
                self.h5Buffer.includeFile(h5path,fp)
                
        # Datafile version tracking
        major = 1
        minor = 0
        self.h5Buffer.setAttribute('/Major',major)
        self.h5Buffer.setAttribute('/Minor',minor)
        self.h5Buffer.setAttribute('/Description','RADAC data file version: %d.%d' % (major,minor))
                 
        # Site information from system.ini
        self.log.info('Reading Site information')
        self.h5Buffer.setStatic('/Site/Name',siteconfig.get('site','name','unknown'))
        self.h5Buffer.setStatic('/Site/Code',siteconfig.getint('site','code',-1))
        self.h5Buffer.setStatic('/Site/Latitude',siteconfig.getfloat('site','latitude',0.0))
        self.h5Buffer.setAttribute('/Site/Latitude/Unit',u'º')
        self.h5Buffer.setStatic('/Site/Longitude',siteconfig.getfloat('site','longitude',0.0))
        self.h5Buffer.setAttribute('/Site/Longitude/Unit',u'º')
        self.h5Buffer.setStatic('/Site/Altitude',siteconfig.getfloat('site','altitude',0.0))
        self.h5Buffer.setAttribute('/Site/Altitude/Unit','m')
   
        # Program version and radac info
        self.log.info('Reading Program and Radac version info')
        self.h5Buffer.setStatic('/Setup/Program',sys.argv[0])
        self.h5Buffer.setAttribute('/Setup/Program/Version',self.getVersion())
        self.h5Buffer.setStatic('/Setup/RadacInfo',[self.radac.info['versionnumber'],self.radac.info['versiondate']])
        self.h5Buffer.setAttribute('/Setup/RadacInfo/VersionString','%x' % (self.radac.info['versionnumber']))
        self.h5Buffer.setAttribute('/Setup/RadacInfo/VersionDateString','%i' % (self.radac.info['versiondate']))
        
        # System.ini and experiment file
        self.log.info('Reading System.ini and experiment ini files')
        self.h5Buffer.setStatic('/Setup/Systemfile',bconf.text('system'))
        self.h5Buffer.setStatic('/Setup/Experimentfile',bconf.text(self.experimentConfigFilename))
        
        # Load the constants section from siteconfig
        self.log.info('Reading Constants and attributes')
        if siteconfig.hassection('constants'):
            consts = siteconfig.vars('constants')
            for p,v in consts.items():
                self.h5Buffer.setStatic(p,float(v))
        if siteconfig.hassection('constant attributes'):        
            attrs = siteconfig.vars('constant attributes')
            for p,v in attrs.items():
                self.h5Buffer.setAttribute(p,v)
        
        # Dynamic boi and eoi placeholders and attributes
        self.log.info('Initial setup of time arrays and attributes')
        self.h5Buffer.setDynamic('/Time/RadacTime',np.array([0.0,0.0]))
        self.h5Buffer.setAttribute('/Time/RadacTime/Unit',u'µs')
        self.h5Buffer.setAttribute('/Time/RadacTime/Descriptions','µSeconds since 00:00:00 UTC on January 1, 1970')
        self.h5Buffer.setDynamic('/Time/RadacTimeString',np.array(['x'*30,'x'*30]))
        self.h5Buffer.setAttribute('/Time/RadacTimeString/Format',u'YYYY-MM-DD HH:MM:SS.µµµµµµ')
        self.h5Buffer.setDynamic('/Time/MatlabTime',np.array([0.0,0.0]))
        self.h5Buffer.setAttribute('/Time/MatlabTime/Unit','Days')
        self.h5Buffer.setAttribute('/Time/MatlabTime/Description','Days since 00:00:00 on January 1, 0000')
        self.h5Buffer.setDynamic('/Time/UnixTime',np.array([0,0]))
        self.h5Buffer.setAttribute('/Time/UnixTime/Unit','s')
        self.h5Buffer.setAttribute('/Time/UnixTime/Description','Seconds since 00:00:00 UTC on January 1, 1970')
        self.h5Buffer.setDynamic('/Time/Synchronized',np.array([0,0]))
        self.h5Buffer.setAttribute('/Time/Synchronized/Description','0=Not Synchronized, 1=Synchronized')
        
        self.h5Buffer.setDynamic('/Integration/MissedPulses',np.array([0]))
        self.h5Buffer.setAttribute('/Integration/MissedPulses/Desciption','Number of missed pulses in integration')
        self.h5Buffer.setAttribute('/Integration/MissedPulses/Unit','Count')
        
        # Load proxy interfaces for external data sources
        self.proxies = proxies(self)
        
        self.h5Buffer.synchronize()
        
        self.log.info('Experiment initialized')
        
    def shutdown(self):
        # h5Buffer should be the first to shut down
        self.h5Buffer.shutdown()
        self.threadShutdown.set()
        
        self.integrator.shutdown()       
        self.dataStorage.shutdown()        
        self.radac.shutdown()
       
 
    def configureSetup(self,number):
        if self.integrator.busy():
            self.syncSettings.put((self.configureSetup,number))
            self.sync.excludeSignal('go')
            return 2
            
        try:    
            self.log.info('Configuring setup: %d' % (number))
            
            self.proxies.setup(number)
            
            setup = self.setups[number]
            self._setup = number
            
            bconf = self.binConfig
            sys = bconf.inifile('system')
            exp = bconf.inifile('experiment')
            
            # Storage configuration
            self.datapath = dataPath(sys,exp,self.expId,number,self.name,self.options)
            self.log.info(self.datapath)
            try:
                if not os.path.exists(self.datapath):
                    os.makedirs(self.datapath)
            except:
                pass
                            
            # Misc.
            # self.displayStorage.record = setup.writeDisplayRecord
            self.dataStorage.setDisplayRecord(setup.writeDisplayRecord)
            self.log.info('Write display record: %i' % (setup.writeDisplayRecord))
                        
            # Load receiver
            self.radac.loadRx(bconf[setup.rxNode])
            if setup.rxFrequency:
                self.radac.rxFrequency(setup.rxFrequency)
                self.log.info('Rx frequency override in experiment file, new value: %7.0f' % (setup.rxFrequency))
            self.log.info('Rx configured')
            
            # Load transmitter
            self.radac.loadTx(setup.txImage)
            self.log.info('Tx configured')
            
            # Load beamcodes
            self.radac.loadBeamcodes(setup.beamcodes)
            self.log.info('Beamcodes loaded')
            
            # Set default values in control register
            self.radac.control(crHeaderEnable,on)
            self.radac.control(crTuSwEnable,on)
            self.radac.control(crUseRx,on)
            self.radac.control(crEnableDmaIntr,on)
            self.radac.control(crEnablePhaseFlip,on)
            self.radac.control(crSyncRx,on)
            self.radac.control(crTestData,off)
            if setup.internalTrig:
                self.radac.control(crRadacIntTrgSel,on)
            else:
                self.radac.control(crRadacIntTrgSel,off)
            
            # Set various other radac registers
            self.radac.writeRegister('wrap',0)
            self.radac.writeRegister('rxattenuation',setup.rxAttenuation)
            
            # Misc radac configuration
            self.radac.setTxEnabled(setup.txEnabled)
            self.log.info('TxEnabled: %i' % (setup.txEnabled))
            
            self.radac.control('syncrftr',setup.syncrftr)
            self.log.info('TR synced to RF: %i' % (setup.syncrftr))
            
            # set default ipp mask
            self.setIppMask(setup.ippMask)
            
            # set output mask
            self.radac.writeRegister('outmask',setup.outputMask)
            
            if setup.controlTr:
                # pulsewidth control of amisr up-converter TR signal enabled
                self.radac.control('controltr',1)
                self.radac.writeRegister('pulseinc',int(setup.widthInc*1e-6/20e-9))
                self.radac.writeRegister('cntlwidth',int(setup.controlWidth*1e-6/20e-9))
            else:
                self.radac.control('controltr',0)    
            
            # configure use special beamcodes control bit
            self.radac.control(crUseSpecialBeamcodes,setup.useSpecialBeamcodes)    
            
            # Stop TU
            self.radac.control(crRunTu,off)
            
            # Load TU file
            self.radac.loadTu(setup.tuImage)
            
            # Start TU
            self.radac.control(crRunTu,on)
            self.log.info('Tu file loaded')
            
            # Load integrator
            self.integrator.resetModes()
            intconf = {'npulsesint':setup.nPulsesInt,'maxsamples':setup.maxSamples,'modes':[]}
            for modeconf in setup.modes:
                m = mode(self.h5Buffer,modeconf.pars)
                intconf['modes'].append(m)
            self.integrator.configure(intconf)
            
            # write relevant info to h5buffer
                        
            # Setup files 
            self.h5Buffer.setStatic('/Setup/Tufile',bconf.text(setup.tuFilename))
            self.h5Buffer.setStatic('/Setup/RxConfigfile',bconf.text(setup.rxcFile))
            self.h5Buffer.setStatic('/Setup/RxFilterfile',bconf.text(setup.filterFile))
            self.h5Buffer.setStatic('/Setup/TxConfigfile',setup.txConfigText)
            self.h5Buffer.setStatic('/Setup/Beamcodefile',bconf.text(setup.beamcodesFilename))
            
            # Rx parameters
            self.h5Buffer.setStatic('/Rx/Bandwidth',bconf['%s/bandwidth' % (setup.rxNode)])
            self.h5Buffer.setAttribute('/Rx/Bandwidth/Unit','Hz')
            self.h5Buffer.setStatic('/Rx/FilterDelay',bconf['%s/filterdelay' % (setup.rxNode)])
            self.h5Buffer.setAttribute('/Rx/FilterDelay/Unit','s')
            self.h5Buffer.setStatic('/Rx/FilterRangeDelay',bconf['%s/filterrangedelay' % (setup.rxNode)])
            self.h5Buffer.setAttribute('/Rx/FilterRangeDelay/Unit','m')
            self.h5Buffer.setStatic('/Rx/FilterTaps',bconf['%s/filtertaps' % (setup.rxNode)])
            self.h5Buffer.setStatic('/Rx/SampleRate',bconf['%s/samplerate' % (setup.rxNode)])
            self.h5Buffer.setAttribute('/Rx/SampleRate/Unit','Hz')
            self.h5Buffer.setStatic('/Rx/SampleTime',bconf['%s/sampletime' % (setup.rxNode)])
            self.h5Buffer.setAttribute('/Rx/SampleTime/Unit','s')
            self.h5Buffer.setStatic('/Rx/SampleSpacing',bconf['%s/samplespacing' % (setup.rxNode)])
            self.h5Buffer.setAttribute('/Rx/SampleSpacing/Unit','m')
            
            # Tu parameters
            self.h5Buffer.setStatic('/Tu/FrameTime',bconf.parameter('frametime',setup.id)/1e6)
            self.h5Buffer.setAttribute('/Tu/FrameTime/Unit','s')
            self.h5Buffer.setStatic('/Tu/TrDuty',bconf.parameter('trduty',setup.id)*100.0)
            self.h5Buffer.setAttribute('/Tu/TrDuty/Unit','%')
            
            self.sync.includeSignal('go')
        except Exception,inst:
            self.log.exception(inst)
        
    def outputFolder(self,outputfolder):
        # try to create output folder in case it is not there
        try:
            os.makedirs(outputfolder)
        except:
            # somebody else has created it
            pass
            
        self.datapath = outputfolder
        self.log.info('Data output folder: %s',self.datapath)
        
        
    
    # Misc helper functions
    
    def getVersion(self):
        ident = self.identity
        try:
            indxver = ident.index('version:')
            sv = ident[indxver:]
            ssv = sv.split(',')[0]
            ver = ssv.split(':')[1]
            return ver.strip()
        except:
            return 'unknown'
    
    def readFile(self,filepath):
        f = open(filepath,'r')
        return f.read()
        
    # Boi and Eoi handling
    def boi(self,vars={}):
        t = self.radac.getTime()
        self.h5Buffer.setBoi('/Time/RadacTime',t['radactime'])
        self.h5Buffer.setBoi('/Time/RadacTimeString',t['radactimestring'])
        self.h5Buffer.setBoi('/Time/MatlabTime',t['matlabtime'])
        self.h5Buffer.setBoi('/Time/UnixTime',t['unixtime'])
        self.h5Buffer.setBoi('/Time/Synchronized',t['sync'])
        self.h5Buffer.boi(vars)
        
    def eoi(self,vars={}):
        # t = self.radac.getTime()
        # self.h5Buffer.setEoi('/Time/RadacTime',t['radactime'])
        # self.h5Buffer.setEoi('/Time/RadacTimeString',t['radactimestring'])
        # self.h5Buffer.setEoi('/Time/MatlabTime',t['matlabtime'])
        # self.h5Buffer.setEoi('/Time/UnixTime',t['unixtime'])
        # self.h5Buffer.setEoi('/Time/Synchronized',t['sync'])
        self.h5Buffer.eoi(vars)
        
    def integrating(self,event):
        if event[0] == '+':
            self.sync.includeSignal('integrating')
        elif event[0] == '-':
            self.sync.excludeSignal('integrating')
            
            # process synchronized settings if any
            while True:
                try:
                    func,pars = self.syncSettings.get(False)
                    func(pars)
                except Empty:
                    break
                except Exception,inst:
                    self.log.exception(inst)
    
    # Xmlrpc exported functions
    
    def startIntegrator(self,timetag):
        self.integrator.start(timetag)
        
    def stopIntegrator(self,timetag):
        self.integrator.stop(timetag)
        
    def startIntegration(self,vars={}):
        if not self.integrator.busy():
            self.h5Buffer.synchronize()
            self.boi(vars)
            self.integrator.start()
            self.sync.includeSignal('integrating')
        else:
            raise integratorInfo('Integrator Busy')
                        
    def saveRecord(self,vars={}):
        self.log.debug('saveRecord')
        self.h5Buffer.swap()
        self.eoi(vars)
        if not self.dataStorage.save(self.h5Buffer):
            self.log.error('Data storage thread too slow')
            
    def setFilename(self,filename,delete=False):            
        if delete:
            if os.path.exists(filename):
                os.unlink(filename)
        self.dataStorage.setFilename(filename)
        
    def setFilenumber(self,filenumber):
        try:
            self._filenumber = filenumber
            snum = '%07d' % (filenumber)
            filename = os.path.join(self.datapath,'d%s.dt%d.h5' % (snum,self.id))
            self.log.info('Filename: %s' % (filename))
            self.setFilename(filename)
        except Exception,inst:
            self.log.exception(inst)
    
    def setIppMask(self,mask):
        if mask < 0:
            mask = 0
        else:
            mask |= 0x80000000
            
        self.radac.writeRegister('ippmask',mask)
        return 1
        
    def setTrackInfo(self,track):
        try:
            modes = self.integrator.config['modes']
            for m in modes:
                if m.pars['mode'].lower() == 'track':
                    m.ext.setTrackInfo(m.index,track)
        except Exception,value:
            self.log.exception(value)
        
        
    def setRf(self,flag):
        self.radac.setRf(flag)
        
    def setRecord(self,flag):
        self.dataStorage.setRecord(flag)
        # notify synchronizer of change
        if flag:
            self.sync.includeSignal('recording')
        else:
            self.sync.excludeSignal('recording')
        
        
    def setDisplayRecord(self,flag):
        self.dataStorage.setDisplayRecord(flag)
        # notify synchronizer of change
        if flag:
            self.sync.includeSignal('writedisplayrecord')
        else:
            self.sync.excludeSignal('writedisplayrecord')
        
    def loadBeamcodes_old(self,filename):
        setup = self.setups[self._setup]
        
        bconf = self.binConfig
        setup.beamcodesFilename = filename
        setup.beamcodes = bconf.beamcodes(bcfile=filename)
        
    def loadBeamcodes(self,filename):
        if self.integrator.busy():
            self.syncSettings.put((self.loadBeamcodes,filename))
            self.sync.excludeSignal('go')
            return 2
            
        id = self._setup
        setup = self.setups[id]
        bconf = self.binConfig
        try:
            allcodes = bconf.beamcodes(bcfile=filename)
        except KeyError:
            bconf = self.reloadConfig()
            allcodes = bconf.beamcodes(bcfile=filename)
            
        self.radac.loadBeamcodes(allcodes)
        for modeconf in setup.modes:
            modegroup = int(modeconf.pars['modegroup'])
            tufn = modeconf.pars['tufile']
            try:
                modeconf.pars['beamcodes'] = bconf.beamcodes(tufn,modegroup,filename)
            except:
                modeconf.pars['beamcodes'] = bconf.beamcodes(tufn,modegroup)

        # Reload integrator
        self.integrator.resetModes()
        intconf = {'npulsesint':setup.nPulsesInt,'maxsamples':setup.maxSamples,'modes':[]}
        for modeconf in setup.modes:
            m = mode(self.h5Buffer,modeconf.pars)
            intconf['modes'].append(m)
        self.integrator.configure(intconf)
            
        self.h5Buffer.setStatic('/Setup/Beamcodefile',bconf.text(filename))
        
        self.sync.includeSignal('go')
            
        
        
        
    def status(self):
        sta = {}
        sta['dtc'] = self.id
        sta['integratorbusy'] = self.integrator.busy()
        sta['txenabled'] = self.radac.isTxEnabled()
        sta['rf'] = self.radac.isRfOn()
        sta['record'] = self.dataStorage.getRecord()
        sta['writedisplayrecord'] = self.dataStorage.getDisplayRecord()
        sta['controlregister'] = self.radac.controlBitsOn()
        sta['filename'] = self.dataStorage.filename
        sta['nrecords'] = self.dataStorage.nrecs
        
        try:
            txf = self.radac.txFrequencies()
            sta['tx%dlo0' % (self.id)] = float(round(txf[0]))
            sta['tx%dlo1' % (self.id)] = float(round(txf[1]))
            sta['tx%dlo2' % (self.id)] = float(round(txf[2]))
            sta['tx%dlo3' % (self.id)] = float(round(txf[3]))
        except:
            pass
        
        rxf = self.radac.rxFrequency()
        sta['rx%dlo0' % (self.id)] = float(round(rxf))
        rxa = self.radac.readRegister('rxattenuation') & 0xf
        sta['rx%dattn' % (self.id)] = rxa       
        return sta
                
    def settings(self,settings):
        try:
            self.radac.writeRegister('rxattenuation',int(settings['rxattenuation']))
            self.radac.rxFrequency(float(settings['rxfrequency']))
            self.radac.txFrequencies([float(settings['txfrequency0']),float(settings['txfrequency1']),
                                 float(settings['txfrequency2']),float(settings['txfrequency3'])])
        except Exception,inst:
            self.log.exception(inst)
            
    def safeSettings(self,settings):
        delayed = settings.get('delayed',False)
        if delayed and self.integrator.busy():
            self.syncSettings.put((self.safeSettings,settings))
            self.sync.excludeSignal('go')
            return 2
            
        # load settings
        if settings.has_key('record'):
            self.setRecord(settings['record'])    
            
        if settings.has_key('writedisplayrecord'):
            self.setDisplayRecord(settings['writedisplayrecord'])
                
        key = 'rx%dattn' % (self.id)
        if settings.has_key(key):
            attn = int(settings[key])
            self.radac.writeRegister('rxattenuation',attn)
            
        key = 'rx%dlo0' % (self.id)
        if settings.has_key(key):
            freq = float(settings[key])
            self.radac.rxFrequency(freq)
            
        key = 'txenabled'    
        if settings.has_key(key):
            self.radac.setTxEnabled(settings[key])
            
        freq = [-1,-1,-1,-1]
        for p in range(4):
            key = 'tx%dlo%d' % (self.id,p)
            if settings.has_key(key):
                f = settings[key]
                freq[p] = float(f)
        if max(freq) >= 0:
            self.radac.txFrequencies(freq)
            
        self.sync.includeSignal('go')    
        self.log.info('sync settings applied')         
        return 1    
            
                
    def radacEvent(self,event,*pars,**dpars):
        # if event == 'lowlevelcheck':
            # self.lowLevelCheck(*pars,**dpars)
        if event == 'rxfrequency':
            sf = '%8.0f' % (pars[0])
            f = float(sf)
            self.log.info('Rx frequency changed: %f' % (f))
            self.sync.variable('rx%dlo0' % (self.id),f)
        elif event == 'rxattenuation':
            self.log.info('Rx attenuation changed: %d' % (pars[0]))
            self.sync.variable('rx%dattn' % (self.id),pars[0])
        elif event == 'txfrequencies':
            self.log.info('txfrequencies changed')
            txfs = pars[0]
            for n,f in enumerate(txfs):
                sf = '%8.0f' % (f)
                f = float(sf)
                v = 'tx%dlo%d' % (self.id,n)
                self.sync.variable(v,f)
                self.log.info('%s changed to: %s' % (v,sf))
        elif event == 'txenabled':
            flag = pars[0]
            self.sync.variable('tx%denabled' % (self.id),flag)
            
    # Integration finallize worker thread
    def intFinishThread(self):
        while not self.threadShutdown.isSet():
            try:
                missedpulses,intcount,startmsw,startlsw,endmsw,endlsw = self.intQueue.get(True,0.1)
                ts = time.clock()
                st = self.radac.getTime([startmsw,startlsw])
                et = self.radac.getTime([endmsw,endlsw])
                print 'Starttime: %s, Endtime: %s' % (st['radactimestring'],et['radactimestring'])
                
                # load integration start time
                self.h5Buffer.h5Dynamic.setOutBoi('/Time/RadacTime',st['radactime'])
                self.h5Buffer.h5Dynamic.setOutBoi('/Time/RadacTimeString',st['radactimestring'])
                self.h5Buffer.h5Dynamic.setOutBoi('/Time/MatlabTime',st['matlabtime'])
                self.h5Buffer.h5Dynamic.setOutBoi('/Time/UnixTime',st['unixtime'])
                self.h5Buffer.h5Dynamic.setOutBoi('/Time/Synchronized',st['sync'])
                
                # load integration end time
                self.h5Buffer.h5Dynamic.setOutEoi('/Time/RadacTime',et['radactime'])
                self.h5Buffer.h5Dynamic.setOutEoi('/Time/RadacTimeString',et['radactimestring'])
                self.h5Buffer.h5Dynamic.setOutEoi('/Time/MatlabTime',et['matlabtime'])
                self.h5Buffer.h5Dynamic.setOutEoi('/Time/UnixTime',et['unixtime'])
                self.h5Buffer.h5Dynamic.setOutEoi('/Time/Synchronized',et['sync'])
                
                
                # load missed pulses
                if missedpulses > 0:
                    self.log.error('Missed pulses: %i' % (missedpulses))
                self.h5Buffer.h5Dynamic.setOut('/Integration/MissedPulses',missedpulses)
                
                
                # collect external information for data files
                tsub = time.clock()
                vars = self.sync.getVariables()
                esub = time.clock()-tsub
                self.log.info('Gathering sync variables: %3.2f [secs]' % (esub))
                
                # collect info from proxies
                self.proxies.storeData(self.h5Buffer,st['radactime']/1e6,et['radactime']/1e6,vars)
                
                # Check to see if filenumber should be incremented
                nfile,nrecs = self.dataStorage.nRecords()
                if nrecs >= self.recordsPerFile:
                    self.setFilenumber(self._filenumber+1)
                    nfile,nrecs = self.dataStorage.nRecords()
                  
                tsub = time.clock()
                self.sync.info(nfile,nrecs,intcount,st['radactime']/1e6)
                esub = time.clock()-tsub
                self.log.info('Sending info to sync: %3.2f [secs]' % (esub))
            
            
                # Save data
                self.dataStorage.save(self.h5Buffer)
                while not self.dataStorage.idle():
                    time.sleep(0.01)
                self.state.Exclude('savingdata')
                et = time.clock()-ts
                    
                self.log.info('Integration completion time: %3.2f [secs]' % (et))
                tsub = time.clock()
                self.sync.variable('%s.intcomptime' % (self.name),et)
                esub = time.clock()-tsub
                self.log.info('Sending int comp time to sync: %3.2f [secs]' % (esub))
            except Empty:
                self.state.Exclude('savingdata')
                continue
            except Exception,inst:
                self.log.exception(inst)
                
        self.log.info('intFinishThread shutdown')        
        
                        
                           
    # This routine is called from the integrator upon end of integration
    def integrationDone(self,fault,msg,skip,missedpulses,intcount,startmsw,startlsw,endmsw,endlsw):
        if fault > 0:
            self.log.critical('Integrator error: %x, message: %s' % (fault,msg))
            self.sync.excludeSignal('go')
            return
            
        if skip:
            return  # if we have a truncate integration drop it!
            
        # swap buffer and load queue to finalize integration    
        if not self.state.HasSignal('savingdata'):
            self.state.Include('savingdata')
            self.h5Buffer.swap()  
            self.intQueue.put([missedpulses,intcount,startmsw,startlsw,endmsw,endlsw]) 
        else:
            self.log.error('Data storage too slow, skipping record')
                    
    def lowLevelCheck(self,tuimage=None,tximage=None):
        self.log.debug('Performing low level system check')
        try:
            systemCheck.systemCheck(tuimage,tximage,self.limits)
            self.log.debug('Low level system check ok')
        except systemCheckError,inst:
            self.radac.control('run',False) # Make sure TU is stopped
            self.log.error(inst.message)
            self.sync.hardError(inst.message)
        except Exception,inst:
            self.log.exception(inst)

    def evStartIntegration(self):
        self.sync.includeSignal('integrating')
        
    def evEndIntegration(self):
        self.sync.excludeSignal('integrating')
            

# Test code        
def test(self):
        # Test code used to check if there might be a speed issue doing integrations in python
        
        df = 'c:/tmp/testdata.h5'
        if os.path.exists(df):
            os.unlink(df)
        self.dataStorage.setFilename(df)
        
        ## Initialize modes
        npi = 500
        maxSamples = 3100
        modes = []
        pars0 = {'mode':'s','name':'Data','npulsesint':npi,'nradacheaderwords':32,
                 'modegroup':1,'indexsample':0,'nlags':120,
                 'ngates':2048,'beamcodes':[int('0x8001',16),int('0x8002',16)],
                 'firstrange':-50e3,'sampletime':30.0e-6,'samplespacing':4496.8869,
                 'subint':10,'substep':10}
        m0 = mode(self.h5Buffer,pars0)
        modes.append(m0)
        pars1 = pars0.copy()
        pars1['name'] = 'Noise'
        m1 = mode(self.h5Buffer,pars1)
        # modes.append(m1)
        
        ## Configure integrator
        self.integrator.configure({'npulsesint':npi,'maxsamples':maxSamples,'modes':modes})
        # self.integrator.start()
        # while self.integrator.busy():
            # self.log.info('integrator busy: %i' % (self.integrator.busy()))
            # sleep(1)
        # self.integrator.shutdown()
        
        while True:
            try:
                if not self.integrator.busy():
                    self.integrator.start()
            except Exception,value:
                self.log.critical(value)
                break    
        self.integrator.shutdown()

        
