"""
This class holds all the info needed to configure
a given setup from an experiment file

"""

import logging
import os
from pylib.Inifile import Inifile
from pylib.TuFile import TuFile
from pylib.rxclasses import rxConfig
from pylib.txclasses import txConfig
from pylib.beamcodeclasses import beamcodesConfig
from modeclasses import modeConfig

class setup:
    def __init__(self,exp,id,extconf):
        self.log = logging.getLogger('setup')
        self.exp = exp
        self.id = int(id)
        self.extconf = extconf
        
        # shortcut to self.exp.binConfig
        bconf = self.exp.binConfig
        
        # Read config values from expfile
        dtc = self.exp.dtc
        commonsec = 'Common Mode:%d' % (self.id)
        mainsec = '%s Mode:%d' % (self.exp.dtc,self.id)
        
        # Common stuff
        self.nPulsesInt = self.exp.experimentConfig.getint('common parameters','npulsesint',0)
        self.ippMask = self.exp.experimentConfig.getint('common parameters','ippmask',1)
        self.maxSamples = self.exp.experimentConfig.getint(mainsec,'maxsamples',50000)
        outputmask = self.exp.systemConfig.getint('common parameters','outputmask',0xffffffff)
        self.outputMask = self.exp.experimentConfig.getint('common parameters','outputmask',outputmask)
        self.useSpecialBeamcodes = self.exp.experimentConfig.getint('common parameters','usespecialbeamcodes',0)
        
        # Trig

        trig = self.exp.systemConfig.getint(dtc,'internaltrig',0)
        self.internalTrig = self.exp.experimentConfig.getint(dtc,'internaltrig',trig)
        
        
        # Synchronize RF and TR (remove TR when RF is turned off)
        self.syncrftr = self.exp.systemConfig.getint('common parameters','syncrftr',1)
        
        # Write display record
        wdr = self.exp.systemConfig.getint(dtc,'writedisplayrecord',True)
        wdr = self.exp.experimentConfig.getint(dtc,'writedisplayrecord',wdr)
        wdr = self.exp.experimentConfig.getint(mainsec,'writedisplayrecord',wdr)
        wdr = self.exp.experimentConfig.getint('common parameters','writedisplayrecord',wdr)
        self.writeDisplayRecord = wdr
        
        # Amisr upconverter TR pulse width control
        controltr = self.exp.systemConfig.getint(dtc,'controltr',0)
        self.controlTr = self.exp.experimentConfig.getint(dtc,'controltr',controltr)
        if self.controlTr:
            widthinc = self.exp.systemConfig.getfloat(dtc,'widthincperpulse',0.5)
            self.widthInc = self.exp.experimentConfig.getfloat(dtc,'widthincperpulse',widthinc)
            cntlwidth = self.exp.systemConfig.getint(dtc,'controlwidth',500)
            self.controlWidth = self.exp.experimentConfig.getint(dtc,'controlwidth',cntlwidth)
            
                
        # TxEnabled
        self.txEnabled = self.exp.experimentConfig.getint(mainsec,'txenabled',False)
        
        
        # Tu handling
        self.tuFilename = tuFile = self.exp.experimentConfig.get(mainsec,'tufile','')
        self.tuImage = bconf.binImage(tuFile)
        self.log.info('Reading tu image: %s' % (tuFile))
        
        # Rx handling
        self.rxAttenuation = self.exp.experimentConfig.getint(mainsec,'rxattenuation',0)
        self.rxFrequency = self.exp.experimentConfig.getfloat(mainsec,'rxfrequency',0)
        self.rxcFile = rxcFile = self.exp.experimentConfig.get(mainsec,'rxconfig','')
        rxc = bconf.inifile(rxcFile)
        self.filterFile = rxc.get('filter','file','')
        self.rxNode = bconf.binPath(rxc.Filename)
        self.log.info('Reading Rx config file: %s' % (rxcFile))
        
        # Tx handling
        txcFile = self.exp.experimentConfig.get(mainsec,'txconfig','')
        if txcFile <> '':
            self.log.info('Reading Tx image: %s' % (txcFile))
            self.txImage = bconf.binImage(txcFile)
            self.txConfigText = bconf.text(txcFile)
        else:
            # No txc file specified. Read default values from system.ini file
            self.log.info('Reading default Tx image')
            self.txImage = bconf.binImage('default.tx.image')
            self.txConfigText = bconf.text('default.tx.config')

        # Beam codes
        bcoFile = self.exp.experimentConfig.get(commonsec,'beamcodefile','')
        if bcoFile == '':
            bcoFile = self.exp.experimentConfig.get(mainsec,'beamcodefile')
        self.beamcodesFilename = bcoFile    
        self.beamcodes = bconf.beamcodes(bcfile=bcoFile)    
            
        # Modes
        self.modes = []
        smodes = self.exp.experimentConfig.get(mainsec,'modes','')
        modenames = smodes.split(' ')
        for m in modenames:
            self.log.info('Reading config info for mode: %s' % (m))
            modeconf = modeConfig(self.exp.experimentConfig,self.exp.dtc,self.id,m)
            
            modegroup = int(modeconf.pars['modegroup'])
            try:
                bcp = bconf.beamcodesPath(tuFile,modegroup,bcoFile)
                modeconf.pars['beamcodes'] = bconf[bcp]
            except:
                # Alternate beamcode used for this mode group
                bcp = bconf.beamcodesPath(tuFile,modegroup)
                modeconf.pars['beamcodes'] = bconf[bcp]
                
            modeconf.pars['activepulses'] = bconf['%s/activepulses' % (bcp)]
            modeconf.pars['totalpulses'] = bconf['%s/totalpulses' % (bcp)]
                        
            # Add special info from other places
            
            # Common stuff
            modeconf.pars['npulsesint'] = self.nPulsesInt
            modeconf.pars['nradacheaderwords'] = extconf['nradacheaderwords']
            
            # Rx stuff
            modeconf.pars['filterrangedelay'] = bconf['%s/filterrangedelay' % (self.rxNode)]
            modeconf.pars['sampletime'] = bconf['%s/sampletime' % (self.rxNode)]
            modeconf.pars['samplespacing'] = bconf['%s/samplespacing' % (self.rxNode)]
            
            self.modes.append(modeconf)

        
    def loadBeamcodes(self,filename):
        if os.path.exists(filename):
            bcoFile = filename
        else:
            bcoFile = os.path.join(self.exp.experimentConfigPath,filename)  
        self.log.debug(bcoFile)
        self.beamcodes = beamcodesConfig(bcoFile)   
        # Change beamcodes in mode configurations
        for modeconf in self.modes:
            self.usedBeamcodes(modeconf)
           
    def usedBeamcodes(self,modeconf):
    
        """
        This function is supposed to find all uniq beamcodes used in a modegroup
        so they can be passed down to the mode dll where they are used for
        storage allocation and data sorting. It does that using modegroup info
        from the exp file, the TU file and currently configured beamcodes table.
        """
        
        modegroup = int(modeconf.pars['modegroup'])
        indxHeaders = self.tu.Index(bit=11)
        npulses = len(indxHeaders)
        indxBct = self.tu.Index(bit=12)
        indxMg = self.tu.Index(bit=-2,specialwordaddress=3,specialword=modegroup)
        bcs = []
        for mg in indxMg:
            indx = 0
            for bct in indxBct:
                if bct > mg:
                    break
                else:
                    indx += 1
            bcs.append(self.beamcodes.codes[indx])
            
        # Find uniq codes and sort them
        sbcs = []
        for c in bcs:
            if c not in sbcs:
                sbcs.append(c)
        sbcs.sort()
        modeconf.pars['beamcodes'] = sbcs
        modeconf.pars['activepulses'] = len(indxMg)
        npi = int(modeconf.pars['npulsesint'])
        ap = modeconf.pars['activepulses']
        modeconf.pars['totalpulses'] = npi*ap/len(indxHeaders)
        if (npi % npulses):
            self.log.error('npulsesframe: %d is not a multiple of npulsesint: %d' % (npulses,npi))
        
                
        
