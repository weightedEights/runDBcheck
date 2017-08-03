"""
This module takes care of storing data to files in hdf5 format.

History:
    Initial implementation
    Date:       20070117
    Author:     John Jorgensen
    Company:    SRI International
    
"""

version = '1.0.0'

import logging
import os
import threading
from threading import Lock
from pylib.sysutils import LockableSet,LockedDict,LockedBuffer,Signal
from tables import openFile,Atom,StringAtom
import numpy as np
from copy import deepcopy
import time

ARRAYFLAVOR = 'numpy'
TYPETRANSLATE = {'bool'       : 'Bool',\
                 'int8'       : 'Int8', \
                 'int16'      : 'Int16',\
                 'int32'      : 'Int32',\
                 'int64'      : 'Int64',\
                 'uint8'      : 'UInt8',\
                 'uint15'     : 'UInt16',\
                 'uint32'     : 'UInt32',\
                 'uint64'     : 'UInt64',\
                 'float32'    : 'Float32',\
                 'float64'    : 'Float64',
                 'complex64'  : 'Complex32',\
                 'complex128' : 'Complex64'}

def typeTranslate(typename):
    try:
        return TYPETRANSLATE[typename]
    except:
        return typename.capitalize()
        
class dynamicBuffer:
    def __init__(self,init={}):
        self.data = [LockedDict(),LockedDict()]
        
    def __getitem__(self,index):
        return self.data[1][index]
        
    def __setitem__(self,index,value):
        self.data[0][index] = value
        
    def __str__(self):
        return '%s\n%s\n' % (str(self.data[0]),str(self.data[1]))
        
    def __delitem__(self,key):    
        if self.data[0].has_key(key):
            del self.data[0][key]
        if self.data[1].has_key(key):
            del self.data[1][key]
        
    def keys(self):
        keys = self.data[0].keys()
        uniq = [k for k in self.data[1].keys() if not k in keys]
        keys.extend(uniq)
        return keys
        
    def setIn(self,index,data):
        self.data[0][index] = data
        
    def setBoi(self,index,data,offset=0):
        if self.data[0].has_key(index):
            self.data[0][index][offset] = data
        else:
            self.data[0][index] = np.array([data,data])
        
    def setOut(self,index,data):
        if self.data[1].has_key(index):
            self.data[1][index][0] = data
        else:    
            self.data[1][index] = np.array([data])
        
    def setEoi(self,index,data,offset=1):
        if self.data[1].has_key(index):
            self.data[1][index][offset] = data
        else:
            self.data[1][index] = np.array([data,data])
            
    def setOutBoi(self,index,data):
        # self.setEoi(index,data,0)
        if isinstance(data,list):
            self.data[1][index][0:len(data)] = data
        else:    
            self.data[1][index][0] = data
        
    def setOutEoi(self,index,data):
        # self.setEoi(index,data,1)
        if isinstance(data,list):
            self.data[1][index][len(data):] = data
        else:    
            self.data[1][index][1] = data
            
    def setOutBoth(self,index,data):
        if isinstance(data,list):
            self.data[1][index][0:len(data)] = data
            self.data[1][index][len(data):] = data
        else:    
            self.data[1][index][0] = data
            self.data[1][index][1] = data        
        
        
    def items(self):
        return self.data[1].items()
        
    def has_key(self,key):
        return self.data[1].has_key(key)
        
    def register(self,index,data):
        self.data[0][index] = data
        self.data[1][index] = deepcopy(data)
        
    def synchronize(self):
        for key,item in self.data[0].items():
            if not self.data[1].has_key(key):
                self.data[1][key] = deepcopy(item)
        
    def clear(self):
        self.data[0].clear()
        self.data[1].clear()
        
    def swap(self):
        self.data.reverse() 
        
        
class h5Buffer:
    def __init__(self):
        self.log = logging.getLogger('h5Buffer')
        
        # Exclude display arrays 
        self.excludeArrays = []
        
        # various h5 buffers
        self.h5Groups = LockedBuffer()
        self.h5Attributes = LockedDict()
        self.h5Static = LockedDict()
        self.h5Dynamic = dynamicBuffer()
        self.async = LockedDict()
        
        # Map methods to dynamicBuffer
        self.swap = Signal()
        self.swap.connect(self.h5Dynamic.swap)
        self.synchronize = Signal()
        self.synchronize.connect(self.h5Dynamic.synchronize)
        
    def shutdown(self):
        self.swap.disconnectAll()
        self.synchronize.disconnectAll()
                
    def copy(self):
        buf = h5Buffer()
        for g in self.h5Groups:
            buf.h5Groups.Append(g)
        for path,data in self.h5Static.items():
            buf.h5Static[path] = deepcopy(data)
        for path,data in self.h5Dynamic.items():
            buf.h5Dynamic[path] = deepcopy(data)
        buf.h5Dynamic.swap()
        for path,data in self.h5Attributes.items():
            buf.h5Attributes[path] = deepcopy(data)
        return buf
            
        
    def setGroup(self,path):
        self.log.debug('Group: %s' % (path))
        pl = path.split('/')[1:]
        for n in range(len(pl)):  ## run throuh subpaths and create anything that doesn't exit all ready
            sp = '/'.join(pl[0:n+1])
            if not sp.startswith('/'):
                sp = '/'+sp
            try:
                indx = self.h5Groups.Index(sp)
            except:
                self.h5Groups.Append(sp)
        
    def setAttribute(self,path,data):
        self.log.debug('Attribute: %s' % (path))
        self.h5Attributes[path] = data

    def setStatic(self,path,data):
        self.log.debug('Static: %s' % (path))
        dp,dn = os.path.split(path)
        self.setGroup(dp) ## Create groups automatically
        
        self.h5Static[path] = data
        
    def setDynamic(self,path,data):
        self.log.debug('Dynamic: %s' % (path))
        dp,dn = os.path.split(path)
        self.setGroup(dp) ## Create groups automatically
        
        self.h5Dynamic[path] = np.array(data)
        # self.h5Dynamic.setOut(path,data)
            
    def setBoi(self,path,data):
        self.h5Dynamic.setBoi(path,data)
        # try:
            # self.lock.acquire()
            # self.h5Dynamic.setBoi(path,data)
        # finally:
            # self.lock.release()
            
    def setEoi(self,path,data):
        self.h5Dynamic.setEoi(path,data)
        # try:
            # self.lock.acquire()
            # self.h5Dynamic.setEoi(path,data)
        # finally:
            # self.lock.release()
            
    def setAsync(self,path,varnames,data,attr={}):
        dp,dn = os.path.split(path)
        self.setGroup(dp) ## Create groups automatically
        
        try:
            self.log.debug('ASync: %s, var names: %s' % (path,str(varnames)))
            # self.lock.acquire()
            self.async[path] = varnames
            self.h5Dynamic[path] = np.array(data)
            for path,value in attr.items():
                self.h5Attributes[path] = value
        except Exception,inst:
            self.log.exception(inst)
        # finally:
            # self.lock.release()
            
    def boi(self,vars):
        try:
            # self.lock.acquire()
            for path,varnames in self.async.items():
                for n,varname in enumerate(varnames):
                    try:
                        self.h5Dynamic.setBoi(path,vars[varname],n)
                    except:
                        self.h5Dynamic.setBoi(path,0,n)
        except Exception,inst:
            self.log.exception(inst)
        # finally:
            # self.lock.release()
                
    def eoi(self,vars):
        try:
            # self.lock.acquire()
            for path,varnames in self.async.items():
                offset = len(varnames)
                for n,varname in enumerate(varnames):
                    try:
                        self.h5Dynamic.setEoi(path,vars[varname],offset+n)
                    except:
                        self.h5Dynamic.setEoi(path,0,offset+n)
        except Exception,inst:
            self.log.exception(inst)
        # finally:
            # self.lock.release()
            
    def includeH5File(self,path,filepath):
        self.log.info('Including hdf5 file: %s' % (filepath))
        try:
            h5 = openFile(filepath,'r')
            for node in h5.walkNodes():
                # copy user attributes for everybody
                attrs = node._v_attrs
                attrnames = attrs._v_attrnamesuser
                for an in attrnames:
                    av = h5.getNodeAttr(node._v_pathname,an)
                    self.setAttribute('/'.join([node._v_pathname,an]),av)
                if node._v_name <> '/':
                    # other groups and datasets 
                    newpath = ''.join([path,node._v_pathname]) 
                    if node._c_classId == 'GROUP':
                        self.setGroup(newpath)
                    elif node._c_classId == 'ARRAY':
                        self.setStatic(newpath,node.read())
        except Exception,inst:
            self.log.exception(inst)
        finally:
            try:
                h5.close()
            except:
                pass
                               
    def includeFile(self,path,filepath):
        self.log.info('Including file: %s' % (filepath))
        fp = filepath.lower()
        if (fp.endswith('.h5') or fp.endswith('.hdf5')):
            self.includeH5File(path,filepath)
            
    def excludeArray(self,path):
        self.excludeArrays.append(path.lower())
        
    def arrayExclude(self,path):
        spath = path.split('/')
        rpath = '/%s' % ('/'.join(spath[1:3]))
        return (rpath.lower() in self.excludeArrays)
        
    def isSubPath(self,path,name):
        sn = name.split('/')[1:]
        sp = path.split('/')[1:]
        for p in sp:
            if not p in sn:
                return False
        return True                
        
    def removeBranch(self,path):
        # handle lockeddict and dynamic buffer instances
        arrays = [self.h5Attributes,self.h5Static,self.h5Dynamic,self.async]
        for a in arrays:
            keys = a.keys()
            for k in keys:
                if self.isSubPath(path,k):
                    try:
                        del a[k]
                        self.log.debug('Removed entry: %s' % (k))
                    except:
                        continue
                    
        # handle lockedbuffer instance
        groups = [g for g in self.h5Groups if self.isSubPath(path,g)]
        for g in groups:
            try:
                self.h5Groups.Remove(g)
                self.log.debug('Removed group: %s' % (g))
            except:
                continue

class h5Storage(threading.Thread):
    def __init__(self,name,h5buffer,options=['append','lock']):
        threading.Thread.__init__(self)
        
        self.name = name
        self.log = logging.getLogger(self.name)
        
        self.lock = Lock()
        
        self.filename = ''
        self.nrecs = 0
        self.lockfile = ''
        self.displayFilename = ''
        self.displayLockfile = ''
        
        self.h5Buffer = h5buffer
                
        # Thread state object
        self.state = LockableSet()
        
        # Write options
        # append: default keeps appending to data file
        # lock: writes to lock file and then rename it to filename. If file exists skip write
        # the default is to append to data file
        self.options = LockableSet(options)
        
        # Event object used to suspend thread when not used
        self.event = threading.Event()
        
        self.start()
        
    def shutdown(self):
        self.state.Include('shutdown')
        self.event.set()
        while self.state.HasSignal('running'):
            self.log.info('waiting for run thead to shut down...')
            time.sleep(1)
        
    def idle(self):
        return not self.event.isSet()
        
    def save(self,buffer):
        # if not self.record:  ## skip if record is not True
            # return 1
            
        if self.idle():
            self.h5Buffer = buffer
            self.event.set()
            return 1
        else:
            return 0
            
    def nRecords(self):
        self.lock.acquire()
        try:
            fn = os.path.basename(self.filename)
            sn = int(fn[1:8])
            return sn,self.nrecs
        finally:    
            self.lock.release()
            
            
    def setFilename(self,filename):    
        self.lock.acquire()
        self.filename = filename
        self.lockfile = '.'.join([filename,'lock'])
        try:
            # if file exists 
            # adjust nrecs to number of records all ready in file
            h5 = openFile(self.filename,'r')
            try:
                self.nrecs = h5.root.Time.RadacTime.shape[0]
            finally:    
                h5.close()
        except:
            self.nrecs = 0
        self.lock.release() 
        
    def setDisplayFilename(self,filename):
        self.lock.acquire()
        self.displayFilename = filename
        self.displayLockfile = '.'.join([filename,'lock'])
        self.lock.release()
        
    def setRecord(self,flag):
        if flag:
            self.options.Include('record')
        else:
            self.options.Exclude('record')
            
    def getRecord(self):
        return self.options.HasSignal('record')
            
    def setDisplayRecord(self,flag):
        if flag:
            self.options.Include('displayrecord')
        else:
            self.options.Exclude('displayrecord')
            
    def getDisplayRecord(self):
        return self.options.HasSignal('displayrecord')
        
    def arrayExclude(self,dp):
        
        return True
            
    def create(self,h5,h5buffer,display=False):
        self.log.debug('Creating new file: %s' % (h5.filename))
            ## first create all groups
        for group in h5buffer.h5Groups:
            try:
                if display and h5buffer.arrayExclude(group):
                    continue
                gp,gn = os.path.split(group)
                h5.createGroup(gp,gn)
                self.log.debug('Group created: %s' % (group))
            except Exception,inst:
                self.log.exception(inst)
                self.log.error('Failed on group: %s, name: %s' % (gp,gn))
                continue
            
        ## then all static arrays
        for path,data in h5buffer.h5Static.items():
            try:
                if display and h5buffer.arrayExclude(path):
                    continue
                dp,dn = os.path.split(path)
                h5.createArray(dp,dn,data,'Static array')
                self.log.debug('Static array created: %s' % (path))
            except Exception,inst:
                self.log.exception(inst)
                self.log.error('Failed on group: %s, name: %s' % (dp,dn))
                continue
            
        ## then all dynamic arrays
        for path,rec in h5buffer.h5Dynamic.items():
            try:
                if display and h5buffer.arrayExclude(path):
                    continue
                dp,dn = os.path.split(path)
                data = rec.copy()
                data.shape = (1,)+data.shape  ## add integration dimension to data array
                shape = list(data.shape)
                shape[0] = 0
                # if data.dtype.name.lower().find('string') <> -1:
                    # atom = StringAtom(shape=shape,length=80,flavor=ARRAYFLAVOR)
                # else:
                atom = Atom.from_dtype(data.dtype)
                    # atom = Atom(typeTranslate(data.dtype.name),shape=shape,flavor=ARRAYFLAVOR)
                arr = h5.createEArray(dp,dn,atom,shape)
                arr.flavor='numpy'
                arr.append(data)
                self.log.debug('Dynamic array created: %s' % (path))
            except Exception,inst:
                self.log.exception(inst)
                self.log.error('Failed on group: %s, name: %s' % (dp,dn))
                continue
        
        ## and finally all attributes
        for path,data in h5buffer.h5Attributes.items():
            try:
                if display and h5buffer.arrayExclude(path):
                    continue
                ap,an = os.path.split(path)
                h5.setNodeAttr(ap,an,data)            
                self.log.debug('Attribute created: %s' % (path))
            except Exception,inst:
                self.log.exception(inst)
                self.log.error('Failed on group: %s, name: %s' % (ap,an))
                continue
        
    def append(self,h5,h5buffer):
        self.log.debug('Appending to file: %s' % (h5.filename))
        ## update all dynamic arrays
        for path,rec in h5buffer.h5Dynamic.items():
            try:
                data = rec.copy()
                data.shape = (1,)+data.shape  ## add integration dimension
                arr = h5.getNode(path)
                arr.append(data)
            except Exception,inst:
                self.log.exception(inst)
                dp,dn = os.path.split(path)
                self.log.error('Failed on group: %s, name: %s' % (dp,dn))
                continue
            
    def run(self):
        self.log.info('Running')
        self.state.Include('running')
        while not self.state.HasSignal('shutdown'):
            self.log.debug('Idle')
            self.event.wait() ## Suspend until event is triggered
            self.log.debug('Busy')
            
            if self.state.HasSignal('shutdown'):
                break
            
            self.lock.acquire()
            datafilename = self.filename
            lockfilename = self.lockfile
            dispdatafile = self.displayFilename
            displockfile = self.displayLockfile
            self.lock.release()
            
            # Actual datafile handling
            try:
                # Normal append to data file
                if self.options.HasSignal('record'):
                    if self.options.HasSignal('append'):
                        t = time.clock()
                        try:
                            if os.path.exists(datafilename):
                                h5 = openFile(datafilename,'a')
                                self.append(h5,self.h5Buffer)
                            else:
                                h5 = openFile(datafilename,'a')
                                self.create(h5,self.h5Buffer)
                                
                            self.lock.acquire()
                            self.nrecs += 1
                            self.lock.release()
                        except Exception,inst:
                            self.log.exception(inst)
                        finally:
                            h5.close()
                            self.log.info('Datafile save time: %2.6f' % (time.clock()-t))
                    
                        
                # Write display record via lock file
                if self.options.HasSignal('displayrecord'):
                    if self.options.HasSignal('lock'):
                        if not os.path.exists(dispdatafile):  ## Only write if datafile is not there
                            # if os.path.exists(lockfilename):  ## if there is an old lock file remove it
                                # try:
                                    # os.unlink(lockfilename)
                                # except Exception,inst:
                                    # self.log.exception(inst)
                            t = time.clock()
                            try:
                                h5 = openFile(displockfile,'w')
                                self.create(h5,self.h5Buffer,display=True)
                            except Exception,inst:
                                self.log.exception(inst)
                            finally:
                                try:
                                    h5.close()
                                    os.rename(displockfile,dispdatafile)
                                except Exception,inst:
                                    self.log.exception(inst)
                                self.log.info('Display record save time: %2.6f' % (time.clock()-t))    
                        
            except Exception,value:
                self.log.critical('Exception writing datafile: %s' % (value))
                break
            finally:
                self.event.clear()
        self.log.info('Shutdown')
        self.state.Exclude('running')
                            
            
def h5wait(h5):
    while not h5.idle():
        sleep(0)
            
def h5save(h5):
    h5.save()
    h5wait(h5)
       
if __name__ == '__main__':
    import numpy as np
    from time import sleep
    logging.basicConfig(level=logging.DEBUG)
    
    try:
        fn = 'c:/tmp/h5test.h5'
        if os.path.exists(fn):
            os.unlink(fn)
        
        buf = h5Buffer()
        h5 = h5Storage(buf)
        h5.setFilename(fn)
        h5.record = True
        buf.setAsync('/Async/var1',['v1','v2'],np.array([0,0,0,0]))
        buf.setDynamic('/Tx/Frequency',np.array([410e6,420e6]))
        buf.synchronize()
        print 'sync: ',buf.h5Dynamic
        buf.boi(dict(v1=25,v2=26))
        print 'boi: ',buf.h5Dynamic
        buf.swap()
        print 'swap: ',buf.h5Dynamic
        buf.eoi(dict(v1=27,v2=28))
        print 'eoi: ',buf.h5Dynamic
        h5.save()
        
        buf.setBoi('/Tx/Frequency',430e6)
        buf.boi(dict(v1=35,v2=36))
        print 'boi: ',buf.h5Dynamic
        h5wait(h5)
        buf.swap()
        print 'swap: ',buf.h5Dynamic
        buf.setEoi('/Tx/Frequency',440e6)
        buf.eoi(dict(v1=37,v2=38))
        print 'eoi: ',buf.h5Dynamic
        h5save(h5)
    except Exception,inst:
        logging.exception(inst)
    finally:
        try:
            h5.shutdown()
        except:
            pass
    


