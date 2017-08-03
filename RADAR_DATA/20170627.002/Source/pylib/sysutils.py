#! /usr/bin/env python

"""
This is a library of misc. utility classes

History:

Date:           20051005
Version:        1.0.0.0
Author:         John Jorgensen
Description:    Initial implementation
"""

version = '1.0.0.0'

import sys,types
from threading import Lock
import weakref as wr
import inspect

class LockedBuffer:
    def __init__(self):
        self._lock = Lock()
        self._buffer = []
        
    def __getitem__(self,key):
        try:
            self.Acquire()
            return self._buffer[key]
        finally:
            self.Release()
            
    def __setitem__(self,key,value):
        try:
            self.Acquire()
            self._buffer[key] = value
        finally:
            self.Release()
            
    def Index(self,value):
        try:
            self.Acquire()
            return self._buffer.index(value)
        finally:
            self.Release()
        
        
    def Acquire(self):
        self._lock.acquire()
        return self._buffer
        
    def Release(self):
        # self._buffer = buffer
        self._lock.release()
        
    def Clear(self):
        if not self._lock.locked():
            self._lock.acquire()
            self._buffer = []
            self.Lock.release()
        else:
            self._buffer = []
                
    def Put(self,item):
        buf = self.Acquire()
        buf.append(item)
        self.Release()
        
    def Append(self,item):
        buf = self.Acquire()
        buf.append(item)
        self.Release()
        
    def Get(self):
        buf = self.Acquire()
        try:
            item = buf.pop(0)
        except:
            item = None
        self.Release()
        return item
        
    def Remove(self,key):
        self._lock.acquire()
        try:
            self._buffer.remove(key)
        finally:
            self._lock.release()

LineBuffer = LockedBuffer        
CommandQueue = LockedBuffer
        
class LockableSet:
    def __init__(self,default=[]):
        self._lock = Lock()
        self._set = set(default)
        
    def __call__(self,aset=None):
        if aset is None:
            return self.Acquire()
        else:
            self.Release(aset)
            
    def __str__(self):
        return str(self._set)
            
    def Acquire(self):
        self._lock.acquire()
        return self._set.copy()
        
    def Release(self,aset):
        changes = self._set.copy()
        changes ^= aset
        changed = (changes != set(['Changed'])) and (changes != set([]))
        self._set = aset
        if changed:
            self._set.add('Changed')
        self._lock.release()
        
    def Include(self,signal):
        if isinstance(signal,str):
            signals = [signal]
        else:
            signals = signal
        s = self.Acquire()
        for sig in signals:
            s.add(sig)
        self.Release(s)
        
    def Exclude(self,signal):
        if isinstance(signal,str):
            signals = [signal]
        else:
            signals = signal
        s = self.Acquire()
        for sig in signals:
            try:
                s.remove(sig)
            except:
                continue
        self.Release(s)
        
    def Update(self,signal):
        s = signal[0]
        v = signal[1:]
        if s == '+':
            self.Include(v)
        elif s == '-':
            self.Exclude(v)
            
    def HasSignal(self,signal):
        try:
            self._lock.acquire()
            return (signal in self._set)
        finally:
            self._lock.release()
        
    def Signals(self):
        try:
            self._lock.acquire()
            return list(self._set)
        finally:
            self._lock.release()
            
    def Clear(self):
        try:
            self._lock.acquire()
            self._set = set()
        finally:
            self._lock.release()
        
class StateMachine(LockableSet):

    def __init__(self,default=[]):
        LockableSet.__init__(self,default)
        
        self.Events = []
        self.CmdBuffer = LockedBuffer()
        
    def AddEvent(self,event):
        try:
            self._lock.acquire()
            edge,signals,function = event
            indx = [n for n,e in enumerate(self.Events) if e[2].f == function.im_func]
            # indx = [n for n,e in enumerate(self.Events) if e[2] == function]
            if len(indx) > 0:
                for i in indx:
                    self.Events[i] = (edge,signals,WeakMethod(function))
                    # self.Events[i] = (edge,signals,function)
            else:
                self.Events.append((edge,signals,WeakMethod(function)))
                # self.Events.append((edge,signals,function))
        finally:
            self._lock.release()
        
                
    def Release(self,aset):
        try:
            if aset != self._set:
                changes = aset ^ self._set
                self._set = aset
                for rising,mask,func in self.Events:
                    overlap = mask & changes   # see if there is an overlap between mask and signals that changed
                    if overlap <> set():
                        if rising and mask.issubset(self._set):
                            self.CmdBuffer.Put(func)
                            # if func is not None:
                                # self.CmdBuffer.Put(func())
                            # else:
                                # print 'disappered'
                        elif not rising and not mask.issubset(self._set):
                            self.CmdBuffer.Put(func)
                            # if func is not None:
                                # self.CmdBuffer.Put(func())
                            # else:
                                # print 'disappered'
        finally:
            self._lock.release()
        f = self.CmdBuffer.Get()
        while f is not None:
            f()
            f = self.CmdBuffer.Get()
            
class Stringlist(dict):
    
    def __init__(self):
        dict.__init__(self)
        
    def CommaText(self,pars=None):
        if pars is not None:
            pairs = pars.split(',')
            for p in pairs:
                n,v = p.split('=')
                self[n] = v            
        else:
            s = ''
            for n,v in self.items():
                if s == '':
                    s = str(n)+'='+str(v)
                else:
                    s += ','+str(n)+'='+str(v)
            return s
            
class LockedDict:
    def __init__(self):
        self.Lock = Lock()
        self.Data = {}
        
    def __getitem__(self,key):
        self.Lock.acquire()
        try:
            val = self.Data[key];
        except:
            val = None
        self.Lock.release()
        return val
        
    def __len__(self):
        try:
            self.Lock.acquire()
            return len(self.Data)
        finally:
            self.Lock.release()
            
    def __repr__(self):
        try:
            self.Lock.acquire()
            return str(self.Data)
        finally:
            self.Lock.release()
        
    def __setitem__(self,key,value):
        self.Lock.acquire()
        self.Data[key] = value
        self.Lock.release()
        
    def keys(self):
        try:
            self.Lock.acquire()
            return self.Data.keys()
        finally:
            self.Lock.release()
            
    def items(self):
        try:
            self.Lock.acquire()
            return self.Data.items()
        finally:
            self.Lock.release()
            
    def has_key(self,key):
        try:
            self.Lock.acquire()
            return self.Data.has_key(key)
        finally:
            self.Lock.release()
        
    def __delitem__(self,key):
        try:
            self.Lock.acquire()
            del self.Data[key]
        finally:
            self.Lock.release()
        
    def __len__(self):
        try:
            self.Lock.acquire()
            return len(self.Data)
        finally:
            self.Lock.release()
        
        
    def Items(self):
        self.Lock.acquire()
        items = self.Data.items()
        self.Lock.release()
        return items
        

class Signal:
    """
    Author:  Patrick Chasco
    Created: July 26, 2005

    Purpose: A signals implementation
    
    class Signal

    A simple implementation of the Signal/Slot pattern. To use, simply 
    create a Signal instance. The instance may be a member of a class, 
    a global, or a local; it makes no difference what scope it resides 
    within. Connect slots to the signal using the "connect()" method. 
    The slot may be a member of a class or a simple function. If the 
    slot is a member of a class, Signal will automatically detect when
    the method's class instance has been deleted and remove it from 
    its list of connected slots.
    """
    def __init__(self):
        self.slots = []

        # for keeping references to _WeakMethod_FuncHost objects.
        # If we didn't, then the weak references would die for
        # non-method slots that we've created.
        self.funchost = []

    def __call__(self, *args, **kwargs):
        for i in range(len(self.slots)):
            slot = self.slots[i]
            if slot != None:
                slot(*args, **kwargs)
            else:
                del self.slots[i]
                
    def call(self, *args, **kwargs):
        self.__call__(*args, **kwargs)

    def connect(self, slot):
        self.disconnect(slot)
        if inspect.ismethod(slot):
            self.slots.append(WeakMethod(slot))
        else:
            o = _WeakMethod_FuncHost(slot)
            self.slots.append(WeakMethod(o.func))
            # we stick a copy in here just to keep the instance alive
            self.funchost.append(o)

    def disconnect(self, slot):
        try:
            for i in range(len(self.slots)):
                wm = self.slots[i]
                if inspect.ismethod(slot):
                    if wm.f == slot.im_func and wm.c() == slot.im_self:
                        del self.slots[i]
                        return
                else:
                    if wm.c().hostedFunction == slot:
                        del self.slots[i]
                        return
        except:
            pass

    def disconnectAll(self):
        del self.slots
        del self.funchost
        self.slots = []
        self.funchost = []
        

class _WeakMethod_FuncHost:
    def __init__(self, func):
        self.hostedFunction = func
        
    def func(self, *args, **kwargs):
        self.hostedFunction(*args, **kwargs)

# this class was generously donated by a poster on ASPN (aspn.activestate.com)
class WeakMethod:
	def __init__(self, f):
            self.f = f.im_func
            self.c = wr.ref(f.im_self)
            
	def __call__(self, *args, **kwargs):
            if self.c() is None: 
                return
            self.f(self.c(), *args, **kwargs)

# Misc functions
def UrlParse(url,defport=80):
    res = {}
    s = url.split(':')
    res['protocol'] = s[0]
    if len(s) > 2:
        res['server'] = s[1][2:]
        path = s[2]
        spath = path.split('/')
        res['port'] = int(spath[0])
        res['path'] = '/'+'/'.join(spath[1:])
    else:
        path = s[1]
        spath = path.split('/')
        res['server'] = spath[2]
        if len(spath) > 2:
            res['path'] = '/'+'/'.join(spath[3:])
        else:
            res['path'] = ''
        res['port'] = defport
    return res
    
def urlExtract(field,url):
    if field == 'url':
        surl = url.split(',')
        return surl[0]
        
    elif field == 'address':
        surl = url.split(',')
        parts = surl[0].split(':')
        return parts[1][2:]
               
    elif field == 'port':
        surl = url.split(',')
        parts = surl[0].split(':')
        return int(parts[2])
        
    elif field == 'proxyport':
        surl = url.split(',')
        if len(surl) > 1:
            pport = int(surl[1])
            return pport
        else:
            parts = surl[0].split(':')
            return int(parts[2])
            
    elif field == 'proxyurl':
        surl = url.split(',')
        if len(surl) == 1:
            return url
        else:
            pport = surl[1]
            parts = surl[0].split(':')
            parts[2] = pport
            return ':'.join(parts)

def getRefCounts():
    d = {}
    sys.modules
    # collect all classes
    for m in sys.modules.values():
        for sym in dir(m):
            o = getattr (m, sym)
            if type(o) is types.ClassType:
                d[o] = sys.getrefcount (o)
    # sort by refcount
    pairs = map (lambda x: (x[1],x[0]), d.items())
    pairs.sort()
    # pairs.reverse()
    return pairs
    
def printRefCounts():
    cnts = getRefCounts()
    for e in cnts:
        print e
    
        
# Tests
def tstStateMachine():
    import time
    
    class obj:
        def __init__(self):
            self.state = StateMachine()
            self.state.AddEvent((True,set(['on']),self.evOn))
            self.state.AddEvent((True,set(['newon']),self.evOn))
            self.state.AddEvent((False,set(['on']),self.evOff))
            
        def evOn(self):
            print 'on'
            
        def evOff(self):
            print 'off'
            
        
    while True:
        o = obj()
        o.state.Include(['on','newon'])
        # if o.state.HasSignal('on'):
            # print 'on came on'
        time.sleep(1)
        o.state.Exclude(['on','newon'])
        time.sleep(1)
        printRefCounts()
                

        
if __name__=='__main__':
    tstStateMachine()
    