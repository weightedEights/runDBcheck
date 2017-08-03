#! /usr/bin/env python

"""
Line based socket communication

History:

Date:           20051005
Version:        1.0.0.0
Author:         John Jorgensen
Description:    Initial implementation
"""

version = '1.0.0.0'

from socket import *
from time import sleep
from jjlib.Logging import Log
import thread
from jjlib.Utils.sysutils import LockableSet,LineBuffer

setdefaulttimeout(5.0)       
  
class ClientLineSocket(socket):

    def __init__(self,address=None):
        Log.Write('ClientLineSocket')
        self.Address = address
        socket.__init__(self,AF_INET,SOCK_STREAM)
        Log.Write('Parent socket created')
        
        self.txbuffer = LineBuffer()
        
        self.State = LockableSet()
        
        self.OnConnect = None
        self.OnDisconnect = None
        self.OnLineReceived = None
        
        thread.start_new_thread(self.Receive,())
        thread.start_new_thread(self.Transmit,())
        
    def Shutdown(self):
        Log.Write('ClientLineSocket: shutdown',1)
        self.State.Exclude('Connected')
        self.State.Include('Shutdown')
        self.shutdown(2)
        self.close()
        while self.State.HasSignal('Rx'):
            sleep(0)
        while self.State.HasSignal('Tx'):
            sleep(0)
            
        
    def Connect(self,address=None):
        if address is not None:
            self.Address = address
         
        try:
            Log.Write('Calling socket connect',1)
            self.connect(self.Address)
            self.State.Include('Connected')
            Log.Write('Connected to server',1)
            if self.OnConnect is not None:
                self.OnConnect()                
        except error,value:
            self.State.Exclude('Connected')
            Log.Write('Error: '+str(value),5)
            
    def Write(self,data):
        if isinstance(data,str):
            self.txbuffer.Put(data)
        elif isinstance(data,list):
            buf = self.txbuffer.Acquire()
            buf.extend(data)
            self.txbuffer.Release()
                                                
    def Receive(self):
        Log.Write('Receive thread started')
        partialline = ''
        self.State.Include('Rx')
        while not self.State.HasSignal('Shutdown'):
            sleep(0)
            try:
                s = self.State()
                self.State(s)
                if 'Connected' not in s:
                    continue
                buf = self.recv(1024)
                linecomplete = buf.endswith('\n')
                lines = buf.split('\n')
                if partialline != '':
                    lines[0] = partialline+lines[0]
                    partialline = ''
                if linecomplete:
                    if self.OnLineReceived is not None:
                        self.OnLineReceived(lines)
                else:
                    partialline = lines[-1]
                    if len(lines) > 1:
                        lines = lines[0:-1]
                        if self.OnLineReceived is not None:
                            self.OnLineReceived(lines)
            except error,value:
                Log.Write('Error: '+str(value),-1)
        self.State.Exclude('Rx')
        Log.Write('Receive shutting down')
                
    def Transmit(self):
        Log.Write('Transmit thread started')
        self.State.Include('Tx')
        while not self.State.HasSignal('Shutdown'):
            try:
                sleep(0)
                s = self.State()
                self.State(s)
                if 'Connected' not in s:
                    continue
                line = self.txbuffer.Get()
                if line is not None:
                    self.sendall(line+'\r\n')               
            except error,value:
                Log.Write('Error: '+str(value))
        self.State.Exclude('Tx')
        Log.Write('Transmit shutting down')
        

class SyncLineSocket:
    def __init__(self,address,timeout=10):
        self.Address = address
        self.Timeout = timeout
        
    def Connect(self):
        try:
            s = socket(AF_INET,SOCK_STREAM)
            s.settimeout(self.Timeout)
            s.connect(self.Address)
            return s
        except error,value:
            raise 'BlockingSocket connect failed: '+str(value)
            
    def Disconnect(self,socket):
        Log.Write('Disconnecting')
        socket.shutdown(2)
        socket.close()
        
    def Write(self,socket,data):
        try:
            socket.sendall(data+'\r\n')
        except:
            raise 'BlockingSocket Write failed'
            
    def Read(self,socket):
        try:
            line = ''
            while True:
                buf = socket.recv(8000)
                line += buf
                if line.endswith('\n'):
                    return line.strip()
        except:
            raise 'BlockingSocket Read failed'
            
    def Request(self,req):
        socket = self.Connect()
        command,value = req.split('=')
        self.Write(socket,req)
        try:
            while True:
                res = self.Read(socket)
                Log.Write('Received: '+res)
                if res.find(command) <> -1:
                    break
        finally:
            self.Disconnect(socket)
        return res.split('=')
        
if __name__ == '__main__':
    s = SynchLineSocket(('localhost',5003))
    
        
                
