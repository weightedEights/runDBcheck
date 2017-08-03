#-----------------------------------------------------------------------------
# Name:          TuFile.py
# Purpose:      This class is used to parse a timing unit file
#
# Author:       John Jorgensen
#
# Created:     2005/01/06
# RCS-ID:     $Id: TuFile.py $
# Copyright:  SRI International
#-----------------------------------------------------------------------------

"""
This class is used to parse a timing unit file for easy 
access of various information like duty cycle frametime etc.
"""

import os.path as path
import logging

def tlsort(e1,e2):
    if e1[1] > e2[1]:
        return 1
    elif e1[1] < e2[1]:
        return -1
    else:
        if e1[2] > e2[2]:
            return 1
        elif e1[1] < e2[2]:
            return -1
        else:
            return 0

class TuFile:
    def __init__(self,filename=None,mode='r'):
        self.log = logging.getLogger('TuFile')
        self.Filename = filename
       
        self.Entries = []
        self.ContTimeEntries = []
        
        if mode == 'r':
            self.Load(filename,mode)
            self.Parse()
            self.image = self.Compile()
            
    def FrameTime(self):
        ipps = self.Extract(bit=[-1,])
        frametime = 0
        for ipp in ipps:
            bit,start,dura,swa,sw,rem = ipp
            frametime += dura
        return frametime
        
    def Duty(self,bit):
        entries = self.Extract(bit)
        sum = 0
        for b,s,d,sa,s,rem in entries:
            sum = sum + d
        return float(sum)/self.FrameTime()
        
    def Check(self,t,val):
        return (((t[0] in val[0]) or (val[0] == [])) and \
                ((t[1] in val[1]) or (val[1] == [])) and \
                ((t[2] in val[2]) or (val[2] == [])) and \
                ((t[3] in val[3]) or (val[3] == [])) and \
                ((t[4] in val[4]) or (val[4] == [])))
        
    def Extract(self,bit=[],start=[],duration=[],specialwordaddress=[],specialword=[]):
        if not isinstance(bit,list):
            bit = [bit,]
        if not isinstance(start,list):
            start = [start,]
        if not isinstance(duration,list):
            duration = [duration,]
        if not isinstance(specialwordaddress,list):
            specialwordaddress = [specialwordaddress,]
        if not isinstance(specialword,list):
            specialword = [specialword,]
        return [t for t in self.Entries if self.Check(t,(bit,start,duration,specialwordaddress,specialword))]            
        
    def Index(self,bit=[],start=[],duration=[],specialwordaddress=[],specialword=[]):
        if not isinstance(bit,list):
            bit = [bit,]
        if not isinstance(start,list):
            start = [start,]
        if not isinstance(duration,list):
            duration = [duration,]
        if not isinstance(specialwordaddress,list):
            specialwordaddress = [specialwordaddress,]
        if not isinstance(specialword,list):
            specialword = [specialword,]
        return [n for n,t in enumerate(self.Entries) if self.Check(t,(bit,start,duration,specialwordaddress,specialword))]            
    
    def Delete(self,n):
        if not isinstance(n,list):
            n = [n,]
        n.reverse()
        for i in n:
            del(self.Entries[i])
    
    def Insert(self,n,entry,above=True):
        if not isinstance(n,list):
            n = [n,]
        n.reverse()
        for i in n:
            if above:
                self.Entries.insert(i,entry)
            else:
                self.Entries.insert(i+1,entry)
    
    def Load(self,filename=None,mode='r'):
        if self.Filename is None:
            raise "TuFile: No filename specified"
        f = open(filename,mode)
        try:
            self.Raw = f.readlines()
            self.Parse()
        except:
            pass
            
    def parseLine(self,line):
        res = []
        s = line.split()
        if len(s) == 0:
            return []
        for e,f in enumerate(s):
            try:
                if f.lower().find('0x') >= 0:
                    n = int(f.strip(),16)
                else:
                    n = int(f.strip())
                res.append(n)
                if len(res) == 5:
                    comment = ' '.join(s[e+1:]).strip()
                    res.append(comment)
                    break
            except:
                if f <> '':
                    print 'Could not convert %s to number' % (f)
                continue
        return res
                           
        
    def Parse(self):
        self.Entries = []
        self.ContTimeEntries = []
        ipp = 0
        time = 0
        if not hasattr(self,'Raw'):
            return
        for n,line in enumerate(self.Raw):
            try:
                if line[0] == '*':   # Throw away comments
                    continue
                bit,start,dura,swa,sw,rem = self.parseLine(line.strip())
                self.Entries.append([bit,start,dura,swa,sw,rem])
            except Exception,value:
                self.log.error(value)
                
            if bit == -1:
                time = time+ipp
                ipp = dura
            start = time + start 
            self.ContTimeEntries.append([bit,start,dura,swa,sw,rem])
            
        
    def ParseEntries(self):
        ipp = 0
        time = 0
        self.ContTimeEntries = []
        for e in self.Entries:
            bit,start,dura,swa,sw,rem = e
            if bit >= 0:
                start += time
            if bit == -1:
                time += ipp
                ipp = dura
                
            self.ContTimeEntries.append([bit,start,dura,swa,sw,rem])
            

    def Save(self,filename=None):
        if filename is None:
            if self.Filename is not None:
                fn = self.Filename
            else:
                raise 'Save: No filename specified'
        else:
            fn = filename
        f = open(fn,'w')
        p,n = path.split(fn)
        f.write('*\n')
        f.write('* File: %s\n' % (n))
        f.write('*\n')
        ipp = 0
        for e in self.Entries:
            if e[0] == -1:
                f.write('%-8s%-8s%-8s%-8s%-8s%-8s\n' % ('*Bit','Start','Dura','SwA','Sw','IPP: %d' % (ipp)))
                ipp += 1
            f.write('%-8d%-8d%-8d%-8d%-8d%-8s\n' % (e[0],e[1],e[2],e[3],e[4],e[5]))
            if e[0] == -9:
                f.write('*\n')
        f.close()
        
        
    def Compile(self):
        if self.ContTimeEntries == []:
            self.ParseEntries()
            
        # Convert special word -2 flag to special word bit
        l = [e for e in self.ContTimeEntries if e[0] == -2]
        for e in l:
            e[0] = 30
            e[2] = 1
            
        # Build transition list
        tl = []
        l = [e for e in self.ContTimeEntries if e[0] >= 0]
        for e in l:
            tl.append([e[0],e[1],True,e[3],e[4]])
            tl.append([e[0],e[1]+e[2],False,0,0])
        tl.sort(tlsort)
        tl.append([31,self.FrameTime()-1,True,0,0])
        tl.append([31,self.FrameTime(),False,0,0])
        
        buf = []
        o = 0
        t = 0
        sw = None
        for e in tl:
            d = e[1]-t
            t = e[1]
            if d > 0:
                buf.append(d)
                buf.append(o)
                if (o & 0x40000000) == 0x40000000:
                    buf.append(sw)
            if e[2]:
                o = o | int(pow(2,e[0]))
            else:
                o = o & ~ int(pow(2,e[0]))
                
            if e[0] == 30 and e[2]:
                sw = e[3]*0x10000+e[4]
        return buf
     
        
        
        
if __name__ == '__main__':
    logging.basicConfig()
    
    f = TuFile(r'\\dtc0\radac\pydaq\setup\poker\faraday01\dtc0.tuf')
    b = f.Compile()
    hb = [hex(v) for v in b]
    sb = '\n'.join(hb)
    fo = open(r'c:\tmp\newcompile.tuf','w')
    fo.write(sb)
    fo.close()
    
    
            
    
            
    
        
    
    
        
        
        
            
            
        
        
        
