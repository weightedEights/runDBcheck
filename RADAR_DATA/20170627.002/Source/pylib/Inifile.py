#-----------------------------------------------------------------------------
# Name:        Inifile.py
# Purpose:     
#
# Author:      John Jorgensen
#
# Created:     2005/28/06
# RCS-ID:      $Id: Inifile.py $
# Copyright:   (c) 2005 SRI International
# Version:     1.0.0.0
#-----------------------------------------------------------------------------

import ConfigParser as cp
import re
from sets import Set


class CommentPreservingInifile:
    def __init__(self,filename):
        self.Filename = filename
        self.Content = []
        
        self.Read(filename)
        
    def Read(self,filename):
        self.Filename = filename
        
        # Read contents of file and split it into a list of lines
        try:
            f = open(filename,'r')
            self.Content = f.read().split('\n')
            f.close()
        except:
            pass
            
    def Write(self,filename=None):
        if filename is None:
            filename = self.Filename
            
        f = open(filename,'w')
        for l in self.Content:
            f.write('%s\n' % l)
        f.close()
        
    def addsection(self,section):
        if len(self.Content) > 0:
            self.Content.append('')
            self.Content.append('[%s]' % (section))
        else:
            self.Content.append('[%s]' % (section))
        
    def get(self,section,option,default=None):
        try:
            v = self.GetOption(section,option)
            v = self.__expand__(v)
            return v
        except:
            if default is None:
                raise Exception('Option: %s not in section: %s' % (option,section))
            else:
                return default
        
    def getint(self,section,option,default=None):
        v = self.get(section,option,default)
        try:
            n = int(v)
        except:
            raise Exception('Option: %s with value: %s is not an integer' % (option,v))
        return n
        
    def getfloat(self,section,option,default=None):
        v = self.get(section,option,default)
        try:
            n = float(v)
        except:
            raise Exception('Option: %s with value: %s is not a float' % (option,v))
        return n
        
    def getboolean(self,section,option,default=None):
        v = self.get(section,option,default)
        if isinstance(v,str):
            f = not ((v == '0') or (v.lower() == 'false') or (v.lower() == 'disable') or (v.lower() == 'off'))
        elif isinstance(v,bool):
            f = v
        return f
        
    def set(self,section,option,value):
        self.SetOption(section,option,str(value))
        
    def has_section(self,section):
        try:
            sec = self.SectionIndex(section)
            return True
        except:
            return False
            
    def options(self,section):
        return self.GetOptions(section)
        
    def items(self,section):
        return self.GetItems(section)
        
    def section(self,section):
        items = self.GetItems(section)
        res = {}
        for vn,vv in items:
            res[vn] = vv
        return res
        
    def vars(self,section):
        vars = self.section(section)
        return self.__expandAll__(vars)
                
    def GetOption(self,section,option):
        sec = self.SectionIndex(section)
        n = sec+1
        while n < len(self.Content):
            l = self.Content[n]
            # ignore ; comments
            try:
                i = l.index(';')
                l = l[:i].strip()
            except:
                pass
            # ignore # comments    
            try:
                i = l.index('#')
                l = l[:i].strip()
            except:
                pass
            # Check to see if we reached new section    
            if '[' in l:
                break
            try:
                na,va = l.split('=')
                if na.lower().strip() == option.lower():
                    return va.strip()
            except:
                pass
            # next line
            n += 1
        raise Exception('Option: %s not found in section: %s' % (option,section))
        
    def SetOption(self,section,option,value):
        if not self.has_section(section):
            self.addsection(section)
                        
        sec = self.SectionIndex(section)
        
        n = sec+1
        if n >= len(self.Content):
            self.Content.append('%s=%s' % (option,value))
            return
            
        while n < len(self.Content):
            l = self.Content[n]
            c = ''
            # Collect ; comments
            try:
                i = l.index(';')
                c = l[i:]
                l = l[:i].strip()
            except:
                pass
            # Collect # comments    
            try:
                i = l.index('#')
                c = l[i:]
                l = l[:i].strip()
            except:
                pass
            # Check to see if we reached new section    
            if ('[' in l):
                # option not there insert it!
                bn = n-1
                while bn >= 0:
                    bl = self.Content[bn]
                    if bl <> '':
                        self.Content.insert(bn+1,'%s=%s' % (option,value))
                        return
                    bn -= 1
                    
            # Check to see if we hit end of content
            if (n == len(self.Content)-1):
                self.Content.append('%s=%s' % (option,value))
                return
            try:
                na,va = l.split('=')
                if na.lower().strip() == option.lower():
                    if c <> '':
                        self.Content[n] = '%s=%s  %s' % (na.strip(),value,c)
                    else:
                        self.Content[n] = '%s=%s' % (na,value)
                    return
            except:
                pass
            # next line
            n += 1
    
    def GetOptions(self,section):
        sec = self.SectionIndex(section)
        res = []
        n = sec+1
        while n < len(self.Content):
            l = self.Content[n]
            # ignore ; comments
            try:
                i = l.index(';')
                l = l[:i].strip()
            except:
                pass
            # ignore # comments    
            try:
                i = l.index('#')
                l = l[:i].strip()
            except:
                pass
            # Check to see if we reached new section    
            if '[' in l:
                break
            try:
                na,va = l.split('=')
                res.append(na.strip())
            except:
                na = l.strip()
                if na <> '':
                    res.append(l.strip())
            # next line
            n += 1
        return res
        
    def GetItems(self,section):
        sec = self.SectionIndex(section)
        res = []
        n = sec+1
        while n < len(self.Content):
            l = self.Content[n]
            # ignore ; comments
            try:
                i = l.index(';')
                l = l[:i].strip()
            except:
                pass
            # ignore # comments    
            try:
                i = l.index('#')
                l = l[:i].strip()
            except:
                pass
            # Check to see if we reached new section    
            if '[' in l:
                break
            try:
                na,va = l.split('=')
                na = na.strip()
                va = va.strip()
                res.append((na,va))
            except:
                pass
            # next line
            n += 1
        return res
                
    def SectionIndex(self,section):
        sec = '[%s]' % (section.lower())
        for n,l in enumerate(self.Content):
            if l.lower() == sec:
                return n
        raise Exception('Section: %s not in file' % (section))
        
    def __expandAll__(self,vars):
        res = {}
        for key,value in vars.items():
            res[key] = self.__expand__(value)
        return res
    
    def __expand__(self,value):
        def expanded(v):
            return (v.find('{') == -1)
        
        try:
            es = self.section('expand').copy()
            es[''] = value
        except:
            return value # no expand section return value it most be fully expanded
            
        while not expanded(es['']):
            for n,v in es.items():
                n = n.lower()
                if v.find('{') == -1: # Fully expanded string
                    for sn,sv in es.items():
                        ln = sn.lower()
                        lv = sv.lower()
                        try:
                            indx = lv.index(n)
                            var = sv[indx:indx+len(n)]
                            es[sn] = es[sn].replace(var,v)
                        except:
                            continue
        return es[''] 

    def getSections(self):
        return [l[1:-1] for l in self.Content if re.match('\[.*\]',l) is not None]
            
        
Inifile = CommentPreservingInifile
 
class CascadingInifile(Inifile):
    def __init__(self,filename,includesection='include'):
        self.includeSection = includesection
        
        Inifile.__init__(self,filename)
        
        self.root = Inifile('')
        
        self.include(self);
        self.loadValues(self);
        self.Content = self.root.Content
        
    def include(self,inifile):
        if inifile.has_section(self.includeSection):
            for o in inifile.options(self.includeSection):
                try:
                    fn = inifile.get(self.includeSection,o,'')
                    ini = Inifile(fn)
                    self.include(ini)
                    self.loadValues(ini)
                except:
                    pass
        
    def loadValues(self,inifile):
        secs = inifile.getSections()
        for s in secs:
            for n,v in inifile.items(s):
                self.root.set(s,n,v)
                
    def loadFile(self,filename):
        ini = Inifile(filename)
        self.include(ini)
        self.loadValues(ini)
            
        
 

        
        
        
if __name__ == '__main__':
    ini = Inifile('c:/work/projects/radac/pydaq/pydaqcontrol/pydaq.ini')
    print ini.vars('environment')
    # ini = CascadingInifile('file1.ini')
    # ini.loadFile('file5.ini')
    # for l in ini.Content:
        # print l
    
    # val = ini.get('section 5','var1','empty')
    # val1 = ini.get('section 6','file','none')
    # print 's5v1: ',val
    # print 's6file: ',val1
    
    
    
    
    
    
    
    
    
        
