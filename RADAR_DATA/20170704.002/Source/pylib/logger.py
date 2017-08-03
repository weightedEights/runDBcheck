"""
System logger class

"""

from logging import Logger
import os,sys

class logger(Logger):
    
    def __init__(self,name):
        Logger.__init__(self,name)
        
        self.computer = os.environ['COMPUTERNAME']
        name = os.path.basename(sys.argv[0]).split('.')[0]
        if name.find('-') <> -1:
            app = name.split('-')[1]
        else:
            app = name
        self.application = app
        
    def makeRecord(self,name, lvl, fn, lno, msg, args, exc_info , func=None, extra=None):
        add = {}
        add['server'] = self.computer
        add['application'] = self.application
        if extra is not None:
            add.update(extra)
        rec = Logger.makeRecord(self,name, lvl, fn, lno, msg, args, exc_info , func=func, extra=add)
        return rec
        