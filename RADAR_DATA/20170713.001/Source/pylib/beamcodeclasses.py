"""
This class deals with beamcode configuration info

"""

import logging

class beamcodesConfig:
    def __init__(self,filename):
        self.log = logging.getLogger('beamcodesConfig')
        self.filename = filename
        
        file = open(filename,'r')
        lines = file.readlines()
        self.codes = []
        for line in lines:
            try:
                c = int(line,16)
                self.codes.append(c)
            except: ## skip comment lines
                continue
            
        # Find uniq codes and sort them
        self.sortedCodes = []
        for c in self.codes:
            if c not in self.sortedCodes:
                self.sortedCodes.append(c)
        self.sortedCodes.sort()
