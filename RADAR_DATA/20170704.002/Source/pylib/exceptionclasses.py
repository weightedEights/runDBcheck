"""

Various exeption classes

"""

# Base exception class

class eBase(Exception):
    """
    Base class for error exceptions
    """
    
    name = 'eBase'
    
    def __init__(self,message):
        self.message = message
        
    def __str__(self):
        return '%s: %s' % (self.name,self.message)

# Error exceptions
class eError(eBase):
    name = 'eError'
        
# Info exceptions    
class eInfo(eBase):
    name = 'eInfo'

# Signal exceptions
class eSignal(eBase):
    name = 'eSignal'
    
    
# Specific exceptions
class integratorInfo(eInfo):
    name = 'integratorInfo'
    

class systemCheckError(eError):
    name = 'systemCheckError'
    
class terminate(eSignal):
    name = 'terminate'


        
        
    
    


if __name__ == '__main__':
    try:
        raise integratorInfo('missed pulses: 34')
    except integratorInfo,inst:
        print inst
    try:
        raise systemCheckError('No tu file loaded')
    except Exception,inst:
        print inst
    try:
        raise terminate('shutting down')
    except terminate,inst:
        print inst
        
    
        
