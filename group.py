'''
Transformation groups.
'''

from abc import ABCMeta, abstractmethod
from numpy import roll

__all__=['RBMGroup','NoGroup','TIGroup']

class RBMGroup(object):
    '''Abstract Group class for Restricted Boltzmann Machine.'''
    __metaclass__ = ABCMeta

    def __init__(self,ng):
        self.ng=ng
    
    @abstractmethod
    def apply(self,config,ig):
        '''
        Apply group operation on config.

        Parameters:
            :config: 1darray, the configuration.
            :ig: int, group index.

        Return:
            1darray, new config.
        '''
        pass

class NoGroup(RBMGroup):
    def __init__(self):
        super(NoGroup,self).__init__(1)

    def apply(self,config,ig):
        return config

class TIGroup(RBMGroup):
    '''
    Group of Translational invariance.
    '''

    def apply(self,config,ig):
        if ig>=self.ng or ig<=-self.ng:
            raise ValueError()
        return roll(config,ig,axis=-1)
