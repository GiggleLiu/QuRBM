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

    @abstractmethod
    def ind_apply(self,ind,ig):
        '''
        Apply group operation on index, notice that changing config is equivalent to changing indices.

        Parameters:
            :ind: ndarray, the indices.
            :ig: int, group index.

        Return:
            ndarray, new indices.
        '''
        pass

class NoGroup(RBMGroup):
    def __init__(self):
        super(NoGroup,self).__init__(1)

    def __str__(self):
        return 'NoGroup'

    def apply(self,config,ig):
        return config

    def ind_apply(self,ind,ig):
        return ind

class TIGroup(RBMGroup):
    '''
    Group of Translational invariance.
    '''
    def __str__(self):
        return 'Translaion Group (%s)'%self.ng

    def apply(self,config,ig):
        if ig>=self.ng or ig<=-self.ng:
            raise ValueError()
        return roll(config,ig,axis=-1)

    def ind_apply(self,ind,ig):
        return ind-ig

