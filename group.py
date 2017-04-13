'''
Transformation groups.
'''

from abc import ABCMeta, abstractmethod
from numpy import *
import pdb

__all__=['RBMGroup','NoGroup','TIGroup']

class RBMGroup(object):
    '''Abstract Group class for Restricted Boltzmann Machine.'''
    __metaclass__ = ABCMeta

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
    def __str__(self):
        return 'NoGroup'

    def apply(self,config,ig):
        return config

    def ind_apply(self,ind,ig):
        return ind

    @property
    def ng(self):
        return 1

class TIGroup(RBMGroup):
    '''
    Group of Translational invariance.
    '''
    def __init__(self,ngs):
        self.ngs=asarray(ngs) if hasattr(ngs,'__iter__') else array([ngs])
        #self.NGS=cumprod(ngs[::-1])[::-1]
        self.NGS=append(cumprod(self.ngs[::-1])[::-1][1:],[1])

    def __str__(self):
        return 'Translaion Group (%s)'%self.ngs

    def _expand_ind(self,ig):
        ig=asarray(ig)
        res=(abs(ig)[...,newaxis]/self.NGS*sign(ig)[...,newaxis])%self.ngs
        return res

    def _pack_ind(self,ig):
        res=sum(self.NGS*ig,axis=-1)
        return res

    @property
    def ng(self):
        return prod(self.ngs)

    def apply(self,config,ig):
        config=asarray(config)
        if ig>=self.ng or ig<=-self.ng:
            raise ValueError()
        igs=self._expand_ind(ig)
        if len(self.ngs)>1:
            config=config.reshape(config.shape[:-1]+tuple(self.ngs))
        gdim,cdim=len(self.ngs),ndim(config)
        for i,igi in enumerate(igs):
            config=roll(config,igi,axis=cdim-gdim+i)
        return config.ravel()

    def _iterate_apply(self,config,iaxis):
        gdim=len(self.ngs)
        if iaxis==len(self.ngs)-1:
            cdim=ndim(config)
            raxis=cdim-gdim+iaxis
            return array([roll(config,ig,axis=raxis).reshape(config.shape[:-gdim]+(-1,)) for ig in xrange(self.ngs[iaxis])])
        else:
            return concatenate([self._iterate_apply(roll(config,ig,axis=iaxis),iaxis+1) for ig in xrange(self.ngs[iaxis])],axis=0)

    def apply_all(self,config):
        if len(self.ngs)>1:
            config=config.reshape(config.shape[:-1]+tuple(self.ngs))
        return self._iterate_apply(config,0)

    def ind_apply(self,ind,ig):
        res=self._pack_ind((self._expand_ind(ind)-self._expand_ind(ig))%self.ngs)
        return res
