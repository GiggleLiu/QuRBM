from numpy import *
from abc import ABCMeta, abstractmethod

__all__=['Optimizer','RMSProp','DefaultOpt']


class Optimizer(object):
    '''
    Compute the change of paramter according to the gradient.
    '''
    __metaclass__ = ABCMeta

    @abstractmethod
    def __call__(self,key,g):
        '''
        Parameters:
            :key: object, key for indexing.
            :g: ndarray, the gradient.

        Return:
            ndarray, the change of data.
        '''
        pass

class DefaultOpt(Optimizer):
    '''Default optimizer.'''
    def __init__(self,rate):
        self.rate=rate

    def __call__(self,key,g):
        return -g*self.rate

class RMSProp(Optimizer):
    '''
    RMSProp adaptive optimizer.

    Parameters:
        :rate: float, learning rate.
        :delta: float, constant to assure numerical stability.
        :rho: float, decay rate.
        :r: array, the learning rate.
    '''
    def __init__(self,rate,rho,delta=1e-6):
        self.rate=rate
        self.rho=rho
        self.delta=delta
        self.rs={}

    def __call__(self,key,g): #g is the gradient
        if self.rs.has_key(key):
            r=self.rs[key]
        else:
            r=0
        #r=r*self.rho+(1-self.rho)*g**2
        r=r*self.rho+(1-self.rho)*abs(g)**2
        self.rs[key]=r
        return -self.rate/sqrt(r+self.delta)*g


