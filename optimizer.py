from numpy import *
from abc import ABCMeta, abstractmethod

__all__=['Optimizer','RMSProp','DefaultOpt','MannKendall']


class Optimizer(object):
    '''
    Compute the change of paramter according to the gradient.
    '''
    __metaclass__ = ABCMeta

    @abstractmethod
    def __call__(self,f,g,p):
        '''
        Parameters:
            :f: float, the lost function.
            :g: ndarray, the gradient.
            :p: int, # of iteration.

        Return:
            ndarray, the change of data.
        '''
        pass

class DefaultOpt(Optimizer):
    '''Default optimizer.'''
    def __init__(self,rate):
        self.rate=rate

    def __call__(self,f,g,p=None):
        return -g*self.rate

class RMSProp(Optimizer):
    '''
    RMSProp adaptive optimizer.

    Parameters:
        :rate: float, learning rate.
        :delta: float, constant to assure numerical stability.
        :rho: float, decay rate.
        :r: ndarray, the memory.
    '''
    def __init__(self,rate,rho,delta=1e-6):
        self.rate=rate
        self.rho=rho
        self.delta=delta
        self.r=0

    def __call__(self,f,g,p=None):
        self.r=self.r*self.rho+(1-self.rho)*abs(g)**2
        return -self.rate/sqrt(self.r+self.delta)*g

class MannKendall(Optimizer):
    '''Mann-Kendall trend test.'''
    def __init__(self,rate,size):
        self.rate=rate
        self.size=size
        self.fl=[]
        self.zeta=0.

    def __call__(self,f,g,p=None):
        self.zeta+=sign(f-array(self.fl)).sum()
        self.fl.append(f)
        if len(self.fl)>self.size:
            #remove one from the begining.
            f0=self.fl.pop(0)
            self.zeta-=sign(array(self.fl)-f0).sum()
        N=len(self.fl)
        trend=2*abs(self.zeta)/N/(N-1) if N>1 else 1
        print 'Trend = %s'%trend
        return -trend*self.rate*g
