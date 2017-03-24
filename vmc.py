'''
Variational Monte Carlo Kernel.
'''

from numpy import *
from abc import ABCMeta, abstractmethod
import pdb

from linop import c_sandwich

__all__=['VMC']

class MCCore(object):
    '''
    Interface of Monte Carlo Kernel.
    '''
    __metaclass__ = ABCMeta

    @abstractmethod
    def set_state(self,state):
        '''
        Set up the state for sampling purpose.

        Parameters:
            :state: <RBM>/..., a state representation.
        '''
        pass

    @abstractmethod
    def random_config(self):
        '''Produce a random config.'''
        pass

    @abstractmethod
    def fire(self,config,**kwargs):
        '''
        Get a possible proposal for the next move.

        Parameters:
            :config: 1darray/None, the old configuration, generate a random config if is None.

        Return:
            proposal,
        '''
        pass

    @abstractmethod
    def reject(self,proposal,*args,**kwargs):
        '''
        Reject the proposal.

        Parameters:
            proposal: object, the proposal.
        '''
        pass

    @abstractmethod
    def confirm(self,proposal,*args,**kwargs):
        '''
        Reject the proposal.

        Parameters:
            proposal: object, the proposal.
        '''
        pass

class VMC(object):
    '''
    Variational Monte Carlo Engine.
    '''
    def __init__(self,core,nbath,nsample,sampling_method='metropolis'):
        self.nbath,self.nsample=nbath,nsample
        self.core=core
        self.sampling_method=sampling_method

    def accept(self,pratio,method='metropolis'):
        '''
        Decide whether accept or reject a move.

        Parameters:
            :pratio: float, the ratio of (distribution probability/transfer probability) between two configurations.

        Return:
            bool, accept if True.
        '''
        if self.sampling_method=='auto':
            method='heat-bath' if self.status=='WARM_UP' else 'metropolis'
        if self.sampling_method=='metropolis':
            A=pratio
        elif self.sampling_method=='heat-bath':
            A=pratio/(1+pratio)
        return random.random()<A

    def measure(self,op,state,initial_config=None):
        '''
        Measure an operator.

        Parameters:
            :op: <LinOp>, a linear operator instance.
            :state: <RBM>/..., a state ansaz
            :initial_config: None/1darray, the initial configuration used in sampling.

        Return:
            number,
        '''
        nskip,nstat=4,100
        ol=[]  #local operator values
        o=None
        self.core.set_state(state)
        config=initial_config if initial_config is not None else self.core.random_config()
        accept_table=[]

        for i in xrange(self.nbath+self.nsample):
            #generate new config
            new_config,pratio=self.core.fire(config)
            if self.accept(pratio):
                self.core.confirm(); accept_table.append(1)
                if i>=self.nbath:
                    o=c_sandwich(op,config,state,runtime=self.core.get_runtime())
                    if i%nskip==0:ol.append(o)      #correlation problem?
                config=new_config
            else:
                self.core.reject(); accept_table.append(0)
                if i>=self.nbath:
                    o=c_sandwich(op,config,state,runtime=self.core.get_runtime()) if o is None else o
                    if i%nskip==0:ol.append(o)      #correlation problem?
            if i%nstat==nstat-1:
                print '%s Accept rate: %s'%(i+1,mean(accept_table))
                accept_table=[]
        return mean(ol,axis=0)

