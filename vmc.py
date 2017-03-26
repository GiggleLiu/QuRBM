'''
Variational Monte Carlo Kernel.
'''

from numpy import *
from abc import ABCMeta, abstractmethod
from profilehooks import profile
import pdb

from linop import c_sandwich,OpQueue

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
    def __init__(self,core,nbath,nsample,nmeasure,nstat=1000,sampling_method='metropolis'):
        self.nbath,self.nsample=nbath,nsample
        self.core=core
        self.sampling_method=sampling_method
        self.nstat=nstat
        self.nmeasure=nmeasure
        self._config_histo=[]

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

    def measure(self,op,state,tol=0,initial_config=None):
        '''
        Measure an operator.

        Parameters:
            :op: <LinOp>, a linear operator instance.
            :state: <RBM>/..., a state ansaz
            :tol: float, tolerence.
            :initial_config: None/1darray, the initial configuration used in sampling.

        Return:
            number,
        '''
        nmeasure,nstat=self.nmeasure,self.nstat
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
                config=new_config
                if i>=self.nbath:
                    o=c_sandwich(op,config,state,runtime=self.core.get_runtime())
                    if i%nmeasure==0:ol.append(o)      #correlation problem?
                self._config_histo.append(config)
            else:
                self.core.reject(); accept_table.append(0)
                if i>=self.nbath:
                    o=c_sandwich(op,config,state,runtime=self.core.get_runtime()) if o is None else o
                    if i%nmeasure==0:ol.append(o)      #correlation problem?
                self._config_histo.append(config)
            if i%nstat==nstat-1 and len(ol)>0:
                if not isinstance(op,OpQueue):
                    varo=abs(var(ol,axis=0).sum()/len(ol))
                else:
                    varo=array([abs(var(oi,axis=0).sum()/len(ol)) for oi in zip(*ol)])
                print '%-10s Accept rate: %.3f, Std Err: %.5f'%(i+1,mean(accept_table),sum(varo))
                accept_table=[]

                #accurate results obtained.
                if len(ol)>100 and all(varo<tol):
                    break

        if not isinstance(op,OpQueue):
            return mean(ol,axis=0)
        else:
            return [mean(oi,axis=0) for oi in zip(*ol)]
