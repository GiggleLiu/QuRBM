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

    def measure(self,op,state,tol=0):
        '''
        Measure an operator.

        Parameters:
            :op: <LinOp>, a linear operator instance.
            :state: <RBM>/..., a state ansaz
            :tol: float, tolerence.

        Return:
            number,
        '''
        nmeasure,nstat=self.nmeasure,self.nstat
        bins=[]
        ol=[]  #local operator values
        o=None
        self.core.set_state(state)
        config=self.core.initial_config
        n_accepted=0

        for i in xrange(self.nbath+self.nsample):
            #generate new config
            new_config,pratio=self.core.fire(config)
            if self.accept(pratio):
                self.core.confirm(); n_accepted+=1
                config=new_config
                o=None
            else:
                self.core.reject()
            if i>=self.nbath:
                if i%nmeasure==0:
                    o=c_sandwich(op,config,state,runtime=self.core.get_runtime()) if o is None else o
                    ol.append(o)      #correlation problem?
            if i%nstat==nstat-1:
                if len(ol)>0:
                    if not isinstance(op,OpQueue):
                        bins.append(mean(ol))
                        var_bin=var(bins,axis=0).mean()
                    else:
                        bins.append([mean(oi) for oi in ol])
                        var_bin=array([var(bi,axis=0).mean() for bi in zip(*bins)])
                    std_bin=sqrt(var_bin/len(bins))
                    print '%-10s Accept rate: %.3f, Std Err: %.5f'%(i+1,n_accepted*1./nstat,sqrt(sum(std_bin**2)))

                    #accurate results obtained.
                    if len(bins)>100 and all(std_bin<tol):
                        break
                    ol=[]  #local operator values
                else:
                    print '%-10s Accept rate: %.3f'%(i+1,n_accepted*1./nstat)
                n_accepted=0

        if not isinstance(op,OpQueue):
            return mean(ol,axis=0)
        else:
            return [mean(oi,axis=0) for oi in zip(*ol)]
