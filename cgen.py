'''Random Boltzmann Machine Kernel for Monte Carlo.'''

from numpy import *
from profilehooks import profile
from scipy.linalg import norm
from abc import ABCMeta, abstractmethod
import pdb,time

from utils import logcosh

__all__=['RBMConfigGenerator','ConfigGenerator']

class ConfigGenerator(object):
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

class RBMConfigGenerator(ConfigGenerator):
    '''
    Monte Carlo sampling core for Restricted Boltzmann Machine state.

    Attributes:
        :state: <RBM>,
        :runtime: dict, runtime variables.
    '''
    counter=0
    def __init__(self,nflip,initial_config=None):
        self.nflip=nflip
        self.state=None
        self.theta=None
        self._theta=None
        if hasattr(initial_config,'__iter__'):
            initial_config=asarray(initial_config)
        self.config=initial_config

    def set_state(self,state):
        self.state=state
        if self.config is None:
            self.config=self.random_config()
        self.theta=state.feed_input(self.config)   # bug fix note: remember these two lines are needed!
        self._theta=None

    def random_config(self):
        rbm=self.state
        config=1-2*random.randint(0,2,rbm.nin)
        return config

    def pop(self,flips):
        '''
        Probability ratio between fliped config and old config.

        Parameters:
            :flips: 1darray, positions to flip.

        Return:
            tuple, (new configuration, new theta table, <c'|Psi>/<c|Psi>).
        '''
        _theta=copy(self.theta)
        rbm=self.state
        nj=rbm.W.shape[1]
        nsite=rbm.nin

        t0=time.time()
        #update new theta table
        for ig in xrange(rbm.group.ng):
            for iflip in flips:
                _theta[ig*nj:(ig+1)*nj]+=-2*self.config[iflip]*rbm.W[rbm.group.ind_apply(iflip,-ig)%nsite]
        t1=time.time()

        #pratio=abs(exp(2*(nc[iflip]*rbm.a[iflip]).sum())*prod(cosh(_theta)/cosh(self.theta)))**2
        pratio=exp(2*(-self.config[flips]*rbm.a[flips]).sum()+sum(logcosh(_theta)-logcosh(self.theta)))
        #pratio_=rbm.get_weight(nc)/rbm.get_weight(asarray(self.config))

        #if self.counter%10000==0:
        #    t1d=time.time()
        #    true_theta=rbm.feed_input(nc)
        #    t2=time.time()
        #    print 'Time: %s, %s. Err: %s'%(t1-t0,t2-t1d,norm(true_theta-_theta))
        #print pratio/pratio_
        t2=time.time()
        #print 'Time: %s, %s.'%(t1-t0,t2-t1)
        return _theta,pratio

    def fire(self):
        '''Fire a proposal.'''
        if self.state is None: raise AttributeError("Please set the state before sampling!")
        rbm=self.state
        nsite=rbm.nin
        config=self.config
        
        #generate a new config by flipping n spin
        if self.nflip==2:
            upmask=config==1
            flips=random.randint(0,nsite/2,2)
            iflip0=where(upmask)[0][flips[0]]
            iflip1=where(~upmask)[0][flips[1]]
            flips=array([iflip0,iflip1])
        else:
            iflip0=random.randint(nsite)
            flips=array([iflip0])

        #transfer probability is equal, pratio is equal to the probability ratio
        self._theta,pop=self.pop(flips)
        self.counter+=1
        return flips,norm(pop)**2

    def reject(self,*args,**kwargs):
        #self._theta=None
        pass

    def confirm(self,flips,*args,**kwargs):
        self.theta=self._theta
        self.config[flips]*=-1
        #self._theta=None
