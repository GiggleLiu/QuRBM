'''Random Boltzmann Machine Kernel for Monte Carlo.'''

from numpy import *
from scipy.linalg import norm
from abc import ABCMeta, abstractmethod
import pdb,time

from utils import logcosh
from clib.cutils import pop1D,pop2D

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
        rbm=self.state
        if rbm.group.ng==1 or len(rbm.group.ngs)==1:
            return pop1D(config=self.config,flips=asarray(flips),W=rbm.W,a=rbm.a,theta=self.theta,ng=rbm.group.ng)
        else:
            return pop2D(config=self.config,flips=asarray(flips),W=rbm.W,a=rbm.a,theta=self.theta,ngs=rbm.group.ngs)

        _theta=copy(self.theta)
        nj=rbm.W.shape[1]
        nsite=rbm.nin
        flips=asarray(flips)

        t0=time.time()
        #update new theta table
        cflip=self.config[flips]
        for ig in xrange(rbm.group.ng):
            for i,iflip in enumerate(flips):
                _theta[ig*nj:(ig+1)*nj]-=2*cflip[i]*rbm.W[rbm.group.ind_apply(iflip,-ig)%nsite]  #-ig is corrent!
        t1=time.time()

        pratio=exp(2*sum([-cflip*rbm.a[rbm.group.ind_apply(flips,-ig)%nsite] for ig in xrange(rbm.group.ng)],axis=0).sum()+sum(logcosh(_theta)-logcosh(self.theta)))
        nc=copy(self.config); nc[flips]*=-1
        _theta_true=rbm.feed_input(nc)
        pratio_true=rbm.get_weight(nc)/rbm.get_weight(asarray(self.config))
        print _theta_true-_theta0,_theta_true-_theta
        print pratio_true-pratio0,pratio_true-pratio
        pdb.set_trace()
        return _theta,pratio

    def fire(self):
        '''Fire a proposal.'''
        nsite=self.state.nin
        
        #generate a new config by flipping n spin
        if self.nflip==2:
            #flips=random.randint(0,nsite,2)       #why this code is wrong?
            #while flips[0]==flips[1]:
                #flips=random.randint(0,nsite,2)
            upmask=self.config==1
            flips=random.randint(0,nsite/2,2)
            iflip0=where(upmask)[0][flips[0]]
            iflip1=where(~upmask)[0][flips[1]]
            flips=array([iflip0,iflip1])
        else:
            iflip0=random.randint(nsite)
            flips=array([iflip0])

        #transfer probability is equal, pratio is equal to the probability ratio
        self._theta,pop1=self.pop(flips=flips)
        return flips,abs(pop1)**2

    def reject(self,*args,**kwargs):
        #self._theta=None
        pass

    def confirm(self,flips,*args,**kwargs):
        self.theta=self._theta
        self.config[flips]*=-1
        #self._theta=None
