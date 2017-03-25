'''Random Boltzmann Machine Kernel for Monte Carlo.'''

from numpy import *
import pdb,time

from vmc import MCCore

__all__=['RBMCore']

class RBMCore(MCCore):
    '''
    Monte Carlo sampling core for Restricted Boltzmann Machine state.

    Attributes:
        :state: <RBM>,
        :runtime: dict, runtime variables.
    '''
    def __init__(self):
        self.state=None
        self.theta=None
        self._theta=None

    def set_state(self,state):
        self.state=state
        self.theta=None   # bug fix note: remember these two lines are needed!
        self._theta=None

    def random_config(self):
        rbm=self.state
        config=1-2*random.randint(0,2,rbm.nin)
        return config

    def fire(self,config):
        '''Fire a proposal.'''
        rbm=self.state
        #generate a new config by flipping a spin
        nc=copy(config)  #why random flip, how about system with good quantum number?
        iflip=random.randint(rbm.nin)
        nc[iflip]*=-1
        #transfer probability is equal, pratio is equal to the probability ratio
        if self.theta is None: self.theta=rbm.feed_input(config)
        self._theta=self.theta+2*nc[iflip]*rbm.W[iflip] 
        pratio=abs(exp(2*nc[iflip]*rbm.a[iflip])*prod(cosh(self._theta)/cosh(self.theta)))**2
        #pratio_=abs(rbm.get_weight(nc,theta=self._theta)/rbm.get_weight(config,theta=self.theta))**2
        return nc,pratio

    def reject(self,*args,**kwargs):
        pass

    def confirm(self,*args,**kwargs):
        self.theta=self._theta

    def get_runtime(self):
        return {'theta':self.theta}
