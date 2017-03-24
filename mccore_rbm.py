'''Random Boltzmann Machine Kernel for Monte Carlo.'''

from numpy import *
import pdb

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

    def random_config(self):
        rbm=self.state
        config=2*random.randint(0,2,rbm.nin)-1
        return config

    def fire(self,config):
        '''Fire a proposal.'''
        rbm=self.state
        #generate a new config by flipping a spin
        nc=copy(config)  #why random flip, how about system with good quantum number?
        iflip=random.randint(rbm.nin)
        nc[iflip]*=-1 #1-iflip
        #transfer probability is equal, pratio is equal to the probability ratio
        if self.theta is None: self.theta=rbm._pack_input(config).dot(rbm.S[:,1:])
        self._theta=self.theta+2*nc[iflip]*rbm.S[iflip+1,1:] 
        pratio=abs(rbm.get_weight(nc,theta=self._theta)/rbm.get_weight(config,theta=self.theta))**2
        return nc,pratio

    def reject(self,*args,**kwargs):
        pass

    def confirm(self,*args,**kwargs):
        self.theta=self._theta

    def get_runtime(self):
        return {'theta':self.theta}