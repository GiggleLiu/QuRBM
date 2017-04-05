'''Random Boltzmann Machine Kernel for Monte Carlo.'''

from numpy import *
from profilehooks import profile
from scipy.linalg import norm
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
    def __init__(self,nflip=1,initial_config=None):
        self.nflip=nflip
        self._initial_config=initial_config
        self.state=None
        self.theta=None
        self._theta=None

    @property
    def initial_config(self):
        return self._initial_config if self._initial_config is not None else self.random_config()

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
        nsite=rbm.nin
        #generate a new config by flipping a spin
        nc=copy(config)  #why random flip, how about system with good quantum number?
        iflip0=random.randint(nsite)
        if self.nflip==2:
            iflip1=random.randint(nsite-1)
            if iflip1>=iflip0: iflip1+=1
            iflip=array([iflip0,iflip1])
        else:
            iflip=array([iflip0])
        nc[iflip]*=-1
        #transfer probability is equal, pratio is equal to the probability ratio
        if self.theta is None: self.theta=rbm.feed_input(config)
        nj=rbm.W.shape[1]
        self._theta=copy(self.theta)
        for ig in xrange(rbm.group.ng):
            #self._theta[ig*nj:(ig+1)*nj]+=2*(nc[iflip,newaxis]*rbm.W[rbm.group.ind_apply(iflip,-ig)%nsite]).sum(axis=0)
            if self.nflip==2:
                self._theta[ig*nj:(ig+1)*nj]+=2*(nc[iflip0]*rbm.W[rbm.group.ind_apply(iflip0,-ig)%nsite]+nc[iflip1]*rbm.W[rbm.group.ind_apply(iflip1,-ig)%nsite])
            else:
                self._theta[ig*nj:(ig+1)*nj]+=2*(nc[iflip0]*rbm.W[rbm.group.ind_apply(iflip0,-ig)%nsite])
        pratio=abs(exp(2*(nc[iflip]*rbm.a[iflip]).sum())*prod(cosh(self._theta)/cosh(self.theta)))**2
        #pratio_=abs(rbm.get_weight(nc)/rbm.get_weight(asarray(config)))**2
        return nc,pratio

    def reject(self,*args,**kwargs):
        pass

    def confirm(self,*args,**kwargs):
        self.theta=self._theta

    def get_runtime(self):
        return {'theta':self.theta}
