'''Several Toy Models.'''

from numpy import *
from numpy.linalg import norm
import pdb

from tba.hgen import SpinSpaceConfig,sx,sy,sz
from linop import *
from sstate import SparseState

__all__=['HeisenbergH','FakeVMC']

class HeisenbergH(LinOp):
    '''
    Heisenberg Hamiltonian

    H = J S_z*S_z' + J/2(S+S-' + S-S+')
    '''
    opt_lmul=False

    def __init__(self,nsite,J=1):
        self.J=J
        self.nsite=nsite

    def rmatmul(self,target):
        if isinstance(target,ndarray):  #series of {1,-1}
            ws,cs=[1],[target]
        else:
            ws,cs=target.w,target.configs
        nsite=len(cs[0])
        wl,configs=[],[]
        for w,c in zip(ws,cs):
            #J(SzSz) terms.
            wl.append(self.J/4.*w*(roll(c,-1)*c).sum(axis=-1))
            configs.append(c)

            for i in xrange(nsite):
                #J/2(S+S- + S-S+) terms
                j=(i+1)%nsite
                if c[i]^c[j]:
                    nc=copy(c)
                    nc[i]*=-1; nc[j]*=-1
                    wl.append(self.J/2.*w)
                    configs.append(nc)
        return SparseState(wl,configs)

class FakeVMC(object):
    '''The Fake VMC program'''
    def get_H(self,nsite):
        J=1.
        scfg=SpinSpaceConfig([nsite,2])
        h2=J/4.*(kron(sx,sx)+kron(sz,sz)+kron(sy,sy))
        H=0
        for i in xrange(nsite-1):
            H=H+kron(kron(eye(2**i),h2),eye(2**(nsite-2-i)))

        #impose periodic boundary
        H=H+J/4.*(kron(kron(sx,eye(2**(nsite-2))),sx)+kron(kron(sy,eye(2**(nsite-2))),sy)+kron(kron(sz,eye(2**(nsite-2))),sz))
        return H

    def measure(self,op,state,initial_config=None):
        '''Measure an operator through detailed calculation.'''
        nsite=state.nin
        H=self.get_H(nsite)
        scfg=SpinSpaceConfig([nsite,2])
        if isinstance(op,HeisenbergH):
            op=H
            v=state.tovec(scfg)
            return v.conj().dot(op).dot(v)/sum(v.conj()*v)
        elif isinstance(op,PartialW):
            configs=1-2*scfg.ind2config(arange(scfg.hndim))
            pS=[]
            pS.append(configs)
            theta=state.feed_input(configs)
            pS.append(tanh(theta))
            configs_g=array([state.group.apply(configs,ig) for ig in xrange(state.group.ng)]).swapaxes(0,1)
            pS.append(sum(configs_g[:,:,:,newaxis]*tanh(theta).reshape([scfg.hndim,state.group.ng,1,state.W.shape[1]]),axis=1).reshape([configs.shape[0],-1]))
            pS=concatenate(pS,axis=-1)
            v=state.tovec(scfg)
            v=v/norm(v)
            return sum((v.conj()*v)[:,newaxis]*pS,axis=0)
        elif isinstance(op,OpQueue):
            #get H
            v=state.tovec(scfg); v/=norm(v)
            OH=v.conj().dot(H).dot(v)

            #get W
            configs=1-2*scfg.ind2config(arange(scfg.hndim))
            pS=[]
            pS.append(configs)
            theta=state.feed_input(configs)
            pS.append(tanh(theta))
            configs_g=array([state.group.apply(configs,ig) for ig in xrange(state.group.ng)]).swapaxes(0,1)
            pS.append(sum(configs_g[:,:,:,newaxis]*tanh(theta).reshape([scfg.hndim,state.group.ng,1,state.W.shape[1]]),axis=1).reshape([configs.shape[0],-1]))
            pS=concatenate(pS,axis=-1)
            OPW=sum((v.conj()*v)[:,newaxis]*pS,axis=0)

            OPW2=sum((v.conj()*v)[:,newaxis,newaxis]*(pS.conj()[:,:,newaxis]*pS[:,newaxis]),axis=0)
            #OPWH=(v.conj()[:,newaxis,newaxis]*(pS.conj()[:,newaxis,:]*OH[:,:,newaxis])*v[newaxis,:,newaxis]).sum(axis=(0,1))
            Hloc=H.dot(v)/v
            OPWH=(Hloc[:,newaxis]*pS.conj()*(v.conj()*v)[:,newaxis]).sum(axis=0)
            return OPW,OH,OPW2,OPWH
        else:
            raise TypeError()
