'''Several Toy Models.'''

from numpy import *
from numpy.linalg import norm

from tba.hgen import SpinSpaceConfig,sx,sy,sz
from linop import LinOp,PartialW
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
            wl.append(self.J/4.*w*(c[:-1]*c[1:]).sum(axis=-1))
            configs.append(c)

            for i in xrange(nsite-1):
                #J/2(S+S- + S-S+) terms
                if c[i]^c[i+1]:
                    nc=copy(c)
                    nc[i:i+2]*=-1
                    wl.append(self.J/2.*w)
                    configs.append(nc)
        return SparseState(wl,configs)

class FakeVMC(object):
    '''The Fake VMC program'''
    def measure(self,op,state,initial_config=None):
        '''Measure an operator through detailed calculation.'''
        nsite=state.nin
        J=1.
        scfg=SpinSpaceConfig([nsite,2])
        h2=J/4.*(kron(sx,sx)+kron(sz,sz)+kron(sy,sy))
        H=0
        for i in xrange(nsite-1):
            H=H+kron(kron(eye(2**i),h2),eye(2**(nsite-2-i)))
        if isinstance(op,HeisenbergH):
            op=H
            v=state.tovec(scfg)
            return v.conj().dot(op).dot(v)/sum(v.conj()*v)
        elif isinstance(op,PartialW):
            configs=1-2*scfg.ind2config(arange(scfg.hndim))
            pS=zeros([scfg.hndim,state.nin+1,state.nhid+1],dtype='complex128')
            pS[:,1:,0]=configs
            configs=concatenate([ones([scfg.hndim,1]),configs],axis=1)
            pS[:,:,1:]=configs[...,newaxis]*tanh(configs.dot(state.S[:,1:]))[:,newaxis]
            v=state.tovec(scfg)
            v=v/norm(v)
            return sum((v.conj()*v)[:,newaxis,newaxis]*pS,axis=0)
        else:
            raise TypeError()
