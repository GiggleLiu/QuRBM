'''Several Toy Models.'''

from numpy import *
from numpy.linalg import norm
import pdb

from tba.hgen import SpinSpaceConfig,sx,sy,sz
from linop import *
from sstate import SparseState

__all__=['TFI','HeisenbergH','FakeVMC']

class TFI(LinOp):
    '''
    Transverse Field Ising Hamiltonian

    H = J S_z*S_z' + h S_x
    '''
    opt_lmul=False

    def __init__(self,nsite,Jz=-1,h=0,periodic=True):
        self.Jz,self.h=Jz,h
        self.nsite=nsite
        self.periodic=periodic

    def _rmatmul(self,config):
        if isinstance(config,ndarray):  #series of {1,-1}
            ws,cs=[1],[config]
        else:
            ws,cs=config.w,config.configs
        nsite=len(cs[0])
        wl,flips=[],[]
        for w,c in zip(ws,cs):
            #J(SzSz) terms.
            nn_par=roll(c,-1)*c
            if not self.periodic: nn_par=nn_par[:-1]
            wl.append(self.Jz/4.*w*(nn_par).sum(axis=-1))
            flips.append([])

            for i in xrange(nsite):
                #h*Sx terms
                wl.append(self.h/2.*w)
                flips.append([i])
        return wl,flips

    def _sandwich(self,cgen,**kwargs):
        wl,flips=self._rmatmul(cgen.config)
        return sum(wl*[cgen.pop(flip)[-1] for flip in flips])

class HeisenbergH(LinOp):
    '''
    Heisenberg Hamiltonian

    H = J S_z*S_z' + J/2(S+S-' + S-S+')
    '''
    opt_lmul=False

    def __init__(self,nsite,J=1.,Jz=None,periodic=True):
        self.J=J
        self.Jz=J if Jz is None else Jz
        self.nsite=nsite
        self.periodic=periodic

    def _rmatmul(self,config):
        if hasattr(config,'__iter__'):  #series of {1,-1}
            ws,cs=[1],[asarray(config)]
        else:
            ws,cs=config.w,config.configs
        nsite=len(cs[0])
        wl,flips=[],[]
        for w,c in zip(ws,cs):
            #J(SzSz) terms.
            nn_par=roll(c,-1)*c
            if not self.periodic: nn_par=nn_par[:-1]
            wl.append(self.Jz/4.*w*(nn_par).sum(axis=-1))
            flips.append([])

            for i in xrange(nsite if self.periodic else nsite-1):
                #J/2(S+S- + S-S+) terms
                j=(i+1)%nsite
                if c[i]^c[j]:
                    wl.append(self.J/2.*w)
                    flips.append([i,j])
        return wl,flips

    def _sandwich(self,cgen,**kwargs):
        wl,flips=self._rmatmul(cgen.config)
        return sum(asarray(wl)*[cgen.pop(flip)[-1] for flip in flips])

class FakeVMC(object):
    '''The Fake VMC program'''
    def __init__(self,h):
        self.h=h
        self.scfg=scfg=SpinSpaceConfig([h.nsite,2])

    def get_H(self):
        '''Get the target Hamiltonian Matrix.'''
        nsite,periodic=self.h.nsite,self.h.periodic
        scfg=SpinSpaceConfig([nsite,2])
        if isinstance(self.h,TFI):
            Jz,h=self.h.Jz,self.h.h
            h2=Jz/4.*kron(sz,sz)
            H=0
            for i in xrange(nsite):
                if i!=nsite-1:
                    H=H+kron(kron(eye(2**i),h2),eye(2**(nsite-2-i)))
                elif periodic: #periodic boundary
                    H=H+Jz/4.*kron(kron(sz,eye(2**(nsite-2))),sz)
                H=H+h/2.*kron(kron(eye(2**i),sx),eye(2**(nsite-i-1)))
            return H
        elif isinstance(self.h,HeisenbergH):
            J,Jz=self.h.J,self.h.Jz
            h2=J/4.*(kron(sx,sx)+kron(sy,sy))+Jz/4.*kron(sz,sz)
            H=0
            for i in xrange(nsite-1):
                H=H+kron(kron(eye(2**i),h2),eye(2**(nsite-2-i)))

            #impose periodic boundary
            if periodic:
                H=H+J/4.*(kron(kron(sx,eye(2**(nsite-2))),sx)+kron(kron(sy,eye(2**(nsite-2))),sy))+Jz/4.*(kron(kron(sz,eye(2**(nsite-2))),sz))
            return H

    def project_vec(self,vec,m=0):
        '''Project vector to good quantum number'''
        scfg=self.scfg
        configs=1-2*scfg.ind2config(arange(scfg.hndim))
        vec[sum(configs,axis=1)!=0]=0
        return vec

    def measure(self,op,state,initial_config=None,**kwargs):
        '''Measure an operator through detailed calculation.'''
        nsite=state.nin
        H=self.get_H()
        scfg=self.scfg
        #prepair state
        v=state.tovec(scfg)
        if isinstance(self.h,HeisenbergH): v=self.project_vec(v,0)
        v/=norm(v)

        if isinstance(op,(HeisenbergH,TFI)):
            return v.conj().dot(H).dot(v)
        elif isinstance(op,PartialW):
            configs=1-2*scfg.ind2config(arange(scfg.hndim))
            pS=[]
            pS.append(configs)
            theta=state.feed_input(configs)
            pS.append(tanh(theta).reshape([len(configs),state.group.ng,len(state.b)]).sum(axis=1))
            configs_g=array([state.group.apply(configs,ig) for ig in xrange(state.group.ng)]).swapaxes(0,1)
            pS.append(sum(configs_g[:,:,:,newaxis]*tanh(theta).reshape([scfg.hndim,state.group.ng,1,state.W.shape[1]]),axis=1).reshape([configs.shape[0],-1]))
            pS=concatenate(pS,axis=-1)
            return sum((v.conj()*v)[:,newaxis]*pS,axis=0)
        elif isinstance(op,OpQueue):
            #get H
            OH=v.conj().dot(H).dot(v)

            #get W
            configs=1-2*scfg.ind2config(arange(scfg.hndim))
            pS=[]
            pS.append(configs)
            theta=state.feed_input(configs)
            pS.append(tanh(theta).reshape([len(configs),state.group.ng,len(state.b)]).sum(axis=1))
            configs_g=array([state.group.apply(configs,ig) for ig in xrange(state.group.ng)]).swapaxes(0,1)
            pS.append(sum(configs_g[:,:,:,newaxis]*tanh(theta).reshape([scfg.hndim,state.group.ng,1,state.W.shape[1]]),axis=1).reshape([configs.shape[0],-1]))
            pS=concatenate(pS,axis=-1)
            OPW=sum((v.conj()*v)[:,newaxis]*pS,axis=0)

            OPW2=sum((v.conj()*v)[:,newaxis,newaxis]*(pS.conj()[:,:,newaxis]*pS[:,newaxis]),axis=0)
            #OPWH=(v.conj()[:,newaxis,newaxis]*(pS.conj()[:,newaxis,:]*OH[:,:,newaxis])*v[newaxis,:,newaxis]).sum(axis=(0,1))
            Hloc=zeros(v.shape,dtype='complex128')
            Hloc[v!=0]=H.dot(v)[v!=0]/v[v!=0]
            #Hloc=H.dot(v)/v
            OPWH=(Hloc[:,newaxis]*pS.conj()*(v.conj()*v)[:,newaxis]).sum(axis=0)
            if op.nop==4:
                return OPW,OH,OPW2,OPWH
            else:
                return OPW,OH,OPWH
        else:
            raise TypeError()
