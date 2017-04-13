'''Several Toy Models.'''

from numpy import *
from numpy.linalg import norm
import pdb

from tba.hgen import SpinSpaceConfig,sx,sy,sz
from linop import *
from sstate import SparseState

__all__=['TFI','HeisenbergH','FakeVMC','HeisenbergH2D']

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
            flips.append(array([],dtype='int64'))

            for i in xrange(nsite):
                #h*Sx terms
                wl.append(self.h/2.*w)
                flips.append(array([i],dtype='int64'))
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
            flips.append(array([],dtype='int64'))

            mask=nn_par!=1
            i=where(mask)[0]
            j=(i+1)%nsite

            wl+=[self.J/2.*w]*len(i)
            flips+=zip(i,j)

        return wl,flips

    def _sandwich(self,cgen,**kwargs):
        wl,flips=self._rmatmul(cgen.config)
        return sum(asarray(wl)*[cgen.pop(flip)[-1] for flip in flips])

class HeisenbergH2D(LinOp):
    '''
    Heisenberg Hamiltonian, 2D version.

    H = J S_z*S_z' + J/2(S+S-' + S-S+')
    '''
    opt_lmul=False

    def __init__(self,N1,N2,J=1.,Jz=None,periodic=True):
        self.J=J
        self.Jz=J if Jz is None else Jz
        self.periodic=periodic
        self.N1,self.N2=N1,N2

    @property
    def nsite(self):
        return self.N1*self.N2

    def _rmatmul(self,config):
        N1,N2=self.N1,self.N2
        config2d=config.reshape([N1,N2])
        wl,flips=[],[]
        #J(SzSz) terms.
        nn_par1=roll(config2d,-1,axis=0)*config2d
        nn_par2=roll(config2d,-1,axis=1)*config2d
        if not self.periodic:
            nn_par1=nn_par1[:-1,:]
            nn_par2=nn_par2[:,:-1]
        wl.append(self.Jz/4.*(nn_par1.sum()+nn_par2.sum()))
        flips.append(array([],dtype='int64'))

        #bond in 1st direction
        mask1=nn_par1!=1
        i1,i2=where(mask1)
        j1,j2=(i1+1)%N1,i2
        i=i1*N2+i2
        j=j1*N2+j2
        wl+=[self.J/2.]*len(i)
        flips+=zip(i,j)

        #bond in 2nd direction
        mask2=nn_par2!=1
        i1,i2=where(mask2)
        j1,j2=i1,(i2+1)%N2
        i=i1*N2+i2
        j=j1*N2+j2
        wl+=[self.J/2.]*len(i)
        flips+=zip(i,j)
        return wl,flips

    def _sandwich(self,cgen,**kwargs):
        wl,flips=self._rmatmul(cgen.config)
        return sum(asarray(wl)*[cgen.pop(flip)[-1] for flip in flips])

    def visualize(self,config,**kwargs):
        from matplotlib.pyplot import pcolor
        from matplotlib import cm
        config=config.reshape([self.N1,self.N2])
        pcolor(config,cmap=cm.gray,**kwargs)
        axis('equal')
        axis('off')

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
        elif isinstance(self.h,HeisenbergH2D):
            from tba.lattice import Square_Lattice
            J,Jz=self.h.J,self.h.Jz
            lattice=Square_Lattice(N=(self.h.N1,self.h.N2))
            if self.h.periodic: lattice.set_periodic([True,True])
            lattice.initbonds(5)
            b1s=lattice.getbonds(1)
            b1s=[b for b in b1s if b.atom1<b.atom2]
            H=0
            for b in b1s:
                for ss,Ji in zip([sx,sy,sz],[J,J,Jz]):
                    H=H+Ji/4.*kron(kron(kron(kron(eye(2**b.atom1),ss),eye(2**(b.atom2-b.atom1-1))),ss),eye(2**(lattice.nsite-b.atom2-1)))
                    #H=H+Ji/4.*kron(kron(kron(kron(eye(2**(lattice.nsite-b.atom2-1)),ss),eye(2**(b.atom2-b.atom1-1))),ss),eye(2**(b.atom1)))
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
            configs_g=state.group.apply_all(configs).swapaxes(0,1)
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
            configs_g=state.group.apply_all(configs).swapaxes(0,1)
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
