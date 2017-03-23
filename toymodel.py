'''Several Toy Models.'''

from numpy import *

from linop import LinOp
from sstate import SparseState

__all__=['HeisenbergH']

class HeisenbergH(LinOp):
    '''
    Heisenberg Hamiltonian

    H = J S_z*S_z' + J/2(S+S-' + S-S+')
    '''
    def __init__(self,nsite,J=1):
        self.J=J
        self.nsite=nsite

    def rmatmul(self,target):
        if isinstance(target,ndarray):
            ws,cs=[1],[target]
        else:
            ws,cs=target.w,target.configs
        nsite=len(cs[0])
        wl,configs=[],[]
        for w,c in zip(ws,cs):
            #J(SzSz) terms.
            wl.append(self.J/2.*w*(0.5-(c[:-1]^c[1:])).sum(axis=-1))
            configs.append(c)

            for i in xrange(nsite-1):
                #J/2(S+S- + S-S+) terms
                if c[i]^c[i+1]:
                    nc=copy(c)
                    nc[i:i+2]=1-nc[i:i+2]
                    wl.append(self.J/2.*w)
                    configs.append(nc)
        return SparseState(wl,configs)

    def lmatmul(self,target):
        pass
