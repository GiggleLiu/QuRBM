from numpy import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
from scipy.sparse.linalg import LinearOperator
from scipy.linalg import kron
from matplotlib.pyplot import *
import sys,pdb,time
from os import path
sys.path.insert(0,'../')

from tba.hgen import SpinSpaceConfig,sx,sy,sz
from rbm import *
from sstate import *
from toymodel import *

def test_model1():
    J=1.
    nsite=6
    scfg=SpinSpaceConfig([nsite,2])
    #construct true H
    config=array([1,1,0,0,1,0])
    for periodic in [True,False]:
        h=HeisenbergH(nsite=nsite,periodic=periodic)
        H=FakeVMC(h).get_H()
        print 'Testing rmatmul of Hamiltonian(%s)'%('PBC' if periodic else 'OBC')
        wl,flips=h._rmatmul(1-2*config)
        configs=[]
        for flip in flips:
            nc=copy(1-2*config)
            nc[array(flip)]*=-1
            configs.append(nc)
        ss=SparseState(wl,configs)

        vec=ss.tovec(scfg)
        v0=zeros(scfg.hndim); v0[scfg.config2ind(config)]=1
        v_true=v0.dot(H)
        assert_allclose(vec,v_true)

def test_model2():
    J=1.
    N1,N2=2,3
    scfg=SpinSpaceConfig([N1*N2,2])
    config=array([1,1,0,0,1,0])
    #config=array([1,1,0,0,1,0,1,1,0])
    #construct true H
    for periodic in [True,False]:
        h=HeisenbergH2D(N1,N2,periodic=periodic)
        H=FakeVMC(h).get_H()
        print 'Testing rmatmul of Hamiltonian(%s)'%('PBC' if periodic else 'OBC')
        wl,flips=h._rmatmul(1-2*config)
        configs=[]
        for flip in flips:
            nc=copy(1-2*config)
            nc[asarray(flip)]*=-1
            configs.append(nc)
        ss=SparseState(wl,configs)

        vec=ss.tovec(scfg)
        v0=zeros(scfg.hndim); v0[scfg.config2ind(config)]=1
        v_true=v0.dot(H)
        assert_allclose(vec,v_true)

if __name__=='__main__':
    test_model1()
    test_model2()
