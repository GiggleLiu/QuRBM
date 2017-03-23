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

def test_vec2sstate():
    print 'Testing @vec2sstate, @SparseState.tovec'
    scfg=SpinSpaceConfig([4,2])
    vec=2*random.random(scfg.hndim)-1
    assert_allclose(vec2sstate(vec,scfg).tovec(scfg),vec,atol=1e-8)

def test_soverlap():
    s1=SparseState([0.1j,0.2],[[1,1,0,1],[1,0,1,0]])
    s2=SparseState([0.5j,0.2j],[[1,1,0,1],[1,1,1,0]])
    assert_(s1*s2==0.05)

def test_visualize():
    ss=SparseState([0.1j,0.2],[[1,1,0,1],[1,0,1,0]])
    ion()
    visualize_sstate(ss)
    colorbar()
    pause(1)

def test_compress():
    scfg=SpinSpaceConfig([4,2])
    ss=SparseState([0.1j,0.2,0.3],[[1,1,0,1],[1,0,1,0],[1,1,0,1]])
    assert_allclose(ss.ws,[0.2,0.3+0.1j])
    assert_allclose(ss.configs,[[1,0,1,0],[1,1,0,1]])

def test_model():
    J=1.
    scfg=SpinSpaceConfig([4,2])
    #construct true H
    I2,I4=eye(2),eye(4)
    h2=J/4.*(kron(sx,sx)+kron(sz,sz)+kron(sy,sy))
    H=kron(h2,I4)+kron(kron(I2,h2),I2)+kron(I4,h2)
    h=HeisenbergH(nsite=4)
    config=array([1,1,0,1])
    res=h.rmatmul(config)
    vec=res.tovec(scfg)
    v0=zeros(scfg.hndim); v0[scfg.config2ind(config)]=1
    v_true=v0.dot(H)
    assert_allclose(vec,v_true)

if __name__=='__main__':
    test_soverlap()
    test_compress()
    test_vec2sstate()
    test_visualize()
    test_model()
