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

def test_model():
    J=1.
    scfg=SpinSpaceConfig([4,2])
    #construct true H
    I2,I4=eye(2),eye(4)
    h2=J/4.*(kron(sx,sx)+kron(sz,sz)+kron(sy,sy))
    H=kron(h2,I4)+kron(kron(I2,h2),I2)+kron(I4,h2)
    h=HeisenbergH(nsite=4)
    config=array([1,1,0,1])
    print 'Testing rmatmul of Hamiltonian'
    res=h.rmatmul(1-2*config)
    vec=res.tovec(scfg)
    v0=zeros(scfg.hndim); v0[scfg.config2ind(config)]=1
    v_true=v0.dot(H)
    assert_allclose(vec,v_true)

if __name__=='__main__':
    test_model()
