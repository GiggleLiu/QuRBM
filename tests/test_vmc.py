from numpy import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
from scipy.sparse.linalg import LinearOperator
from scipy.linalg import kron
from matplotlib.pyplot import *
import sys,pdb,time
from os import path
sys.path.insert(0,'../')

from tba.hgen import SpinSpaceConfig,sx,sy,sz
from vmc import *
from rbm import *
from toymodel import *
from mccore_rbm import *
from linop import *

random.seed(2)

def test_measureh():
    print 'VMC measurements.'
    nsite=4

    #construct operator H act on config
    h=HeisenbergH(nsite=nsite)

    #generate a random rbm and the corresponding vector v
    rbm=random_rbm(nin=nsite,nhid=nsite)

    #vmc config
    core=RBMCore()
    vmc=VMC(core,nbath=50,nsample=10000,sampling_method='metropolis')

    #measurements
    O_true=FakeVMC().measure(h,rbm)
    O_vmc=vmc.measure(h,rbm)

    err=abs(O_vmc-O_true)/abs(O_true)
    print 'Error = %.4f%%'%(err*100)
    assert_(err<0.05)

def test_measurepw():
    print 'VMC measurements.'
    nsite=4

    #construct operator H act on config
    pw=PartialW()

    #generate a random rbm and the corresponding vector v
    rbm=random_rbm(nin=nsite,nhid=nsite)

    #vmc config
    core=RBMCore()
    vmc=VMC(core,nbath=1000,nsample=100000,sampling_method='metropolis')

    #measurements
    O_true=FakeVMC().measure(pw,rbm)
    O_vmc=vmc.measure(pw,rbm)

    err=abs(O_vmc-O_true).sum()/abs(O_true).sum()
    print 'Error = %.4f%%'%(err*100)
    pdb.set_trace()
    #assert_(err<0.01)

if __name__=='__main__':
    #test_measureh()
    test_measurepw()
