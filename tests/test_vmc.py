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

random.seed(21)

def test_measureh():
    print 'VMC measurements on HeisenbergH.'
    nsite=4

    #construct operator H act on config
    h=HeisenbergH(nsite=nsite)

    #generate a random rbm and the corresponding vector v
    rbm=random_rbm(nin=nsite,nhid=nsite)

    #vmc config
    core=RBMCore()
    vmc=VMC(core,nbath=50,nsample=50000,sampling_method='metropolis')

    #measurements
    O_true=FakeVMC().measure(h,rbm)
    O_vmc=vmc.measure(h,rbm)

    err=abs(O_vmc-O_true)/abs(O_true)
    print 'Error = %.4f%%'%(err*100)
    assert_(err<0.05)

def test_measurepw():
    print 'VMC measurements on PartialW.'
    nsite=4

    #construct operator pw act on config
    pw=PartialW()

    #generate a random rbm and the corresponding vector v
    rbm=random_rbm(nin=nsite,nhid=nsite)

    #vmc config
    core=RBMCore()
    vmc=VMC(core,nbath=1000,nsample=50000,sampling_method='metropolis')

    #measurements
    O_true=FakeVMC().measure(pw,rbm)
    O_vmc=vmc.measure(pw,rbm)

    err=abs(O_vmc-O_true).sum()/abs(O_true).sum()
    print 'Error = %.4f%%'%(err*100)
    assert_(err<0.08)

def test_measureq():
    print 'VMC measurements on OpQueue.'
    nsite=4

    #construct operator q act on config
    pw=PartialW()
    h=HeisenbergH(nsite=nsite)
    q=OpQueue((pw,h),(lambda a,b:a.conj()[...,newaxis,newaxis]*a,lambda a,b:b*a.conj()))

    #generate a random rbm and the corresponding vector v
    rbm=random_rbm(nin=nsite,nhid=nsite)

    #vmc config
    core=RBMCore()
    vmc=VMC(core,nbath=200,nsample=50000,sampling_method='metropolis')

    #measurements
    O_trues=FakeVMC().measure(q,rbm)
    O_vmcs=vmc.measure(q,rbm)

    for O_true,O_vmc in zip(O_trues,O_vmcs):
        err=abs(O_vmc-O_true).sum()/abs(O_true).sum()
        print 'Error = %.4f%%'%(err*100)
        assert_(err<0.05)

if __name__=='__main__':
    test_measureq()
    test_measureh()
    test_measurepw()
