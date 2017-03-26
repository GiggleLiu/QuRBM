from numpy import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
from matplotlib.pyplot import *
import sys,pdb,time
from os import path
sys.path.insert(0,'../')

from tba.hgen import SpinSpaceConfig,sx,sy,sz
from toymodel import *
from rbm import *
from linop import *
from group import *

class TestLinop(object):
    def __init__(self):
        J=1.
        nsite=4
        self.scfg=SpinSpaceConfig([nsite,2])

        #construct true H
        I2,I4=eye(2),eye(4)
        h2=J/4.*(kron(sx,sx)+kron(sz,sz)+kron(sy,sy))
        self.H=kron(h2,I4)+kron(kron(I2,h2),I2)+kron(I4,h2)+J/4.*(kron(kron(sz,I4),sz)+kron(kron(sy,I4),sy)+kron(kron(sx,I4),sx))

        #construct operator H act on config
        self.h=HeisenbergH(nsite=4)
        self.pW=PartialW()

        #generate a random rbm and the corresponding vector v
        self.rbm=random_rbm(nin=nsite,nhid=nsite)
        self.v=self.rbm.tovec(self.scfg)

        self.rbm_g=random_rbm(nin=nsite,nhid=nsite,group=TIGroup(2))
        self.v_g=self.rbm_g.tovec(self.scfg)

    def test_sandwichh(self):
        print 'Testing sandwich H.'
        config=random.randint(0,2,4)
        ket=zeros(self.scfg.hndim); ket[self.scfg.config2ind(config)]=1

        H,v,v_g=self.H,self.v,self.v_g
        O_true=ket.conj().dot(H).dot(v)/sum(ket.conj()*v)
        O_s=c_sandwich(self.h,1-config*2,self.rbm)

        O_sg=c_sandwich(self.h,1-config*2,self.rbm_g)
        O_trueg=ket.conj().dot(H).dot(v_g)/sum(ket.conj()*v_g)

        assert_almost_equal(O_s,O_true,decimal=8)
        assert_almost_equal(O_sg,O_trueg,decimal=8)

    def test_sandwichpw(self):
        print 'Testing sandwich PartialW.'
        config=random.randint(0,2,4)
        ket=zeros(self.scfg.hndim); ket[self.scfg.config2ind(config)]=1
        H,v,v_g=self.H,self.v,self.v_g

        O_true=ket.conj().dot(Wmat).dot(v)/sum(ket.conj()*v)
        O_s=c_sandwich(self.pW,1-config*2,self.rbm)

        O_trueg=ket.conj().dot(Wmat).dot(v_g)/sum(ket.conj()*v_g)
        O_sg=c_sandwich(self.pW,1-config*2,self.rbm_g)

        assert_almost_equal(O_s,O_true,decimal=8)
        assert_almost_equal(O_sg,O_trueg,decimal=8)

if __name__=='__main__':
    tl=TestLinop()
    tl.test_sandwichh()
    #tl.test_sandwichpw()
