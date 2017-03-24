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

class TestLinop(object):
    def __init__(self):
        J=1.
        nsite=4
        self.scfg=SpinSpaceConfig([nsite,2])

        #construct true H
        I2,I4=eye(2),eye(4)
        h2=J/4.*(kron(sx,sx)+kron(sz,sz)+kron(sy,sy))
        self.H=kron(h2,I4)+kron(kron(I2,h2),I2)+kron(I4,h2)

        #construct operator H act on config
        self.h=HeisenbergH(nsite=4)

        #generate a random rbm and the corresponding vector v
        self.rbm=random_rbm(nin=nsite,nhid=nsite)
        self.v=self.rbm.tovec(self.scfg)

    def test_sandwich(self):
        print 'Testing sandwich.'
        H,v=self.H,self.v
        config=random.randint(0,2,4)
        ket=zeros(self.scfg.hndim); ket[self.scfg.config2ind(config)]=1
        O_true=ket.conj().dot(H).dot(v)/sum(ket.conj()*v)
        O_s=c_sandwich(self.h,1-config*2,self.rbm)

        assert_almost_equal(O_s,O_true,decimal=8)

if __name__=='__main__':
    tl=TestLinop()
    tl.test_sandwich()
