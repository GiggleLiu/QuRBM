from numpy import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
from matplotlib.pyplot import *
import sys,pdb,time
from os import path
sys.path.insert(0,'../')

from tba.hgen import SpinSpaceConfig
from rbm import *

class RBMTest(object):
    def __init__(self):
        a=[0.1,0.2]
        b=[0.1,0.2,0.3]
        W=[[0.1,-0.1,0.1],[-0.1j,0.1,0.1j]]
        S=zeros([3,4],dtype='complex128')
        S[1:,1:]=W
        S[0,1:]=b
        S[1:,0]=a
        self.rbm=RBM(S)
        #get vec in the brute force way.
        self.vec=[]
        scfg1=SpinSpaceConfig([2,2])
        scfg2=SpinSpaceConfig([3,2])
        for i in xrange(scfg1.hndim):
            config1=scfg1.ind2config(i)
            s=1-config1*2
            vi=0j
            for j in xrange(scfg2.hndim):
                config2=scfg2.ind2config(j)
                h=1-2*config2
                #vi+=exp((s*a).sum()+(h*b).sum()+s.dot(asarray(W).dot(h)))
                vi+=exp((s*a).sum()+(h*b).sum()+s.dot(W).dot(h))
            self.vec.append(vi)
        self.vec=array(self.vec)

    def test_tovec(self):
        scfg=SpinSpaceConfig([2,2])
        configs=scfg.ind2config(arange(scfg.hndim))
        vec=self.rbm.get_weight(1-2*configs)
        assert_allclose(vec,self.vec,atol=1e-8)

if __name__=='__main__':
    rt=RBMTest()
    rt.test_tovec()
