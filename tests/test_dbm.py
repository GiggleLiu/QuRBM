from numpy import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
from matplotlib.pyplot import *
import sys,pdb,time
from os import path
sys.path.insert(0,'../')

from tba.hgen import SpinSpaceConfig
from dbm import *
from group import *

class RBMTest(object):
    def __init__(self):
        a=[0.1j,0.2]
        b=[0.1j,0.2,0.3]
        b2=[-0.1j,0.2,-0.3]
        W=[[0.1,-0.1,0.1],[-0.1j,0.1,0.1j]]
        self.rbm=RBM(a,b,W)

        #translational invariant group
        tig=TIGroup(ngs=[2])
        self.rbm_t=RBM(a,b2,W,group=tig)

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
                vi+=exp((s*a).sum()+(h*b).sum()+s.dot(W).dot(h))
            self.vec.append(vi)
        self.vec=array(self.vec)

        #the second vec
        self.vec2=[]
        scfg1=SpinSpaceConfig([2,2])
        scfg2=SpinSpaceConfig([6,2])
        for i in xrange(scfg1.hndim):
            config1=scfg1.ind2config(i)
            s=1-config1*2
            vi=0j
            for j in xrange(scfg2.hndim):
                config2=scfg2.ind2config(j)
                h=1-2*config2
                vi+=exp((s*a).sum()+(s[::-1]*a).sum()+(h*concatenate([b2,b2])).sum()+s.dot(W).dot(h[:3])+s[::-1].dot(W).dot(h[3:]))
            self.vec2.append(vi)
        self.vec2=array(self.vec2)

    def test_tovec(self):
        print 'Test @RBM.tovec'
        scfg=SpinSpaceConfig([2,2])
        configs=scfg.ind2config(arange(scfg.hndim))
        vec=self.rbm.get_weight(1-2*configs)
        assert_allclose(vec,self.vec,atol=1e-8)

        print 'Test Translational invariant @RBM.tovec.'
        vect=self.rbm_t.tovec(scfg)
        assert_allclose(vect,self.vec2,atol=1e-8)

def test_randomdbm():
    dbm=random_dbm(dims=[2,3,4,2])
    print 'Testing __str__.'
    print dbm,[dbm]
    print 'Testing dump, load to array.'
    dbm_arr=dbm.dump_arr()
    dbm2=random_dbm(dims=[2,3,4,2])
    dbm2.load_arr(dbm_arr)
    for W1,W2 in zip(dbm.W_L,dbm2.W_L):
        assert_allclose(W1,W2)
    for b1,b2 in zip(dbm.b_L,dbm2.b_L):
        assert_allclose(b1,b2)
    print 'Testing dump, load to array - with mask.'
    cmask=[True,True,False,False,True,False,False]
    dbm3=random_dbm(dims=[2,3,4,2],var_mask=cmask)
    dbm.var_mask=cmask
    dbm3.load_arr(dbm.dump_arr())
    assert_allclose(dbm3.W_L[0],dbm.W_L[0])
    assert_allclose(dbm3.b_L[0],dbm.b_L[0])
    assert_allclose(dbm3.b_L[1],dbm.b_L[1])
    assert_(abs(dbm3.W_L[1]-dbm.W_L[1]).sum()>1e-3)
    assert_(abs(dbm3.W_L[2]-dbm.W_L[2]).sum()>1e-3)
    assert_(abs(dbm3.b_L[2]-dbm.b_L[2]).sum()>1e-3)
    assert_(abs(dbm3.b_L[3]-dbm.b_L[3]).sum()>1e-3)

if __name__=='__main__':
    test_randomdbm()
