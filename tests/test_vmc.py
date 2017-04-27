from numpy import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
from scipy.sparse.linalg import LinearOperator
from scipy.linalg import kron,norm
from matplotlib.pyplot import *
import sys,pdb,time
from os import path
sys.path.insert(0,'../')

from tba.hgen import SpinSpaceConfig,sx,sy,sz
from vmc import *
from rbm import *
from toymodel import *
from cgen import *
from linop import *
from group import TIGroup

random.seed(21)

def analyse_sampling(configs,rbm):
    scfg=SpinSpaceConfig([rbm.nin,2])
    s=zeros(scfg.hndim)
    add.at(s,scfg.config2ind((1-asarray(configs))/2),1)
    v0=rbm.tovec(scfg)
    s2=abs(v0/norm(v0))**2
    ion()
    plot(s/s.sum())
    plot(s2)
    pdb.set_trace()

class VMCTest(object):
    def __init__(self,model='AFH'):
        self.nsite=4
        #construct operator H act on config
        if model=='AFH':
            self.h=HeisenbergH(nsite=self.nsite,J=1.,periodic=True)
        elif model=='AFH2D':
            N=int(sqrt(self.nsite))
            self.h=HeisenbergH2D(N,N,J=-4.,Jz=2.,periodic=True)

        #generate a rbm and the corresponding vector v
        self.rbm=RBM(a=[0.1,0.2j,0.3,-0.5],b=[-0.1,0.2,0.,-0.5j],W=kron(sx,sx)+kron(sy,sy))
        self.rbm_g=RBM(a=[0.1,0.2j,0.3,-0.5],b=[-0.5j],\
                W=reshape([0.3,-0.2,0.4j,0.1],[self.nsite,1]),group=TIGroup([self.nsite] if model!='AFH2D' else [N,N]))

        #vmc config
        cgen=RBMConfigGenerator(nflip=2,initial_config=array([-1,1]*2))
        self.vmc=VMC(cgen,nbath=500*self.nsite,nsample=5000*self.nsite,nmeasure=self.nsite,sampling_method='metropolis')

        #fake vmc
        self.fv=FakeVMC(self.h)

    def test_measureh(self):
        print 'VMC measurements on HeisenbergH.'
        for rbm in [self.rbm_g,self.rbm]:
            #measurements
            O_true=self.fv.measure(self.h,rbm)/self.nsite
            O_vmc=self.vmc.measure(self.h,rbm)/self.nsite

            err=abs(O_vmc-O_true)
            print 'E/site = %s (%s), Error/site = %s'%(O_vmc,O_true,err)
            #analyse_sampling(self.vmc._config_histo,rbm)
            assert_(err<0.1)

    def test_measurepw(self):
        print 'VMC measurements on PartialW.'
        #construct operator pw act on config
        pw=PartialW()

        for rbm in [self.rbm_g,self.rbm]:
            #measurements
            O_true=self.fv.measure(pw,rbm)
            O_vmc=self.vmc.measure(pw,rbm)

            err=abs(O_vmc-O_true).mean()
            print 'Error = %.4f%%'%(err*100)
            #analyse_sampling(self.vmc._config_histo,rbm)
            assert_(err<0.1)

    def test_measureq(self):
        print 'VMC measurements on OpQueue.'
        #construct operator q act on config
        pw=PartialW()
        q=OpQueue((pw,self.h),(lambda a,b:a.conj()[...,newaxis]*a,lambda a,b:a.conj()*b))

        for rbm in [self.rbm_g,self.rbm]:
            #measurements
            O_trues=self.fv.measure(q,rbm)
            O_vmcs=self.vmc.measure(q,rbm)

            for O_true,O_vmc in zip(O_trues,O_vmcs):
                err=abs(O_vmc-O_true).mean()
                print 'Error = %.4f%%'%(err*100)
                #assert_(err<0.1)

if __name__=='__main__':
    t=VMCTest(model='AFH2D')
    t.test_measureh()
    pdb.set_trace()
    t.test_measureq()
    t.test_measurepw()
