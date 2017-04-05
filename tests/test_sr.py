'''
Test Stochastic Reconfiguration with minimal model.
'''
from numpy import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
from scipy.linalg import kron,eigh,norm
from matplotlib.pyplot import *
import sys,pdb,time
from os import path
sys.path.insert(0,'../')

from tba.hgen import SpinSpaceConfig,sx,sy,sz
from rbm import *
from toymodel import *
from sr import *
from mccore_rbm import *
from vmc import *
from group import TIGroup,NoGroup

from test_vmc import analyse_sampling

random.seed(2)

class SRTest(object):
    def __init__(self,nsite,periodic,model='AFH'):
        self.nsite,self.periodic=nsite,periodic
        if model=='AFH':
            self.h=HeisenbergH(nsite=nsite,J=4.,Jz=4.,periodic=periodic)
        elif model=='TFI':
            self.h=TFI(nsite=nsite,Jz=-4.,h=-1.,periodic=periodic)
        else:
            raise ValueError()
        self.scfg=SpinSpaceConfig([nsite,2])
        self.fv=FakeVMC(self.h)

        #vmc config
        core=RBMCore(initial_config=[-1,1]*(nsite/2)+[1]*(nsite%2))
        self.vmc=VMC(core,nbath=200,nsample=50000,nmeasure=nsite,sampling_method='metropolis')

    def test_sr_fake(self):
        b=0.9
        el=[]
        H=self.fv.get_H()
        e_true,v_true=eigh(H)
        #generate a random rbm and the corresponding vector v
        self.rbm=random_rbm(nin=self.nsite,nhid=self.nsite,group=TIGroup(self.nsite) if self.periodic else NoGroup())
        for k in xrange(100):
            print 'Running %s-th batch.'%k
            rbm=SR(self.h,self.rbm,handler=self.fv,niter=1,gamma=0.2,reg_params=('delta',{'b':b,'lambda0':100*b**(5*k)}))
            v=rbm.tovec(self.scfg)
            v=v/norm(v)
            #err=1-abs(v.conj().dot(v_true[:,0]))
            ei=v.conj().dot(H).dot(v)
            err=abs(e_true[0]-ei)/(abs(e_true[0])+abs(ei))
            print 'Error = %.4f%%'%(err*100)
            el.append(err)
        savetxt('data/err0-%s%s.dat'%(self.nsite,'p' if self.periodic else 'o'),el)
        assert_(err<0.01)

    def test_sr(self):
        b=0.85
        el=[]
        H=self.fv.get_H()
        e_true,v_true=eigh(H)
        #generate a random rbm and the corresponding vector v
        self.rbm=random_rbm(nin=self.nsite,nhid=self.nsite,group=TIGroup(self.nsite) if self.periodic else NoGroup())
        for k in xrange(100):
            print 'Running %s-th batch.'%k
            rbm,info=SR(self.h,self.rbm,handler=self.vmc,niter=1,gamma=0.2,reg_params=('delta',{'lambda0':100*b**(5*k),'b':b}))
            v=rbm.tovec(self.scfg); v=v/norm(v)
            #err=1-abs(v.conj().dot(v_true[:,0]))
            ei=v.conj().dot(H).dot(v)
            err=abs(e_true[0]-ei)/(abs(e_true[0])+abs(ei))
            print 'Error = %.4f%%'%(err*100)
            el.append(err)
        savetxt('data/err-%s%s.dat'%(self.nsite,'p' if self.periodic else 'o'),el)
        assert_(err<0.05)

def show_err_sr(nsite):
    from matplotlib.pyplot import plot,ion
    ion()
    fig=figure(figsize=(5,4))
    for b,c in zip(['p','o'],['gray','k']):
        f='data/err-%s%s.dat'%(nsite,b)
        f0='data/err0-%s%s.dat'%(nsite,b)
        el=loadtxt(f)
        el0=loadtxt(f0)
        plot(el0,lw=2,color=c)
        plot(el,lw=2,color=c,ls='--')
    xlabel('iteration',fontsize=16)
    ylabel(r'$Err=\frac{|E-E_0|}{|E|+|E_0|}$',fontsize=16)
    #ylabel(r'$1-\|\left\langle\psi|\tilde{\psi}\right\rangle\|_2$',fontsize=16)
    ylim(1e-8,1)
    yscale('log')
    legend(['Exact/Periodic','VMC/Periodic','Exact/Open','VMC/Open'],loc=3)
    tight_layout()
    pdb.set_trace()
    savefig('data/err-%s.pdf'%nsite)

if __name__=='__main__':
    t=SRTest(nsite=5,periodic=True,model='TFI')
    #t.test_sr_fake()
    t.test_sr()
    #show_err_sr(nsite=4)
