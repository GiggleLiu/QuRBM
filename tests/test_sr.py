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
from cgen import *
from vmc import *
from group import TIGroup,NoGroup
from optimizer import *

from test_vmc import analyse_sampling

random.seed(2)

class SRTest(object):
    def __init__(self,nsite,periodic,model='AFH'):
        self.nsite,self.periodic=nsite,periodic
        self.model=model
        if model=='AFH':
            self.h=HeisenbergH(nsite=nsite,J=1.,Jz=1.,periodic=periodic)
        elif model=='TFI':
            self.h=TFI(nsite=nsite,Jz=-4.,h=-1.,periodic=periodic)
        else:
            raise ValueError()
        self.scfg=SpinSpaceConfig([nsite,2])
        self.fv=FakeVMC(self.h)

        #vmc config
        cgen=RBMConfigGenerator(initial_config=[-1,1]*(nsite/2)+[1]*(nsite%2),nflip=2 if model=='AFH' else 1)
        self.vmc=VMC(cgen,nbath=2000,nsample=20000,nmeasure=nsite,sampling_method='metropolis')

    def test_sr_fake(self):
        b=0.9
        el=[]
        H=self.fv.get_H()
        e_true,v_true=eigh(H)
        #generate a random rbm and the corresponding vector v
        self.rbm=random_rbm(nin=self.nsite,nhid=self.nsite,group=TIGroup(self.nsite) if self.periodic else NoGroup())
        for k in xrange(100):
            print 'Running %s-th batch.'%k
            #rbm,info=SR(self.h,self.rbm,handler=self.fv,niter=1,optimizer=DefaultOpt(rate=0.1),reg_params=('delta',{'lambda0':1e-4}))
            #rbm,info=SR(self.h,self.rbm,handler=self.fv,niter=1,optimizer=RMSProp(rho=0.8,rate=0.01),reg_params=('carleo',{'b':b,'lambda0':100*b**(k)}))
            #rbm,info=SR(self.h,self.rbm,handler=self.fv,niter=1,optimizer=RMSProp(rho=0.8,rate=0.01),reg_params=('trunc',{'lambda0':0.2,'eps_trunc':1e-3}))
            #rbm,info=SD(self.h,self.rbm,handler=self.fv,niter=1,optimizer=DefaultOpt(0.1))
            rbm,info=SD(self.h,self.rbm,handler=self.fv,niter=1,optimizer=RMSProp(rho=0.9,rate=0.001))
    
            v=rbm.tovec(self.scfg)
            if self.model=='AFH': self.fv.project_vec(v,0)
            v=v/norm(v)
            err_v=1-abs(v.conj().dot(v_true[:,0]))
            ei=info['opl'][-1][1]
            err=abs(e_true[0]-ei)/(abs(e_true[0])+abs(ei))
            print 'Error = %.4f%%, Err_v = %.4f%%'%(err*100,err_v*100)
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
            #rbm,info=SR(self.h,self.rbm,handler=self.vmc,niter=1,optimizer=RMSProp(rho=0.7,rate=0.1),reg_params=('carleo',{'lambda0':100*b**k,'b':b}))
            rbm,info=SR(self.h,self.rbm,handler=self.vmc,niter=1,optimizer=RMSProp(rho=0.8,rate=0.01),reg_params=('trunc',{'lambda0':0.2,'eps_trunc':1e-3}))
            #rbm,info=SR(self.h,self.rbm,handler=self.vmc,niter=1,optimizer=DefaultOpt(rate=0.1),reg_params=('carleo',{'lambda0':100*b**k,'b':b}))
            v=rbm.tovec(self.scfg); v=v/norm(v)
            #err=1-abs(v.conj().dot(v_true[:,0]))
            #ei=v.conj().dot(H).dot(v)
            ei=info['opl'][-1][1]
            err=abs(e_true[0]-ei)/(abs(e_true[0])+abs(ei))
            print 'E = %s, Error = %.4f%%'%(ei/self.nsite,err*100)
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
    t=SRTest(nsite=4,periodic=False,model='AFH')
    t.test_sr_fake()
    #t.test_sr()
    #show_err_sr(nsite=4)
