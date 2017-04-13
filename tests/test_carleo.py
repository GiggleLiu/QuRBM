'''
Test Stochastic Reconfiguration with Carleo's model.
'''
from numpy import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
from scipy.linalg import kron,eigh,norm
from matplotlib.pyplot import *
import sys,pdb,time
from os import path
sys.path.insert(0,'../')

from tba.hgen import SpinSpaceConfig,sx,sy,sz,quicksave,quickload
from rbm import *
from toymodel import *
from sr import *
from cgen import *
from vmc import *
from group import TIGroup,NoGroup
from optimizer import *

from test_vmc import analyse_sampling

random.seed(3)

class SRTest(object):
    def __init__(self,nsite,periodic,model='AFH'):
        self.nsite,self.periodic=nsite,periodic
        self.model=model
        if model=='AFH':
            self.h=HeisenbergH(nsite=nsite,J=-1.,Jz=1.,periodic=periodic)
        elif model=='TFI':
            self.h=TFI(nsite=nsite,Jz=-4.,h=-1.,periodic=periodic)
        else:
            raise ValueError()
        self.scfg=SpinSpaceConfig([nsite,2])

        #vmc config
        c0=[-1,1]*(nsite/2); random.shuffle(c0)
        cgen=RBMConfigGenerator(initial_config=c0,nflip=2 if model=='AFH' else 1)#repeat([-1,1],(nsite/2)))
        self.vmc=VMC(cgen,nbath=200*nsite,nsample=2000*nsite,nmeasure=nsite,sampling_method='metropolis')

    def test_sample_carleo(self):
        from utils import load_carleo_wf
        rbm=RBM(*load_carleo_wf('test.wf'))
        nsite=rbm.nin
        h=HeisenbergH(nsite=nsite,J=-4.,Jz=4.,periodic=True)
        c0=[-1,1]*(nsite/2); random.shuffle(c0)
        cgen=RBMConfigGenerator(initial_config=c0,nflip=2)
        vmc=VMC(cgen,nbath=1000*nsite,nsample=10000*nsite,nmeasure=nsite,sampling_method='metropolis')

        E=vmc.measure(h,rbm)
        print 'E/site = %s'%(E/nsite)
        assert_(abs(E/nsite+1.7738)<1e-3)

    def test_carleo(self):
        el=[]
        fname='data/eng-%s-%s%s.dat'%(self.nsite,self.model,'p' if self.periodic else 'o')
        #generate a random rbm and the corresponding vector v
        self.rbm=random_rbm(nin=self.nsite,nhid=self.nsite,group=TIGroup(self.nsite) if self.periodic else NoGroup())
        rbm,info=SR(self.h,self.rbm,handler=self.vmc,niter=600,reg_params=('carleo',{'lambda0':100,'b':0.9}),optimizer=lambda f,g,p: -0.1*0.95**p*g)
        #rbm,info=SR(self.h,self.rbm,handler=self.vmc,niter=200,reg_params=('trunc',{'lambda0':0.2,'eps_trunc':1e-3}))
        #rbm,info=SR(self.h,self.rbm,handler=self.vmc,niter=200,optimizer=RMSProp(rho=0.9,rate=0.001),reg_params=('trunc',{'lambda0':0.2,'eps_trunc':1e-3}))
        #rbm,info=SR(self.h,self.rbm,handler=self.vmc,niter=200,optimizer=RMSProp(rho=0.9,rate=0.001),reg_params=('carleo',{'b':0.9,'lambda0':100}))

        #show error
        el=asarray(zip(*info['opl'])[1])/self.nsite
        e_true=-0.443663
        err=abs(e_true-el)
        print 'E = %s, Error/site = %.4f'%(el,err)
        savetxt(fname,el)
        #assert_(err<0.05)

def show_err_sr(nsite):
    from matplotlib.pyplot import plot,ion
    fname='data/eng-%s-%s%s.dat'%(self.nsite,self.model,'p' if self.periodic else 'o')
    el=loadtxt(fname)
    ion()
    fig=figure(figsize=(5,4))
    for b,c in zip(['p','o'],['gray','k']):
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
    t=SRTest(nsite=16,periodic=False,model='AFH')
    #t.test_sample_carleo()
    t.test_carleo()
    #show_err_sr(nsite=4)
