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
from mccore_rbm import *
from vmc import *
from group import TIGroup,NoGroup

from test_vmc import analyse_sampling

random.seed(2)

class SRTest(object):
    def __init__(self,nsite,periodic,model='AFH'):
        self.nsite,self.periodic=nsite,periodic
        self.model=model
        if model=='AFH':
            self.h=HeisenbergH(nsite=nsite,J=4.,Jz=4.,periodic=periodic)
        elif model=='TFI':
            self.h=TFI(nsite=nsite,Jz=-4.,h=-1.,periodic=periodic)
        else:
            raise ValueError()
        self.scfg=SpinSpaceConfig([nsite,2])

        #vmc config
        core=RBMCore(initial_config=random.shuffle([-1,1]*(nsite/2)))#repeat([-1,1],(nsite/2)))
        self.vmc=VMC(core,nbath=500*nsite,nsample=3000*nsite,nmeasure=nsite,sampling_method='metropolis')

    def test_sr(self):
        b=0.9
        el=[]
        fname='data/eng-%s-%s%s.dat'%(self.nsite,self.model,'p' if self.periodic else 'o')
        #generate a random rbm and the corresponding vector v
        self.rbm=random_rbm(nin=self.nsite,nhid=self.nsite,group=TIGroup(self.nsite) if self.periodic else NoGroup())
        e_true=-0.443663
        for k in xrange(100):
            print 'Running %s-th batch.'%k
            rbm,info=SR(self.h,self.rbm,handler=self.vmc,niter=1,gamma=0.1,reg_params=('delta',{'lambda0':100*b**(5*k),'b':b}))
            ei=info['opl'][-1][1]
            err=abs(e_true-ei/self.nsite)/(abs(e_true)+abs(ei/self.nsite))
            print 'E = %s, Error = %.4f%%'%(ei/self.nsite,err*100)
            el.append(ei)
        savetxt(fname,el)
        assert_(err<0.05)

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
    t=SRTest(nsite=40,periodic=True,model='AFH')
    #t.test_sr_fake()
    t.test_sr()
    #show_err_sr(nsite=4)
