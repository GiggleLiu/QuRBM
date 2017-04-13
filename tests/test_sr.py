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
from climin import RmsProp,GradientDescent,Adam

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
        elif model=='AFH2D':
            N1=N2=int(sqrt(nsite))
            self.h=HeisenbergH2D(N1,N2,J=1.,Jz=1.,periodic=periodic)
        else:
            raise ValueError()
        self.scfg=SpinSpaceConfig([nsite,2])
        self.fv=FakeVMC(self.h)

        #vmc config
        cgen=RBMConfigGenerator(initial_config=[-1,1]*(nsite/2)+[1]*(nsite%2),nflip=2 if model=='AFH' else 1)
        self.vmc=VMC(cgen,nbath=2000,nsample=20000,nmeasure=nsite,sampling_method='metropolis')

    def test_sr(self,fakevmc=False):
        el=[]
        H=self.fv.get_H()
        e_true,v_true=eigh(H)
        #generate a random rbm and the corresponding vector v
        group=(TIGroup(self.nsite if not isinstance(self.h,HeisenbergH2D) else 2*[int(sqrt(self.nsite))])) if self.periodic else NoGroup()
        self.rbm=random_rbm(nin=self.nsite,nhid=self.nsite,group=group)
        reg_params=('delta',{'lambda0':1e-4})
        #reg_params=('trunc',{'lambda0':0.2,'eps_trunc':1e-3})
        #reg_params=('carleo',{'lambda0':100,'b':0.9})
        #reg_params=('identity',{})
        #reg_params=('pinv',{})
        sr=SR(self.h,self.rbm,handler=self.vmc if not fakevmc else self.fv,reg_params=reg_params)
        optimizer=RmsProp(wrt=self.rbm.dump_arr(),fprime=sr.compute_gradient,step_rate=2e-3,decay=0.9,momentum=0.5)
        #optimizer=Adam(wrt=self.rbm.dump_arr(),fprime=sr.compute_gradient,step_rate=1e-2)
        #optimizer=GradientDescent(wrt=self.rbm.dump_arr(),fprime=sr.compute_gradient,step_rate=5e-2,momentum=0.)
        for k,info in enumerate(optimizer):
            print 'Running %s-th Iteration.'%k
            v=self.rbm.tovec(self.scfg); v=v/norm(v)
            ei=sr._opq_vals[1]
            err=abs(e_true[0]-ei)/(abs(e_true[0])+abs(ei))
            print 'E = %s (%s), Error = %.4f%%'%(ei/self.nsite,e_true[0]/self.nsite,err*100)
            el.append(err)
        #    if k>50:optimizer.momentum=0.8
            if k>2000: break
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
    t=SRTest(nsite=4,periodic=True,model='AFH2D')
    t.test_sr(fakevmc=False)
    #show_err_sr(nsite=4)
