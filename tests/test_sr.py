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
from group import TIGroup

from test_vmc import analyse_sampling

#random.seed(2)

def test_sr_fake():
    nsite=6
    h=HeisenbergH(nsite=nsite)
    scfg=SpinSpaceConfig([nsite,2])
    b=0.9

    #generate a random rbm and the corresponding vector v
    rbm=random_rbm(nin=nsite,nhid=nsite*2)

    fv=FakeVMC()
    v_true=eigh(fv.get_H(nsite))[1][:,0]
    el=[]
    for k in xrange(100):
        print 'Running %s-th batch.'%k
        rbm=SR(h,rbm,handler=fv,niter=1,gamma=0.2,reg_params=('delta',{'b':b,'lambda0':100*b**(5*k)}))
        v=rbm.tovec(scfg)
        v=v/norm(v)
        err=1-abs(v.conj().dot(v_true))
        print 'Error = %.4f%%'%(err*100)
        el.append(err)
    pdb.set_trace()
    savetxt('err0.dat',el)
    assert_(err<0.01)

def test_sr():
    nsite=5
    h=HeisenbergH(nsite=nsite)
    scfg=SpinSpaceConfig([nsite,2])
    b=0.8

    #generate a random rbm and the corresponding vector v
    #rbm=random_rbm(nin=nsite,nhid=nsite)
    rbm=random_rbm(nin=nsite,nhid=nsite,group=TIGroup(nsite))

    #vmc config
    core=RBMCore()
    vmc=VMC(core,nbath=200,nsample=10000,nmeasure=4,sampling_method='metropolis')

    fv=FakeVMC()
    v_true=eigh(fv.get_H(nsite))[1][:,0]

    el=[]
    for k in xrange(100):
        print 'Running %s-th batch.'%k
        rbm=SR(h,rbm,handler=vmc,niter=1,gamma=0.2,reg_params=('delta',{'lambda0':100*b**(5*k),'b':b}))
        v=rbm.tovec(scfg); v=v/norm(v)
        err=1-abs(v.conj().dot(v_true))
        print 'Error = %.4f%%'%(err*100)
        el.append(err)
    savetxt('err.dat',el)
    assert_(err<0.05)

def show_err_sr():
    from matplotlib.pyplot import plot,ion
    ion()
    fig=figure(figsize=(5,4))
    el=loadtxt('err.dat')
    el0=loadtxt('err0.dat')
    plot(el0,lw=2)
    plot(el,lw=2)
    xlabel('iteration',fontsize=16)
    ylabel(r'$1-\|\left\langle\psi|\tilde{\psi}\right\rangle\|_2$',fontsize=16)
    ylim(1e-8,1)
    yscale('log')
    legend(['Exact','VMC'])
    tight_layout()
    pdb.set_trace()
    savefig('err.pdf')

if __name__=='__main__':
    test_sr_fake()
    test_sr()
    #show_err_sr()
