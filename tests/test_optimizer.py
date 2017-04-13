'''trying to solve M*v=b'''

from numpy import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
from scipy.sparse.linalg import LinearOperator
from scipy.linalg import kron,norm,inv
from matplotlib.pyplot import *
import sys,pdb,time
from os import path
sys.path.insert(0,'../')

from optimizer import *

random.seed(2)

def compute_gradient(M,v,b):
    return M.T.conj().dot((M.dot(v)-b))

def compute_distance(M,v,b):
    return norm(M.dot(v)-b)

def test_all():
    #initialize matrix and vectors
    M=random.random([100,100])+1j*random.random([100,100])
    M=(M+M.T.conj())/100
    v0=random.random(100)+1j*random.random(100)
    b=random.random(100)+1j*random.random(100)
    v_true=inv(M).dot(b)

    rms=RMSProp(rho=0.9,rate=0.1)
    default=DefaultOpt(rate=0.5)
    mk=MannKendall(rate=3,size=20)
    niter=500
    ion()
    opts=[rms,default]
    for opt in opts:
        print 'Optimizing using %s'%opt
        err=[]
        v=v0.copy()
        for i in xrange(niter):
            distance=compute_distance(M,v,b)**2
            g=compute_gradient(M,v,b)
            dv=opt(distance**2,g,i)
            v+=dv
            print 'Iter %s, Error = %s'%(i,distance**2 if not isinstance(opt,MannKendall) else distance)
            err.append(distance**2)
        plot(err)
    legend(['%s'%opt.__class__ for opt in opts])
    pdb.set_trace()

if __name__=='__main__':
    test_all()
