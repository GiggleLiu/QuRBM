from numpy import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
from scipy.sparse.linalg import LinearOperator
from scipy.linalg import kron
from matplotlib.pyplot import *
import sys,pdb,time
from os import path
sys.path.insert(0,'../')

from utils import *

def test_log2cosh():
    x=array([-13,-1,0,1,13])+(1j*random.random()-0.5j)*1000
    assert_allclose(exp(log(2*cosh(x))),exp(log2cosh(x)),atol=1e-8)
    assert_allclose(exp(log(2*sinh(x))),exp(log2sinh(x)),atol=1e-8)

if __name__=='__main__':
    test_log2cosh()
