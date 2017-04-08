from numpy import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
from scipy.sparse.linalg import LinearOperator
from scipy.linalg import kron
from matplotlib.pyplot import *
import sys,pdb,time
from os import path
sys.path.insert(0,'../')

from binner import *

def test_bin():
    print 'Test Scalar.'
    a=[1,2,3]
    b=[3,2,1]
    bb=Bin()
    print var([],ddof=1)
    assert_(isnan(bb.var_unbinned()))
    assert_(isnan(bb.var()))
    print bb
    bb.push(a)
    bb.push(b)
    print bb
    bb.print_stat()
    assert_almost_equal(bb.var(),0)
    assert_almost_equal(bb.var_unbinned(),var(a+b,ddof=1))

    print 'Test matrix.'
    bb=Bin()
    I=eye(2)
    a=[I,2*I,3*I]
    b=[3*I,2*I,I]
    c=[3*I,5*I,I]
    bb.push(a)
    bb.push(b)
    bb.push(c)
    print bb
    bb.print_stat()
    assert_allclose(bb.var_unbinned(),var(a+b+c,axis=0,ddof=1))
    assert_allclose(bb.mean(),mean(a+b+c,axis=0))
    pdb.set_trace()

if __name__=='__main__':
    test_bin()
