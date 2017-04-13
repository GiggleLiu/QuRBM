from numpy import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
from scipy.sparse.linalg import LinearOperator
from scipy.linalg import kron
from matplotlib.pyplot import *
import sys,pdb,time
from os import path
sys.path.insert(0,'../')

from group import TIGroup

def test_group():
    print 'Testing 1D translational invariant group.'
    config=array([1,2,3])
    g=TIGroup(3)
    assert_allclose(g.apply(config,2),[2,3,1])
    assert_allclose(g.apply_all(config),[[1,2,3],[3,1,2],[2,3,1]])
    inds=arange(g.ng)
    assert_allclose(config[g.ind_apply(inds,ig=1)],g.apply(config,ig=1))
    assert_allclose(config,g.apply(g.apply(config,ig=1),-1))
    print 'Testing 2D translational invariant group.'
    config=array([[1,2,3],
            [4,5,6]]).ravel()
    g=TIGroup([2,3])
    assert_allclose(g.apply(config,4),[6,4,5,3,1,2])
    assert_allclose(g.apply(g.apply(config,4),-4),config)
    assert_allclose(g.apply_all(config),[[1,2,3,4,5,6],
        [3,1,2,6,4,5],
        [2,3,1,5,6,4],
        [4,5,6,1,2,3],
        [6,4,5,3,1,2],
        [5,6,4,2,3,1]])
    inds=arange(g.ng)
    assert_allclose(config[g.ind_apply(inds,ig=4)],g.apply(config,ig=4))

if __name__=='__main__':
    test_group()
