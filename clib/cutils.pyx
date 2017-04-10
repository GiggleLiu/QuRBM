cimport cython
cimport numpy as np
import numpy

ctypedef np.complex128_t complex_t
ctypedef np.int_t int_t

cdef extern from "cutils.cpp":
    double lncoshd(double x)
    complex_t lncoshc(complex_t x)
cdef extern from "cmath":
    complex_t exp(complex_t x)

def lncosh(complex_t x):
    '''
    ln(cosh(x)).
    '''
    cdef complex_t y=lncoshc(x)
    return y

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def pop(np.ndarray[int_t,ndim=1,mode='c'] config not None,
        np.ndarray[int_t,ndim=1,mode='c'] flips not None,
        np.ndarray[complex_t,ndim=2,mode='c'] W not None,
        np.ndarray[complex_t,ndim=1,mode='c'] a not None,
        np.ndarray[complex_t,ndim=1,mode='c'] theta not None,int ng):
    cdef int nv=W.shape[0]
    cdef int nf=W.shape[1]  #number of features.

    cdef int ig,iflip,ci
    cdef np.ndarray[complex_t,ndim=1,mode='c'] _theta=theta.copy()
    cdef complex_t pratio=0
    cdef complex_t th
    cdef complex_t _th

    for iflip in flips:
        ci=config[iflip]
        for ig in xrange(ng):
            _theta[ig*nf:(ig+1)*nf]-=2*ci*W[(iflip+ig)%nv]
        pratio-=2*config[iflip]*a[iflip]
    for th,_th in zip(theta,_theta):
        pratio+=lncoshc(_th)-lncoshc(th)
    pratio=exp(pratio)
    return _theta,pratio
