cimport cython
cimport numpy as np
import time

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
def pop_nogroup(np.ndarray[int_t,ndim=1,mode='c'] config not None,
        np.ndarray[int_t,ndim=1,mode='c'] flips,
        np.ndarray[complex_t,ndim=2,mode='c'] W not None,
        np.ndarray[complex_t,ndim=1,mode='c'] a not None,
        np.ndarray[complex_t,ndim=1,mode='c'] theta not None):

    cdef int iflip,ci,i
    cdef np.ndarray[complex_t,ndim=1,mode='c'] _theta=theta.copy()
    cdef complex_t pratio=0

    for iflip in flips:
        ci=config[iflip]
        _theta-=2*ci*W[iflip]
        pratio-=2*ci*a[iflip]
    for i in range(theta.shape[0]):
        pratio+=lncoshc(_theta[i])-lncoshc(theta[i])
    pratio=exp(pratio)
    return _theta,pratio

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def pop1D(np.ndarray[int_t,ndim=1,mode='c'] config not None,
        np.ndarray[int_t,ndim=1,mode='c'] flips,
        np.ndarray[complex_t,ndim=2,mode='c'] W not None,
        np.ndarray[complex_t,ndim=1,mode='c'] a not None,
        np.ndarray[complex_t,ndim=1,mode='c'] theta not None,int ng):
    cdef int nv=W.shape[0]
    cdef int nf=W.shape[1]  #number of features.

    cdef int ig,iflip,ci,i
    cdef np.ndarray[complex_t,ndim=1,mode='c'] _theta=theta.copy()
    cdef complex_t pratio=0
    cdef complex_t sa=a.sum()

    for iflip in flips:
        ci=config[iflip]
        for ig in range(ng):
            _theta[ig*nf:(ig+1)*nf]-=2*ci*W[(iflip+ig)%nv]
        if ng==config.shape[0]:
            pratio-=2*ci*sa
        elif ng==1:
            pratio-=2*ci*a[iflip]
        else:
            raise ValueError
    for i in range(theta.shape[0]):
        pratio+=lncoshc(_theta[i])-lncoshc(theta[i])
    pratio=exp(pratio)
    return _theta,pratio

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def pop2D(np.ndarray[int_t,ndim=1,mode='c'] config not None,
        np.ndarray[int_t,ndim=1,mode='c'] flips,
        np.ndarray[complex_t,ndim=2,mode='c'] W not None,
        np.ndarray[complex_t,ndim=1,mode='c'] a not None,
        np.ndarray[complex_t,ndim=1,mode='c'] theta not None,
        np.ndarray[int_t,ndim=1,mode='c'] ngs not None):
    cdef int nv=W.shape[0]
    cdef int nf=W.shape[1]  #number of features.

    cdef int ig,iflip,ci,i,ig1,ig2,ng1=ngs[0],ng2=ngs[1],ng=ng1*ng2
    cdef np.ndarray[complex_t,ndim=1,mode='c'] _theta=theta.copy()
    cdef complex_t pratio=0
    cdef complex_t sa=a.sum()

    for iflip in flips:
        ci=config[iflip]
        for ig1 in range(ng1):
            for ig2 in range(ng2):
                ig=ig1*ng2+ig2
                iflip1,iflip2=iflip/ng2,iflip%ng2
                _theta[ig*nf:(ig+1)*nf]-=2*ci*W[((iflip1+ig1)%ng1)*ng2+(iflip2+ig2)%ng2]
        if ng==config.shape[0]:
            pratio-=2*ci*sa
        elif ng==1:
            pratio-=2*ci*a[iflip]
        else:
            raise ValueError
    for i in range(theta.shape[0]):
        pratio+=lncoshc(_theta[i])-lncoshc(theta[i])
    pratio=exp(pratio)
    return _theta,pratio
