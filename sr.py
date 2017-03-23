'''
Stochestic Reconfiguration.
'''

from numpy import *
from scipy.linalg import pinv
from linop import PartialW,PartialW2,HPartialcW,

def SR(H,rbm,handler,gamma=0.1,reg_params=('delta',{})):
    '''
    Stochestic Reconfiguration.

    Attributes:
        :H: LinOp, Hamiltonian.
        :rbm: <RBM>, the state.
        :handler: <VMC>/..., the object with @measure(op) method.
        :gamma: float/func, the update ratio, or as a function of interation p.
        :reg_params: (str,dict), tuple of (method name, parameter dict) for regularization of S matrix. Methods are

            * 'delta' -> S_{kk'}^{reg} = S_{kk'} + \lambda(p) \delta_{kk'} S_{kk}, \lambda(p)=max(\lambda_0 b^p,\lambda_{min}), with p the # of iteration.
            * 'pinv'  -> use pseudo inverse instead.
    '''
    for p in xrange(niter):
        S=handler.measure(PartialW2,rbm)-conj(handler.measure(PartialW,rbm))[:,newaxis]*handler.measure(PartialW,rbm)
        F=handler.measure(HPartialcW,rbm)-handler.measure(H,rbm)*handler.measure(PartialW,rbm).conj()
        #regularize S matrix to get Sinv.
        reg_method,reg_var=reg_params
        if reg_method=='delta':
            lamb=max(reg_var.get('lambda0',100)*ref_var.get('b',0.9)**p,1e-4)
            fill_diagonal(S,S.diagonal()+lamb)
            Sinv=inv(S)
        elif reg_method=='pinv':
            Sinv=pinv(S)
        else:
            raise ValueError()
        g=gamma if not hasattr(gamma,'__call__') else gamma(p)
        rbm.W-=g*Sinv.dot(F).reshape(rbm.W.shape)
    return rbm
