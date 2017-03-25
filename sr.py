'''
Stochestic Reconfiguration.
'''

from numpy import *
from scipy.linalg import pinv,inv
import pdb

from linop import PartialW,OpQueue

__all__=['SR']

def SR(H,rbm,handler,gamma=0.1,niter=200,reg_params=('delta',{})):
    '''
    Stochestic Reconfiguration.

    Attributes:
        :H: LinOp, Hamiltonian.
        :rbm: <RBM>, the state.
        :handler: <VMC>/..., the object with @measure(op) method.
        :gamma: float/func, the update ratio, or as a function of interation p.
        :niter: int, number of iteration.
        :reg_params: (str,dict), tuple of (method name, parameter dict) for regularization of S matrix. Methods are

            * 'delta' -> S_{kk'}^{reg} = S_{kk'} + \lambda(p) \delta_{kk'} S_{kk}, \lambda(p)=max(\lambda_0 b^p,\lambda_{min}), with p the # of iteration.
            * 'pinv'  -> use pseudo inverse instead.
    '''
    q=OpQueue((PartialW(),H),(lambda a,b:a.conj()[...,newaxis]*a,lambda a,b:b*a.conj()))
    for p in xrange(niter):
        OPW,OH,OPW2,OPWH=handler.measure(q,rbm)
        S=OPW2-OPW[:,newaxis].conj()*OPW
        F=OPWH-OPW.conj()*OH
        #regularize S matrix to get Sinv.
        reg_method,reg_var=reg_params
        if reg_method=='delta':
            lamb=max(reg_var.get('lambda0',100)*reg_var.get('b',0.9)**p,1e-4)
            fill_diagonal(S,S.diagonal()+lamb)
            Sinv=inv(S)
        elif reg_method=='pinv':
            Sinv=pinv(S)
        else:
            raise ValueError()
        g=gamma if not hasattr(gamma,'__call__') else gamma(p)
        ds=g*Sinv.dot(F)
        rbm.a-=ds[:rbm.nin]
        rbm.b-=ds[rbm.nin:rbm.nin+rbm.nhid]
        rbm.W-=ds[rbm.nin+rbm.nhid:].reshape(rbm.W.shape)
    return rbm
