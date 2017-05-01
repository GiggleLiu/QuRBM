'''
Stochestic Reconfiguration.
'''

from numpy import *
from scipy.linalg import pinv,inv,norm,eigh
import pdb

from linop import PartialW,OpQueue

__all__=['SR','SD']

class SR(object):
    '''
    Stochestic Reconfiguration optimization problem.

    Attributes:
        :H: LinOp, Hamiltonian.
        :rbm: <RBM>, the state.
        :handler: <VMC>/..., the object with @measure(op) method.
        :niter: int, number of iteration.
        :reg_params: (str,dict), tuple of (method name, parameter dict) for regularization of S matrix. Methods are

            * 'carleo' -> S_{kk'}^{reg} = S_{kk'} + \lambda(p) \delta_{kk'} S_{kk}, \lambda(p)=max(\lambda_0 b^p,\lambda_{min}), with p the # of iteration.
            * 'delta' -> S_{kk'}^{reg} = S_{kk'} + \lambda_0 \delta_{kk'}
            * 'trunc' -> carleo's approach S_{kk}*(1+lambda0), diacarding near singular values (s/s_max < eps_trunc).
            * 'pinv'  -> use pseudo inverse instead.
            * 'identity' -> equivalence to SD.
    '''
    def __init__(self,H,rbm,handler,reg_params=('delta',{})):
        self.H=H
        self.rbm=rbm
        self.handler=handler
        self.reg_params=reg_params
        self._opq=OpQueue((PartialW(),H),(lambda a,b:a[:,newaxis].conj()*a,lambda a,b:a.conj()*b))
        self._opq_vals=None
        self._counter=0

    def compute_gradient(self,v):
        reg_method,reg_var=self.reg_params
        #update RBM
        self.rbm.load_arr(v)

        #perform measurements
        self._opq_vals=self.handler.measure(self._opq,self.rbm,tol=0)
        OPW,OH,OPW2,OPWH=self._opq_vals
        S=OPW2-OPW[:,newaxis].conj()*OPW
        F=OPWH-OPW.conj()*OH

        #regularize S matrix to get Sinv.
        if reg_method=='carleo':
            lambda0,b=reg_var.get('lambda0',100),reg_var.get('b',0.9)
            lamb=max(lambda0*b**self._counter,1e-4)
            fill_diagonal(S,S.diagonal()*(1+lamb)+1e-8)
            Sinv=inv(S)
        elif reg_method=='delta':
            lambda0=reg_var.get('lambda0',1e-4)
            fill_diagonal(S,S.diagonal()+lambda0)
            Sinv=inv(S)
        elif reg_method=='trunc':
            eps_trunc,lambda0=reg_var.get('eps_trunc',1e-3),reg_var.get('lambda0',0.2)
            fill_diagonal(S,S.diagonal()*(1+lambda0))
            L,U=eigh(S)
            kpmask=L/L[-1]>eps_trunc
            U=U[:,kpmask]
            Sinv=(U/L[kpmask]).dot(U.T.conj())
        elif reg_method=='pinv':
            Sinv=pinv(S)
        elif reg_method=='identity':
            Sinv=identity(len(S))
        else:
            raise ValueError()
        self._counter+=1
        return Sinv.dot(F)

class SD(object):
    '''
    Steepest descend.

    Attributes:
        :H: LinOp, Hamiltonian.
        :rbm: <RBM>, the state.
        :handler: <VMC>/..., the object with @measure(op) method.
    '''
    def __init__(self,H,rbm,handler):
        self.H=H
        self.rbm=rbm
        self.handler=handler
        self._opq=OpQueue((PartialW(),H),(lambda a,b:a.conj()*b,))
        self._opq_vals=None
        self._counter=0

    def compute_gradient(self,v):
        handler,rbm,q=self.handler,self.rbm,self._opq
        #update RBM
        rbm.load_arr(v)

        #perform measurements
        self._opq_vals=handler.measure(q,rbm,tol=0)
        OPW,OH,OPWH=self._opq_vals; OH=OH.real
        F=OPWH-OPW.conj()*OH
        self._counter+=1
        return F
