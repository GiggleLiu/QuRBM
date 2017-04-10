'''
Stochestic Reconfiguration.
'''

from numpy import *
from scipy.linalg import pinv,inv,norm,eigh
import pdb

from linop import PartialW,OpQueue
from optimizer import DefaultOpt

__all__=['SR','SD']

def SR(H,rbm,handler,niter=200,optimizer=DefaultOpt(0.1),reg_params=('delta',{})):
    '''
    Stochestic Reconfiguration.

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
    '''
    q=OpQueue((PartialW(),H),(lambda a,b:a[...,newaxis].conj()*a,lambda a,b:a.conj()*b))
    reg_method,reg_var=reg_params
    nb=rbm.nhid/rbm.group.ng
    info={}
    info['opl']=[]
    for p in xrange(niter):
        print '#'*20+' ITERATION %s '%p+'#'*20
        ops=handler.measure(q,rbm,tol=0); info['opl'].append(ops)
        OPW,OH,OPW2,OPWH=ops
        S=OPW2-OPW[:,newaxis].conj()*OPW
        F=OPWH-OPW.conj()*OH

        #regularize S matrix to get Sinv.
        if reg_method=='carleo':
            lambda0,b=reg_var.get('lambda0',100),reg_var.get('b',0.9)
            lamb=max(lambda0*b**p,1e-4)
            fill_diagonal(S,S.diagonal()*(1+lamb))
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
        else:
            raise ValueError()
        #g=gamma if not hasattr(gamma,'__call__') else gamma(p)
        #ds=g*Sinv.dot(F)
        ds=optimizer(1,Sinv.dot(F))  #decide the move according to the gradient
        rbm.a+=ds[:rbm.nin]
        rbm.b+=ds[rbm.nin:rbm.nin+nb]
        rbm.W+=ds[rbm.nin+nb:].reshape(rbm.W.shape)
        print 'Energy/site = %s'%(OH/rbm.nin)
    return rbm,info

def SD(H,rbm,handler,niter=200,optimizer=DefaultOpt(0.03)):
    '''
    Stochestic Reconfiguration.

    Attributes:
        :H: LinOp, Hamiltonian.
        :rbm: <RBM>, the state.
        :handler: <VMC>/..., the object with @measure(op) method.
        :niter: int, number of iteration.
        :optimizer: <Optimizer>/func, optimization engine.
    '''
    q=OpQueue((PartialW(),H),(lambda a,b:a.conj()*b,))
    nb=rbm.nhid/rbm.group.ng
    info={}
    info['opl']=[]
    for p in xrange(niter):
        print '#'*20+' ITERATION %s '%p+'#'*20
        ops=handler.measure(q,rbm,tol=0); info['opl'].append(ops)
        OPW,OH,OPWH=ops
        F=OPWH-OPW.conj()*OH

        ds=optimizer(1,F)  #decide the move according to the gradient
        rbm.a+=ds[:rbm.nin]
        rbm.b+=ds[rbm.nin:rbm.nin+nb]
        rbm.W+=ds[rbm.nin+nb:].reshape(rbm.W.shape)
        print 'Energy/site = %s'%(OH/rbm.nin)
    return rbm,info
