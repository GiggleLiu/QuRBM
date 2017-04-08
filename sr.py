'''
Stochestic Reconfiguration.
'''

from numpy import *
from scipy.linalg import pinv,inv,norm
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
    reg_method,reg_var=reg_params
    lambda0,b=reg_var.get('lambda0',100),reg_var.get('b',0.9)
    nb=rbm.nhid/rbm.group.ng
    info={}
    info['opl']=[]
    for p in xrange(niter):
        print '#'*20+' ITERATION %s '%p+'#'*20
        ops=handler.measure(q,rbm,tol=lambda0*0.2*b**p); info['opl'].append(ops)
        OPW,OH,OPW2,OPWH=ops
        S=OPW2-OPW[:,newaxis].conj()*OPW
        F=OPWH-OPW.conj()*OH
        #regularize S matrix to get Sinv.
        if reg_method=='delta':
            lamb=1e-4 #max(lambda0*b**p,1e-4)
            fill_diagonal(S,S.diagonal()+lamb)
            #fill_diagonal(S,S.diagonal()*(1+lamb))
            Sinv=inv(S)
            #Sinv/=norm(Sinv)
        elif reg_method=='pinv':
            Sinv=pinv(S)
        else:
            raise ValueError()
        g=gamma if not hasattr(gamma,'__call__') else gamma(p)
        ds=g*Sinv.dot(F)
        rbm.a-=ds[:rbm.nin]
        rbm.b-=ds[rbm.nin:rbm.nin+nb]
        rbm.W-=ds[rbm.nin+nb:].reshape(rbm.W.shape)
    return rbm,info
