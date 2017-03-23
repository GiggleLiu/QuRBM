'''
Stochestic Reconfiguration.
'''

from scipy.linalg import pinv

class SR(object):
    '''
    Engine of Stochestic Reconfiguration.

    Attributes:
        :O: LinOp, partial derivative operator.
        :H: LinOp, Hamiltonian.
        :gamma: float/func, the update ratio, or as a function of interation p.
        :reg_params: (str,dict), tuple of (method name, parameter dict) for regularization of S matrix. Methods are

            * 'delta' -> S_{kk'}^{reg} = S_{kk'} + \lambda(p) \delta_{kk'} S_{kk}, \lambda(p)=max(\lambda_0 b^p,\lambda_{min}), with p the # of iteration.
            * 'pinv'  -> use pseudo inverse instead.
    '''
    def __init__(self,O,H,gamma=0.1,reg_params=('delta',{})):
        self.O=O
        self.H=H
        self.gamma=gamma

        #private counter
        self._p=0

    def reset(self): self._p=0

    def update(self,params):
