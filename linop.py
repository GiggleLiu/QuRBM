from numpy import *
from abc import ABCMeta, abstractmethod

__all__=['LinOp']

class LinOp(object):
    '''A new linear operator prototype.'''
    __metaclass__ = ABCMeta
    
    opt_l=True   #lmatmul is prefered.

    @abstractmethod
    def rmatmul(self,target):
        '''
        matmul a vector at the right side.
        
        Parameters:
            :target: 1darray/<SparseState>, the input state.

        Return:
            <SparseState>, the output state.
        '''
        pass

    @abstractmethod
    def lmatmul(self,target):
        '''matmul a vector at the left side.
        
        Parameters:
            :target: 1darray/<SparseState>, the input state.

        Return:
            <SparseState>, the output state.
        '''
        pass

def sandwich(ket,op,bra):
    '''Evaluate <ket|op|bra>'''
    if op.opt_l:
        ket*op.lmatmul(bra)
