from numpy import *
from abc import ABCMeta, abstractmethod
import pdb

__all__=['LinOp','PartialW','c_sandwich']

class LinOp(object):
    '''A new linear operator prototype.'''
    __metaclass__ = ABCMeta
    
    opt_lmul=False   #lmatmul is prefered.

    def rmatmul(self,target):
        '''
        matmul a vector at the right side.
        
        Parameters:
            :target: 1darray/<SparseState>, the input state.

        Return:
            <SparseState>, the output state.
        '''
        pass

    def lmatmul(self,target):
        '''matmul a vector at the left side.
        
        Parameters:
            :target: 1darray/<SparseState>, the input state.

        Return:
            <SparseState>, the output state.
        '''
        pass

class PartialW(LinOp):
    '''Partial weight operator.'''
    def _sandwich(self,config,runtime,**kwargs):
        #config=1-2*config
        theta=runtime['theta']
        partialS=zeros((len(config)+1,len(theta)+1),dtype=theta.dtype)
        #get partial ai
        partialS[1:,0]=config
        #get partial bj,Wij
        partialS[:,1:]=append([1],config)[:,newaxis]*tanh(theta)
        return partialS

#class HPartialWc(LinOp):
#    def __init__(self,H):
#        self.H=H
#
#    def _sandwich(self,config,runtime,**kwargs):
#        if not (runtime.haskey('partialW_loc') and runtime.haskey('H_loc')):
#            raise ValueError('We need runtime partialW_loc and H_loc to calculate')
#        theta=runtime['theta']

#def sandwich(op,ket,state,runtime={}):
#    '''Evaluate <ket|op|bra>'''
#    if hasattr(op,'_sandwich'):
#        return op._sandwich(ket,runtime=runtime,state=state)
#    if op.opt_lmul:
#        ket*op.lmatmul(bra)/(ket*bra)
#    else:
#        op.rmatmul(ket)*bra/(ket*bra)

def c_sandwich(op,config,state,runtime={}):
    '''
    Evaluate <config|op|state>

    Parameters:
        :op: <LinOp>,
        :config: 1darray, single state configuration.
        :runtime: dict, runtime variables.

    Return:
        number,
    '''
    if hasattr(op,'_sandwich'):
        return op._sandwich(config,runtime=runtime,state=state)
    if op.opt_lmul:
        return ket*op.lmatmul(state)/state.get_weight(config,theta=runtime.get('theta'))
    else:
        return op.rmatmul(config)*state/state.get_weight(config,theta=runtime.get('theta'))
