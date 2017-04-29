from numpy import *
from abc import ABCMeta, abstractmethod
import pdb

__all__=['LinOp','PartialW','c_sandwich','OpQueue']

class LinOp(object):
    '''A new linear operator prototype.'''
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def _sandwich(self,config,state,runtime,**kwargs):
        '''
        Exceptation value like <config|self|state>.

        Parameters:
            :config: ndarray, configuration(s).
            :state: <RBM>,
            :runtime: dict, runtime variables to boost caculation.

        Return:
            ndarray, expectation value of this operator.
        '''
        pass

class PartialW(LinOp):
    '''Partial weight operator.'''
    def _sandwich(self,cgen,**kwargs):
        config=cgen.config
        theta=cgen.theta
        state=cgen.state
        partialS=[]
        if state.var_mask[0]:
            #get partial ai
            partialS.append(sum(state.group.apply_all(config),axis=0))
        if state.var_mask[1]:
            #get partial bj
            partialS.append(tanh(theta).reshape([state.group.ng,len(state.b)]).sum(axis=0))
        if state.var_mask[2]:
            #get partial Wij
            config_g=state.group.apply_all(config)
            partialS.append(sum(config_g[:,:,newaxis]*tanh(theta).reshape([state.group.ng,1,state.W.shape[1]]),axis=0).ravel())
            #partialS.append(sum(config_g[:,:,newaxis]*tanh(theta).reshape([state.group.ng,1,state.W.shape[1]]),axis=0).ravel())
        #partialS.append((config[:,newaxis]*tanh(theta)).ravel())
        return concatenate(partialS)

class OpQueue(LinOp):
    '''
    Queue of linear operators with dependency.

    Attributes:
        :op_base: tuple, linear operators.
        :op_derive: tuple, functions that decide derived operators (used in local measurements).
    '''
    def __init__(self,op_base,op_derive):
        self.op_base,self.op_derive=op_base,op_derive

    @property
    def nop(self):
        return len(self.op_base)+len(self.op_derive)

    def _sandwich(self,*args,**kwargs):
        valb,vald=[],[]
        for op in self.op_base:
            valb.append(c_sandwich(op,*args,**kwargs))
        for op in self.op_derive:
            vald.append(op(*valb))
        return valb+vald

class sx(LinOp):
    '''sigma_x'''
    def __init__(self,i):
        self.i=i

    def _sandwich(self,cgen,**kwargs):
        config=cgen.config
        state=cgen.state
        nc=copy(config)
        nc[self.i]*=-1
        return state.get_weight(nc)/state.get_weight(config,theta=cgen.theta)
        

def c_sandwich(op,cgen):
    '''
    Evaluate <config|op|state>

    Parameters:
        :op: <LinOp>,
        :config: 1darray, single state configuration.

    Return:
        number,
    '''
    return op._sandwich(cgen=cgen)
