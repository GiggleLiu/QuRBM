'''
Restricted Boltzmann Machine.
'''

from numpy import *
import numbers,pdb
from scipy.special import expit

from sstate import SparseState
from group import NoGroup
from utils import fh

__all__=['RBM','random_rbm']

class RBM(object):
    '''
    Restricted Boltzmann Machine class.

    Attributes:
        :a,b: 1darray, the bias
        :W: 2darray, the weights
        :group: Group, translation group.

        :nin,nhid: int, number of input and hidden layer, (nin,nh) = shape(W)
    '''
    def __init__(self,a,b,W,group=NoGroup(),var_mask=[True,True,True],input_node_type='linear',hidden_node_type='linear',array_order='A'):
        if array_order=='C':
            self.a,self.b,self.W=ascontiguousarray(a),ascontiguousarray(b),ascontiguousarray(W)
        elif array_order=='F':
            self.a,self.b,self.W=asfortranarray(a),asfortranarray(b),asfortranarray(W)
        else:
            self.a,self.b,self.W=asarray(a),asarray(b),asarray(W)
        self.group=group
        self.var_mask=var_mask
        self.input_node_type,self.hidden_node_type=input_node_type,hidden_node_type
        if not len(self.a)*len(self.b)==prod(self.W.shape):raise ValueError()

    def __rmul__(self,target):
        if isinstance(target,SparseState):
            res=0.
            for w,config in target:
                res=res+w*self.get_weight(config)
            return res
        elif isinstance(target,ndarray) and target.ndim==1:
            return self.get_weight(target)
        else:
            raise TypeError()

    def __mul__(self,target):
        if isinstance(target,(SparseState,ndarray)):
            return self.__rmul__(target).conj()
        else:
            raise TypeError()

    def __str__(self):
        return '<RBM>\na = %s\nb = %s\nW = %s\nGroup = %s'%(self.a,self.b,self.W,self.group)

    def __repr__(self):
        return '<RBM> in[%s] hid[%s x %s]'%(self.nin,self.group.ng,self.W.shape[1])

    @property
    def nhid(self): return len(self.b)*self.group.ng

    @property
    def nin(self): return len(self.a)

    @property
    def weight_dtype(self):
        return self.W.dtype

    def get_W_nogroup(self):
        '''Get the group expanded W.'''
        return self.group.unfold_W(self.W)

    def get_b_nogroup(self):
        '''Get the group expanded b.'''
        return concatenate([self.b]*self.group.ng)

    def get_a_nogroup(self):
        '''Get the group expanded a.'''
        return self.group.unfold_a(self.a)

    def feed_input(self,v):
        '''
        Feed visible inputs, and get output in hidden layers.

        Parameters:
            :v: 1d array, input vector.

        Return:
            1darray, raw output in hidden nodes.
        '''
        res=v.dot(self.get_W_nogroup())+self.get_b_nogroup()
        if self.hidden_node_type=='binary':
            return expit(res)
        else:
            return res

    def feed_hidden(self,h):
        '''
        Feed hidden inputs, and reconstruct visible layers.

        Parameters:
            :h: 1d array, input vector.

        Return:
            1darray, raw output in input nodes.
        '''
        if h.ndim>1:
            res=self.get_W_nogroup().dot(h.T).T+self.get_a_nogroup()
        else:
            res=self.get_W_nogroup().dot(h)+self.get_a_nogroup()
        if self.input_node_type=='binary':
            return expit(res)
        else:
            return res

    def tovec(self,spaceconfig):  #poor designed interface.
        '''
        Get the state vector.

        \Psi(s,W)=\sum_{\{hi\}} e^{\sum_j a_j\sigma_j^z+\sum_i b_ih_i +\sum_{ij}W_{ij}h_i\sigma_j}
        '''
        return self.get_weight(config=1-2*spaceconfig.ind2config(arange(spaceconfig.hndim)))

    def get_weight(self,config,theta=None):
        '''
        Get the weight for specific configuration.

        Parameters:
            :config: 1darray,
            :theta: 1darray/None, table of hidden layer output: b+v.dot(W), intended to boost operation.

        Return:
            number,
        '''
        group=self.group
        if theta is None: theta=self.feed_input(config)
        return exp(sum([group.apply(asarray(config),ig).dot(self.a) for ig in xrange(group.ng)],axis=0))*prod(fh(theta),axis=-1)

    def dump_arr(self):
        '''Dump values to an array.'''
        return concatenate([x for i,x in enumerate([self.a,self.b,self.W.ravel()]) if self.var_mask[i]])

    def load_arr(self,v):
        '''Load data from an array.'''
        nb=self.nhid/self.group.ng
        nin=self.nin
        offset=0
        if self.var_mask[0]:
            self.a[...]=v[offset:nin]
            offset+=nin
        if self.var_mask[1]:
            self.b[...]=v[offset:offset+nb]
            offset+=nb
        if self.var_mask[2]:
            self.W[...]=v[offset:].reshape([nin,nb])

def random_rbm(nin,nhid,group=NoGroup(),dtype='complex128',**kwargs):
    '''Get a random Restricted Boltzmann Machine'''
    if nhid%group.ng!=0: raise ValueError()
    nb=nhid/group.ng
    #data=(random.random(nin+nhid+nin*nhid/group.ng)-0.5)/2**nhid+1j*random.random(nin+nhid+nin*nhid/group.ng)-0.5j
    if dtype=='complex128':
        data=(random.random(nin+nb+nin*nb)-0.5)+1j*random.random(nin+nb+nin*nb)-0.5j
        data*=1e-2/abs(data)
    elif dtype=='float64':
        data=(random.random(nin+nb+nin*nb)-0.5)*0.01
    else:
        raise ValueError('unsupported dtype %s'%dtype)
    a=data[:nin]
    b=data[nin:nin+nb]
    W=data[nin+nb:].reshape([nin,nb])
    return RBM(a,b,W,group=group,**kwargs)
