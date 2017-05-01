'''
Restricted Boltzmann Machine.
'''

from numpy import *
import numbers,pdb
from scipy.special import expit

from sstate import SparseState
from group import NoGroup
from utils import fh

__all__=['DBM','random_dbm']

class DBM(object):
    '''
    Restricted Boltzmann Machine class.

    Attributes:
        :a: 1darray, the bias for input layer
        :b_L: list of 1darray, biases for hidden layers.
        :W_L: list of 2darray, the weights
        :group: Group, translation group.

        :nin,nhid: int, number of input and hidden layer, (nin,nh) = shape(W)
    '''
    def __init__(self,b_L,W_L,group=NoGroup(),var_mask=None,input_node_type='linear',hidden_node_type='linear'):
        self.b_L,self.W_L=b_L,W_L
        self.group=group
        if var_mask is None:
            var_mask=[True]*(len(b_L)+len(W_L))
        else:
            if len(var_mask)!=len(b_L)+len(W_L):raise ValueError('number of variable mask not match.')
        self.var_mask=var_mask
        self.input_node_type,self.hidden_node_type=input_node_type,hidden_node_type
        #check data
        for i in xrange(len(W_L)):
            w=W_L[i]
            bl=b_L[i]
            br=b_L[i+1]
            if w.shape!=(len(bl),len(br)):
                raise ValueError('Matrix-bias shape mismatch.')
        if not len(self.b_L)==len(self.W_L)+1: raise ValueError('# of layer weights and biases not match.')

    def __str__(self):
        return '<DBM>\n%s\n%s\nGroup = %s'%('\n'.join(['b(%s) %s'%(i,b) for i,b in enumerate(self.b_L)]),\
                '\n'.join(['W(%s,%s) %s'%(i,i+1,W) for i,W in enumerate(self.W_L)]),self.group)

    def __repr__(self):
        return '<DBM> in[%s] hid[%s]'%(self.nin,' x '.join([str(len(b)) for b in self.b_L[1:]]))

    @property
    def num_layers(self):return len(self.b_L)

    @property
    def nin(self): return len(self.b_L[0])

    @property
    def weight_dtype(self):
        return self.W[0].dtype

    def layer_dim(self,i):
        '''dimension of i-th layer.'''
        return len(self.b_L[i])

    def get_W0_nogroup(self):
        '''Get the group expanded W.'''
        return self.group.unfold_W(self.W_L[0])

    def get_a_nogroup(self):
        '''Get the group expanded a.'''
        return self.group.unfold_a(self.b_L[0])

    def feed_input(self,v):
        '''
        Feed visible inputs, and get output in hidden layers.

        Parameters:
            :v: 1d array, input vector.

        Return:
            1darray, raw output in hidden nodes.
        '''
        for W,b in zip(self.W_L,self.b_L[1:]):
            v=v.dot(W)+b
            if self.hidden_node_type=='binary':
                v=expit(v)
        return v

    def feed_hidden(self,h):
        '''
        Feed hidden inputs, and reconstruct visible layers.

        Parameters:
            :h: 1d array, input vector.

        Return:
            1darray, raw output in input nodes.
        '''
        for W,b in zip(self.W_L,self.b_L):
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
        return concatenate([b for b,mask in zip(self.b_L,self.var_mask[:self.num_layers]) if mask]+\
                [W.ravel() for W,mask in zip(self.W_L,self.var_mask[self.num_layers:]) if mask])

    def load_arr(self,v):
        '''Load data from an array.'''
        offset=0
        for b,mask in zip(self.b_L,self.var_mask[:self.num_layers]):
            if mask:
                layer_size=len(b)
                b[:]=v[offset:offset+layer_size]
                offset+=layer_size
        for W,mask in zip(self.W_L,self.var_mask[self.num_layers:]):
            if mask:
                layer_size=W.shape[0]*W.shape[1]
                W[...]=v[offset:offset+layer_size].reshape(W.shape)
                offset+=layer_size

def random_dbm(dims,group=NoGroup(),dtype='complex128',magnitude=2e-2,**kwargs):
    '''Get a random Restricted Boltzmann Machine'''
    num_layers=len(dims)
    b_L,W_L=[],[]
    if dtype=='complex128':
        rng=lambda shape:random.uniform(-magnitude,magnitude,shape)+1j*random.uniform(-magnitude,magnitude,shape)
    elif dtype=='float64':
        rng=lambda shape:random.uniform(-magnitude,magnitude,shape)
    else:
        raise ValueError('unsupported dtype %s'%dtype)

    for i in xrange(num_layers):
        b_L.append(rng(dims[i]))
        if i!=0:
            W_L.append(rng((dims[i-1],dims[i])))
    return DBM(b_L=b_L,W_L=W_L,group=group,**kwargs)
