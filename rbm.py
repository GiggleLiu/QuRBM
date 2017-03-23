'''
Restricted Boltzmann Machine.
'''

from numpy import *

__all__=['RBM','random_rbm']

class RBM(object):
    '''
    Restricted Boltzmann Machine class.

        S = [[x b b b],   #what is x here?
             [a W W W],
             [a W W W]]

    Attributes:
        :S: 2darray, data matrix.
        :a,b: 1darray, the bias
        :W: 2darray, the weights
        :nin,nhid: int, number of input and hidden layer, (nin,nh) = shape(W)
    '''
    def __init__(self,S):
        self.S=asarray(S)

    def _pack_input(self,config):
        ''''add 1 to the head of config if needed.'''
        return config if config.shape[-1]==self.S.shape[0] else insert(config,0,1,axis=-1)

    def _pack_hidden(self,config):
        ''''add 1 to the head of config if needed.'''
        return config if config.shape[-1]==self.S.shape[1] else insert(config,0,1,axis=-1)

    @property
    def nhid(self): return self.S.shape[1]-1

    @property
    def nin(self): return self.S.shape[0]-1

    @property
    def a(self): return self.S[1:,0]

    @property
    def b(self): return self.S[0,1:]

    @property
    def W(self): return self.S[1:,1:]

    def feed_visible(self,v,return_prob=False):
        '''
        Feed visible inputs, and get output in hidden layers.

        Parameters:
            :v: 1d array, input vector.
            :return_prob: bool, 

        Return:
            uint8, the output. Return (ouput, probability) pair instead if return_prob is True.
        '''
        v=self._pack_input(v)
        h=expit(v.dot(self.S))
        binary=(random.random(h.shape)<h).view('uint8')
        if return_prob:
            return binary,h
        else:
            return binary

    def feed_hidden(self,h,return_prob=False):
        '''
        Feed hidden inputs, and reconstruct visible layers.

        Parameters:
            :h: 1d array, input vector.
            :return_prob: bool, 

        Return:
            uint8, the output. Return (ouput, probability) pair instead if return_prob is True.
        '''
        h=self._pack_hidden(h)
        v=expit(h.dot(self.S.T))
        binary=(random.random(v.shape)<v).view('uint8')
        if return_prob:
            return binary,v
        else:
            return binary

    def tovec(self,spaceconfig):
        '''
        Get the state vector.

        \Psi(s,W)=\sum_{\{hi\}} e^{\sum_j a_j\sigma_j^z+\sum_i b_ih_i +\sum_{ij}W_{ij}h_i\sigma_j}
        '''
        return self.get_weight(config=spaceconfig.ind2config(arange(spaceconfig.hndim)))

    def get_weight(self,config):
        '''
        Get the weight for specific configuration.

        Parameters:
            :config: 1darray,
        '''
        config=self._pack_input(config)
        return exp(config.dot(self.S[:,0]))*prod(2*cosh(config.dot(self.S[:,1:])),axis=-1)

    def get_weight_ratio(self,config1,config2):
        pass

def random_rbm(nin,nhid):
    '''Get a random Restricted Boltzmann Machine'''
    S=random.random([nin+1,nhid+1])
    S[:,0]=0
    S[0,:]=0
    return RBM(S)