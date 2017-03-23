'''Sparse State Representation.'''

from numpy import *

__all__=['SparseState','visualize_sstate','vec2sstate','soverlap']

def _compact_form(ws,configs):
    '''Merge duplicate configs.'''
    ws,configs=asarray(ws),asarray(configs)
    order=lexsort(configs.T)
    configs=configs[order]
    ws=ws[order]
    row_mask = append([True],any(diff(configs,axis=0),1))  #unique rows
    ws=asarray([a.sum() for a in split(ws,where(row_mask)[0][1:])])
    configs=configs[row_mask]
    return ws,configs

class SparseState(object):
    '''sparse representation of state.'''
    def __init__(self,ws,configs):
        assert(len(ws)==len(configs))
        self.ws,self.configs=_compact_form(ws,configs)

    def __str__(self):
        return '\n'.join(['%.2f: %s'%(wi,list(ci)) for wi,ci in zip(self.ws,self.configs)])

    def __mul__(self,target):
        if isinstance(target,SparseState):
            return soverlap(self,target)
        elif isinstance(target,numbers.Number):
            return SparseState(self.ws*target,self.configs)
        else:
            raise TypeError()

    def __rmul__(self,target):
        if isinstance(target,SparseState):
            return target.__mul__(self)
        elif isinstance(target,numbers.Number):
            return SparseState(self.ws*target,self.configs)
        else:
            raise TypeError()
        #if hasattr(target,'opt_l'):
        #    return target.rmul

    def tovec(self,spaceconfig):
        '''To vector.'''
        vec=zeros(spaceconfig.hndim,dtype=self.ws.dtype)
        inds=spaceconfig.config2ind(self.configs)
        add.at(vec,inds,self.ws)
        return vec

    def tobra(self):
        '''hermitian conjugate.'''
        return SparseState(self.ws.conj(),self.configs)

def visualize_sstate(sstate,**kwargs):
    '''Visualize SparseState.'''
    from matplotlib.pyplot import pcolormesh
    pcolormesh(sstate.configs*abs(sstate.ws[:,newaxis]),**kwargs)

def vec2sstate(vec,spaceconfig,tol=1e-8):
    '''Transform a vector into SparseState'''
    mask=abs(vec)>1e-8
    inds=where(mask)
    configs=spaceconfig.ind2config(inds[0])
    return SparseState(vec[mask],configs)

def soverlap(bra,ket):
    '''
    Sparse version overlap of bra and ket.
    '''
    j0=0
    res=0.
    for w1,config1 in zip(bra.ws,bra.configs):
        w1=conj(w1)
        for j,(w2,config2) in enumerate(zip(ket.ws,ket.configs[j0:])):
            if allclose(config1,config2):
                res=res+w1*w2
                j0=j+1
    return res
