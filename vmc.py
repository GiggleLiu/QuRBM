'''
Variational Monte Carlo Kernel.
'''

from numpy import *
from profilehooks import profile
import pdb

from linop import c_sandwich,OpQueue
from binner import Bin

__all__=['VMC']

class VMC(object):
    '''
    Variational Monte Carlo Engine.
    '''
    def __init__(self,cgen,nbath,nsample,nmeasure,nbin=50,sampling_method='metropolis',iprint=1):
        self.nbath,self.nsample=nbath,nsample
        self.cgen=cgen
        self.sampling_method=sampling_method
        self.nmeasure=nmeasure
        self.nbin=nbin
        self.iprint=iprint

    def accept(self,pratio,method='metropolis'):
        '''
        Decide whether accept or reject a move.

        Parameters:
            :pratio: float, the ratio of (distribution probability/transfer probability) between two configurations.

        Return:
            bool, accept if True.
        '''
        if self.sampling_method=='auto':
            method='heat-bath' if self.status=='WARM_UP' else 'metropolis'
        if self.sampling_method=='metropolis':
            A=pratio
        elif self.sampling_method=='heat-bath':
            A=pratio/(1+pratio)
        return random.random()<A

    def measure(self,op,state,tol=0):
        '''
        Measure an operator.

        Parameters:
            :op: <LinOp>, a linear operator instance.
            :state: <RBM>/..., a state ansaz
            :tol: float, tolerence.

        Return:
            number,
        '''
        nmeasure,nstat=self.nmeasure,int(ceil(1.*self.nsample/self.nbin))
        bins=[Bin() for i in xrange(op.nop if isinstance(op,OpQueue) else 1)]
        ol=[]  #local operator values
        o=None
        self.cgen.set_state(state)
        n_accepted=0
        nprint=10

        for i in xrange(self.nbath+self.nsample):
            #generate new config
            flips,pratio=self.cgen.fire()
            if self.accept(pratio):
                self.cgen.confirm(flips); n_accepted+=1
                o=None
            else:
                self.cgen.reject(flips)
            if i>=self.nbath:
                if i%nmeasure==0:
                    o=c_sandwich(op,cgen=self.cgen) if o is None else o
                    ol.append(o)
            isample=i-self.nbath
            if isample%nstat==nstat-1:
                do_print=(isample/nstat)%nprint==nprint-1
                if do_print: print '%-10s Accept rate: %.3f'%(i+1,n_accepted*1./nstat)
                n_accepted=0
                if len(ol)>0:
                    if isinstance(op,OpQueue):
                        for i,oli in enumerate(zip(*ol)):
                            bins[i].push(oli)
                            if do_print: bins[i].print_stat()
                    else:
                        bins[0].push(ol)
                        if do_print: bins[0].print_stat()

                    #accurate results obtained.
                    if len(bins)>100 and all(std_bin<tol):
                        break
                    ol=[]  #local operator values

        if isinstance(op,OpQueue):
            return [b.mean() for b in bins]
        else:
            return bins[0].mean()
