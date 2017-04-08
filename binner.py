'''
Binning staticstics.
'''

from numpy import *
import pdb

__all__=['Bin']

class Bin(object):
    '''
    class for Binning statistics.

    Attributes:
        :n: list, length of each block.
        :m: list, mean of each block.
        :sqm: list, square root mean.
    '''
    def __init__(self):
        self.n=[]
        self.m=[]
        self.sqm=[]

    def __str__(self):
        return 'Bin(%s)'%len(self.m)

    def __repr__(self):
        return self.__str__()

    @property
    def nbin(self):
        return len(self.m)

    def print_stat(self):
        '''
        Show statistics.
        '''
        print '> Binning statistics: \n  Autocorrelation Time: %.4f\n  Standard Error: %.4f'%(nan if self.nbin<2 else abs(self.t_auto()),nan if self.nbin<2 else abs(mean(sqrt(self.std_err()))))

    def push(self,vals):
        '''
        Push datas to bin.
        '''
        vals=asarray(vals)
        o_mean=vals.mean(axis=0)
        o_sq_mean=(vals**2).mean(axis=0)
        self.n.append(len(vals))
        self.m.append(o_mean)
        self.sqm.append(o_sq_mean)

    def std_err(self):
        '''Standard Error of result.'''
        return sqrt(self.var()/sum(self.n))

    def var(self):
        '''Binned Variance.'''
        return var(self.m,axis=0,ddof=1)

    def var_unbinned(self):
        '''Unbinned variance.'''
        if self.nbin==0:
            return nan
        n=reshape(self.n,[-1]+[1]*ndim(self.m[0]))
        N=sum(n)
        return (sum(n*self.sqm,axis=0)/N-(sum(n*self.m,axis=0)/N)**2)*N/(N-1)

    def t_auto(self):
        '''Autocorrelation time.'''
        return 0.5*mean(self.n)*mean(self.var())/mean(self.var_unbinned())

    def mean(self):
        '''
        Mean value of observable.
        '''
        n=reshape(self.n,[-1]+[1]*ndim(self.m[0]))
        return sum(n*self.m,axis=0)/n.sum()
