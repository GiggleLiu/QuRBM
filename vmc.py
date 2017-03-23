'''
Variational Monte Carlo Kernel.
'''

from abc import ABCMeta, abstractmethod

__all__=['VMC']

class MCCore(object):
    '''
    Interface of Monte Carlo Kernel.
    '''
    __metaclass__ = ABCMeta

    def __init__(self,sampling_method='metropolis'):
        self.sampling_method=sampling_method

    @abstractmethod
    def fire(self,**kwargs):
        '''
        Get a possible proposal for the next move.

        Return:
            proposal,
        '''
        pass

    @abstractmethod
    def reject(self,proposal,*args,**kwargs):
        '''
        Reject the proposal.

        Parameters:
            proposal: object, the proposal.
        '''
        pass

class VMC(object):
    '''
    Variational Monte Carlo Engine.
    '''
    def __init__(self,core,nbath,nsample,sampling_method='metropolis'):
        self.nbath,self.nsample=nbath,nsample
        self.core=core
        self.sampling_method=sampling_method

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

    def measure(self,op,state):
        '''
        Measure an operator.

        Parameters:
            :op: <LinOp>, a linear operator instance.
            :state: <RBM>/..., a state ansaz

        Return:
            number,
        '''
        ol=[]  #local operator values
        for i in xrange(self.nbath):
            #generate new config
            new_config,pratio=self.core.fire(config)
            if self.accept(pratio):
                #update config
                config=new_config
        for i in xrange(self.nsample):
            #generate new config
            new_config,pratio=self.core.fire(config,state)
            if self.accept(pratio):
                o=sandwich(config,op,state)
                w=state.get_weight(config)
                ol.append(o/w)      #correlation problem?
                config=new_config
            else:
                ol.append(o/w)
        return mean(ol)

class RBMCore(MCCore):
    def __init__(self):
        self.thl=[]

    def fire(self,config,rbm):
        '''Fire a proposal.'''
        nc=copy(config)  #why random flip, how about system with good quantum number?
        iflip=random.randint(len(config))
        nc[iflip]=1-iflip
        #transfer probability is equal, pratio is equal to the probability ratio
        pratio=abs(rbm.get_weight(nc)/rbm.get_weight(config))**2
        return nc,pratio
