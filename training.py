'''
Train a RBM.
'''
from numpy import *

def train_rbm(rbm,train_data,niter=5000,rate=0.1):
    '''train RBM.'''
    vi=insert(train_data,0,1,axis=1)
    for i in xrange(niter):
        hi,hi_prob=rbm.feed_visible(vi,return_prob=True)
        #reconstruction
        vi2,vi2_prob=rbm.feed_hidden(hi,return_prob=True)
        vi2_prob[:,0]=1
        hi2,hi2_prob=rbm.feed_visible(vi2_prob,return_prob=True)
        if i%100==0:
            print 'Iter %s, Error = %s'%(i,sum((train_data-vi2_prob[:,1:])**2))
        rbm.W+=rate*(vi.T.dot(hi_prob)-vi2_prob.T.dot(hi2_prob))/len(train_data)


