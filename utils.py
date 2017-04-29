'''
Utilities.
'''

from numpy import *
import pdb

def load_carleo_wf(filename,group=None):
    '''Load wavefunction of carleo's program.'''
    with open(filename,'r') as f:
        fl=f.readlines()
    nv=int(fl[0].strip('\n'))
    nh=int(fl[1].strip('\n'))
    datas=array([fromstring(s.strip('()\n'),sep=',') for s in fl[2:]])
    datas=datas[:,0]+1j*datas[:,1]
    a=datas[:nv]/group.ng
    b=datas[nv:nv+nh]
    W=datas[nv+nh:].reshape(nv,nh)
    if group is not None or group.ng==1:
        b=b[::group.ng]
        W=ascontiguousarray(W[:,::group.ng])
    return a,b,W

def logcosh(theta):
    '''log(cosh(theta)).'''
    res=zeros_like(theta)
    overflow=abs(theta.real)>12
    res[overflow]=theta[overflow]-log(2.)
    res[~overflow]=log(cosh(theta[~overflow]))
    return res
