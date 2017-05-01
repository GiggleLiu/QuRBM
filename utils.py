'''
Utilities.
'''

from numpy import *
import pdb

from setting import hyper_func

__all__=['load_carleo_wf','log2cosh','log2sinh','logfh']

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

def twocosh(theta):
    '''
    cosh(theta).
    '''
    return 2*cosh(theta)

def twosinh(theta):
    '''
    sinh(theta).
    '''
    return 2*sinh(theta)

def log2cosh(theta):
    '''
    log(cosh(theta)).
    '''
    res=zeros_like(theta)
    overflow=abs(theta.real)>12
    to=theta[overflow]
    res[overflow]=sign(to.real)*to
    res[~overflow]=log(2*cosh(theta[~overflow]))
    return res

def log2sinh(theta):
    '''
    log(sinh(theta)).
    '''
    res=zeros_like(theta)
    overflow=abs(theta.real)>12
    to=theta[overflow]
    res[overflow]=sign(to.real)*to-1j*pi*(to.real<0)
    res[~overflow]=log(2*sinh(theta[~overflow]))
    return res

def log2cosh_prime(theta):
    '''partial 2*cosh(theta)/partial theta'''
    return tanh(theta)

def log2sinh_prime(theta):
    '''partial cosh(theta)/partial theta'''
    return 1./tanh(theta)

if hyper_func=='cosh':
    logfh=log2cosh
    fh=twocosh
    logfh_prime=log2cosh_prime
elif hyper_func=='sinh':
    logfh=log2sinh
    fh=twosinh
    logfh_prime=log2sinh_prime
else:
    raise ValueError('undefined hyper function')

