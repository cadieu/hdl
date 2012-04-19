import numpy as np
import theano
from hdl.fista import Fista

def test_sparseslow_fista_theano(largeproblem=False,verbose=True,cast=True,display=False):

    np.random.seed(113)

    # Set up some variables
    if not largeproblem:
        N = 10 # number of sparse components
        nN = 3
        M = 20 # size of the input space
        D = 32 # number of samples
        lam_sparse = .1/N
        lam_slow = .1/N
    else:
        N = 512 # number of sparse components
        nN = 128
        M = 1024 # size of the input space
        D = 48 # number of samples
        lam_sparse = .1/N
        lam_slow = .1/N

    maxiter = 200
    maxline = 10
    errthres = 1e-8

    A = np.random.randn(M,N)
    A = np.dot(A,np.diag(1./np.sqrt(np.sum(A**2,axis=0)))) # normalize

    var_delta = .2

    amp = np.zeros((N/2,D))
    for i in range(nN):
        ii = np.floor(N/2*np.random.rand())
        cc = np.random.uniform(0,1./nN/2)
        amp[ii,0] += cc
        for d in range(1,D):
            amp[ii,d] = amp[ii,d-1] + var_delta*np.random.randn()

    #amp[:,0] = np.abs(np.random.randn(N/2,))
    amp *= amp > 0
    z = amp*np.exp(1j*2*np.pi*np.random.rand(N/2,D))
    a = np.zeros((N,D))
    a[::2,:] = np.real(z)
    a[1::2,:] = np.imag(z)

    x = np.dot(A,a)

    ainit = .1*np.random.randn(*a.shape)
    #ainit = np.dot(A.T,x)

    if cast:
        ainit = getattr(np,theano.config.floatX)(ainit)
        x = getattr(np,theano.config.floatX)(x)
        A = getattr(np,theano.config.floatX)(A)
        lam_sparse = getattr(np,theano.config.floatX)(lam_sparse)
        lam_slow = getattr(np,theano.config.floatX)(lam_slow)

    T_lam_sparse = theano.shared(getattr(np,theano.config.floatX)(lam_sparse))
    T_lam_slow = theano.shared(getattr(np,theano.config.floatX)(lam_slow))
    T_x = theano.shared(np.random.randn(M,D).astype(theano.config.floatX))
    T_A = theano.shared(np.random.randn(M,N).astype(theano.config.floatX))
    T_u = np.random.randn(N,D).astype(theano.config.floatX)

    _fista = Fista(xinit=T_u,A=T_A,lam_sparse=T_lam_sparse,lam_slow=T_lam_slow,x=T_x,problem_type='l2subspacel1slow')

    T_A.set_value(A)
    ahat, history = _fista(ainit, x, maxiter=maxiter,maxline=maxline,errthres=errthres,verbose=verbose)

    aerror0 = a - ainit
    aerror1 = a - ahat
    print 'mse(a-a0) =', np.sum(aerror0**2), 'mse(a-ahat) =', np.sum(aerror1**2)

    if display:
        from matplotlib import pyplot as plt

        ampinit = np.sqrt(ainit[::2,:]**2 + ainit[1::2,:]**2)
        amphat = np.sqrt(ahat[::2,:]**2 + ahat[1::2,:]**2)

        plt.figure(1)
        plt.clf()
        plt.subplot(2,3,1)
        hval = np.max(np.abs(a))
        plt.imshow(ainit,vmin=-hval,vmax=hval,interpolation='nearest')
        plt.title('ainit')
        plt.subplot(2,3,2)
        plt.imshow(ahat,vmin=-hval,vmax=hval,interpolation='nearest')
        plt.title('ahat')
        plt.subplot(2,3,3)
        plt.imshow(a,vmin=-hval,vmax=hval,interpolation='nearest')
        plt.title('atrue')
        plt.subplot(2,3,4)
        hval = np.max(np.abs(amp))
        plt.imshow(ampinit,vmin=0,vmax=hval,interpolation='nearest',cmap=plt.cm.gray)
        plt.title('ainit')
        plt.subplot(2,3,5)
        plt.imshow(amphat,vmin=-hval,vmax=hval,interpolation='nearest',cmap=plt.cm.gray)
        plt.title('ahat')
        plt.subplot(2,3,6)
        plt.imshow(amp,vmin=-hval,vmax=hval,interpolation='nearest',cmap=plt.cm.gray)
        plt.title('atrue')
        plt.draw()

        plt.figure(2)
        plt.clf()
        plt.subplot(1,2,1)
        plt.scatter(a.ravel(),ahat.ravel())
        plt.xlabel('atrue')
        plt.ylabel('ahat')
        plt.subplot(1,2,2)
        plt.scatter(amp.ravel(),amphat.ravel())
        plt.xlabel('amptrue')
        plt.ylabel('amphat')

        plt.draw()
        plt.show()

if __name__ == '__main__':
    test_sparseslow_fista_theano(True)