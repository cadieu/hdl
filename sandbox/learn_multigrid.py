import numpy as np

import hdl
from hdl.models import SparseSlowModel
from hdl.learners import SGD
from hdl.display import display_final

whitenpatches = 160000
l = SGD(model=SparseSlowModel(patch_sz=2,N=8,T=16,sparse_cost='l1',slow_cost=None,perc_var=99.9),
    datasource='berkeleysegmentation',batchsize=16,save_every=20000,display_every=100,
    eta_target_maxupdate=.02)

def learn_loop(l,iterations=20000):
    print 'Start learn_loop with %d iterations'%iterations

    l.learn(iterations=int(np.ceil(.9*iterations)))
    l.change_target(.5)
    l.learn(iterations=int(np.ceil(.05*iterations)))
    l.change_target(.5)
    l.learn(iterations=int(np.ceil(.05*iterations)))
    return l

def double_model(l):

    newN = 2*l.model.N

    A = l.model.A.get_value()
    newA = np.zeros((A.shape[0],newN))
    avgA = np.zeros_like(A)
    avgA[:,:-1] = .5*(A[:,:-1] + A[:,1:])
    avgA[:,-1] = .5*(A[:,-1] + A[:,0])
    newA[:,::2] = A
    newA[:,1::2] = avgA

    l.model.N = newN
    l.model.NN = newN
    newA = l.model.normalize_A(newA)
    l.model.A.set_value(newA.astype(hdl.models.theano.config.floatX))
    l.model.reset_functions()

    return l

def double_patch_sz(l):

    from scipy.misc.pilutil import imresize
    new_patch_sz = 2*l.model.patch_sz
    newN = 2*l.model.N

    A = l.model.A.get_value()
    A = np.dot(l.model.dewhitenmatrix,A)

    upA = np.zeros((new_patch_sz**2,l.model.N))
    for j in range(A.shape[1]):
        a = imresize(A[:,j].reshape(l.model.patch_sz,l.model.patch_sz),(new_patch_sz,new_patch_sz),mode='F')
        upA[:,j] = a.ravel()

    newA = np.zeros((new_patch_sz**2,newN))
    avgA = np.zeros_like(upA)
    avgA[:,:-1] = .5*(upA[:,:-1] + upA[:,1:])
    avgA[:,-1] = .5*(upA[:,-1] + upA[:,0])
    newA[:,::2] = upA
    newA[:,1::2] = avgA

    l.model.N = newN
    l.model.NN = newN
    l.model.patch_sz = new_patch_sz
    l.model.D = new_patch_sz**2

    databatch = l.get_databatch(whitenpatches)
    l.model.learn_whitening(databatch)

    newA = np.dot(l.model.whitenmatrix,newA)
    newA = l.model.normalize_A(newA)
    l.model.A.set_value(newA.astype(hdl.models.theano.config.floatX))
    l.model.reset_functions()

    return l

initial_target = l.eta_target_maxupdate

databatch = l.get_databatch(whitenpatches)
l.model.learn_whitening(databatch)
l.model.setup()
l.eta_target_maxupdate
iterations = 2000
l = learn_loop(l,iterations=iterations)
display_final(l.model,save_string='doubling_0')

doublings = 4
for doubling in range(doublings):
    iterations *= 2
    l.eta_target_maxupdate = initial_target
    print 'Doubling model...'
    l = double_patch_sz(l)

    display_final(l.model,save_string='doubling_%d_before_learning'%(doubling+1))
    l = learn_loop(l,iterations=iterations)

    display_final(l.model,save_string='doubling_%d_after_learning'%(doubling+1))
