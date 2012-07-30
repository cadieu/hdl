from collections import defaultdict
import numpy as np
import theano
from theano import function
import hdl
hdl.config.verbose_timing = False
from time import time as now

from hdl.theano_methods import T_l2_cost_norm

reload(hdl)

from hdl.models import SparseSlowModel
from hdl.learners import SGD, autoSGD

debug_grad = False
# Create auotSGD class
# Create fixedSGD class
# write metric evaluation

# run test for all three that measures time, and loss. -> then plots

loss_setsize = 10000

whitenpatches = 40000
ldefault = SGD(model=SparseSlowModel(patch_sz=16,N=256,T=1,sparse_cost='l1',slow_cost=None,u_init_method='proj'),datasource='vid075-chunks',display_every=100000,save_every=100000,batchsize=1)

databatch = ldefault.get_databatch(whitenpatches)
ldefault.model.learn_whitening(databatch)
ldefault.model.setup()

l = autoSGD(model=SparseSlowModel(patch_sz=16,N=256,T=1,sparse_cost='l1',slow_cost=None,u_init_method='proj'),datasource='vid075-chunks',display_every=100000,save_every=100000,batchsize=1)

l.model.inputmean = ldefault.model.inputmean.copy()
l.model.whitenmatrix = ldefault.model.whitenmatrix.copy()
l.model.dewhitenmatrix = ldefault.model.dewhitenmatrix.copy()
l.model.zerophasewhitenmatrix = ldefault.model.zerophasewhitenmatrix.copy()
l.model.M = ldefault.model.M
l.model.D = ldefault.model.D
l.model.setup()
l.model.A.set_value(ldefault.model.A.get_value())

if debug_grad:
    X = l.get_databatch(100)
    preX = l.model.preprocess(X)
    u = l.model.inferlatent(preX)

    Tx = theano.tensor.matrix('x')
    Tu = theano.tensor.matrix('u')

    f = l.model._df_dA

    diff = 1e-5
    A0 = l.model.A.get_value()
    A1 = np.zeros_like(A0)
    A1[:] = A0
    f0 = f(preX,u)
    fvals = []
    for i in range(10):
        for j in range(10):
            A1[i,j] = A0[i,j] + diff
            l.model.A.set_value(A1)
            f1 = f(preX,u)[i,j]
            d = (f1 - f0[i,j])/diff
            fvals.append(d)
            A1[i,j] = A0[i,j]

    print 'Theano 2nd:'
    theano_grad2 = l.model._df_dA2(preX,u).reshape(A0.shape)
    theano_fvals = [theano_grad2[i,j] for i in range(10) for j in range(10)]
    print theano_fvals

    print 'finite:'
    print fvals


X = ldefault.get_databatch(loss_setsize)
preXdefault = ldefault.model.preprocess(X)
preX = l.model.preprocess(X)

def compute_loss(m,X):
    A = m.A.get_value()
    u = m.inferlatent(X)
    Xhat = np.dot(A,m.inferlatent(X))
    return .5*np.sum((X - Xhat)**2) + np.sum(np.abs(u))*m.lam_sparse.get_value()


iterations = 100
epochs = 100
t0 = now()
lossdefault = defaultdict(list)
loss = compute_loss(ldefault.model,preXdefault)
print ldefault.iter, 'Default method loss:', loss
tic = now() - t0
lossdefault['loss'].append(loss)
lossdefault['time'].append(tic)
lossdefault['iter'].append(ldefault.iter)
for epoch in range(int(.8*epochs)):
    ldefault.learn(iterations=iterations)
    loss = compute_loss(ldefault.model,preXdefault)
    print ldefault.iter, 'Default method loss:', loss, 'learner.eta', ldefault.eta
    tic = now() - t0
    lossdefault['loss'].append(loss)
    lossdefault['time'].append(tic)
    lossdefault['iter'].append(ldefault.iter)

ldefault.change_target(.5)
for epoch in range(int(.1*epochs)):
    ldefault.learn(iterations=iterations)
    loss = compute_loss(ldefault.model,preXdefault)
    print ldefault.iter, 'Default method loss:', loss, 'learner.eta', ldefault.eta
    tic = now() - t0
    lossdefault['loss'].append(loss)
    lossdefault['time'].append(tic)
    lossdefault['iter'].append(ldefault.iter)

ldefault.change_target(.5)
for epoch in range(int(.1*epochs)):
    ldefault.learn(iterations=iterations)
    loss = compute_loss(ldefault.model,preXdefault)
    print ldefault.iter, 'Default method loss:', loss, 'learner.eta', ldefault.eta
    tic = now() - t0
    lossdefault['loss'].append(loss)
    lossdefault['time'].append(tic)
    lossdefault['iter'].append(ldefault.iter)


t0 = now()
lossnew = defaultdict(list)
loss = compute_loss(l.model,preX)
print l.iter, 'Default method loss:', loss
tic = now() - t0
lossnew['loss'].append(loss)
lossnew['time'].append(tic)
lossnew['iter'].append(l.iter)
for epoch in range(epochs):
    l.learn(iterations=iterations)
    loss = compute_loss(l.model,preX)
    print l.iter, 'Default method loss:', loss, 'learner.eta', l.eta
    tic = now() - t0
    lossnew['loss'].append(loss)
    lossnew['time'].append(tic)
    lossnew['iter'].append(l.iter)

print lossdefault
print lossnew

from matplotlib import pyplot as plt

fig = plt.figure(1)
plt.clf()
plt.subplot(121)
plt.plot(lossdefault['iter'],lossdefault['loss'],label='SGD Adapt')
plt.plot(lossnew['iter'], lossnew['loss'], label='Auto SGD')
plt.legend()
plt.subplot(122)
plt.plot(lossdefault['time'],lossdefault['loss'],label='SGD Adapt')
plt.plot(lossnew['time'], lossnew['loss'], label='Auto SGD')
plt.legend()
plt.savefig('test_autoeta.png')
