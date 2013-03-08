import hdl

reload(hdl)

from hdl.models import ConvSparseSlowModel
from hdl.learners import SGD

whitenpatches = 400

convwhitenfiltershp = (7, 7)
perc_var = 100.
N = 256
kshp = (8, 8)
stride = (4, 4)
imsz = kshp[0] * 6
imshp = (2, 1, imsz, imsz)

print 'Init...'
l = SGD(
    model=ConvSparseSlowModel(imshp=imshp, convwhitenfiltershp=convwhitenfiltershp, perc_var=perc_var, N=N, kshp=kshp,
        stride=stride, sparse_cost='l1', slow_cost=None, lam_sparse=1.),
    datasource='berkeleysegmentation', batchsize=imshp[0], save_every=20000, display_every=20000,
    )

print 'Estimate whitening...'
databatch = l.get_databatch(whitenpatches)
l.model.learn_whitening(databatch)
l.model.setup()

l.learn(iterations=100)

from time import time as now

t0 = now()
l.learn(iterations=100)
print 'time = ', now() - t0

# Using EC2 1 nodes (cc1.4xlarge)
#l = SGD(model=ConvSparseSlowModel(imshp=imshp,convwhitenfiltershp=convwhitenfiltershp,perc_var=perc_var,N=N,kshp=kshp,stride=stride,sparse_cost='l1',slow_cost=None,lam_sparse=1.),
#    datasource='berkeleysegmentation',batchsize=imshp[0],save_every=20000,display_every=20000,
#    ipython_profile=profile)
# l.learn(iterations=100)
# took 39.39

#l.change_target(.5)
#l.learn(iterations=2000)
#l.change_target(.5)
#l.learn(iterations=2000)
#
#from hdl.display import display_final
#display_final(l.model)

