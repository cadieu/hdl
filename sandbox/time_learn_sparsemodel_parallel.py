import hdl
reload(hdl)

from hdl.models import SparseSlowModel
from hdl.parallel_learners import SGD

whitenpatches = 160000

l = SGD(model=SparseSlowModel(patch_sz=16,N=25600,T=16,sparse_cost='l1',slow_cost=None,lam_sparse=1.0),
    datasource='berkeleysegmentation',batchsize=16,save_every=20000,display_every=20000,
    ipython_profile='gpupbs')

databatch = l.get_databatch(whitenpatches)
l.model.learn_whitening(databatch)
l.model.setup()

l.learn(iterations=100)

from time import time as now

t0 = now()
l.learn(iterations=1000)
print 'time = ', now() - t0

#l.change_target(.5)
#l.learn(iterations=2000)
#l.change_target(.5)
#l.learn(iterations=2000)
#
#from hdl.display import display_final
#display_final(l.model)

# Using 4 gpu GeForce GTX 480
# l = SGD(model=SparseSlowModel(patch_sz=16,N=2560,T=16,sparse_cost='l1',slow_cost=None,lam_sparse=1.0),datasource='berkeleysegmentation',batchsize=16,save_every=20000,display_every=20000)
# l.learn(iterations=1000)
# took 27.38 seconds

# Using 16 gpus
# l = SGD(model=SparseSlowModel(patch_sz=16,N=2560,T=16,sparse_cost='l1',slow_cost=None,lam_sparse=1.0),datasource='berkeleysegmentation',batchsize=16,save_every=20000,display_every=20000)
# l.learn(iterations=1000)
# took 14.92 seconds

# Using 16 gpus + active coefficient selection and transmission:
# l = SGD(model=SparseSlowModel(patch_sz=16,N=2560,T=16,sparse_cost='l1',slow_cost=None,lam_sparse=1.0),datasource='berkeleysegmentation',batchsize=16,save_every=20000,display_every=20000)
# l.learn(iterations=1000)
# took 11.37
