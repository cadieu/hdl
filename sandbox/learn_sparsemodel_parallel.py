import argparse

parser = argparse.ArgumentParser(description='Initialize Theano on IPython Cluster GPUs')
parser.add_argument('--profile',type=str,default='nodb',
                    help='profile name of IPython Cluster')
args = parser.parse_args()
profile = args.profile

import hdl
reload(hdl)

from hdl.models import SparseSlowModel
from hdl.parallel_learners import SGD
from hdl.display import display_final

whitenpatches = 160000

#l = SGD(model=SparseSlowModel(patch_sz=16,N=10240,T=48,sparse_cost='l1',slow_cost=None,lam_sparse=1.0,perc_var=99.0),
#        datasource='berkeleysegmentation',batchsize=48,save_every=20000,display_every=20000,
#        ipython_profile=profile)

def go(l):

    databatch = l.get_databatch(whitenpatches)
    l.model.learn_whitening(databatch)
    l.model.setup()

    l.learn(iterations=350000)
    l.change_target(.5)
    l.learn(iterations=50000)
    l.change_target(.5)
    l.learn(iterations=50000)
    l.change_target(.5)
    l.learn(iterations=50000)

    display_final(l.model)


lam_sparse_values = [4.]#[.1, 1., 2., 4.]

for lam_sparse in lam_sparse_values:
    l = SGD(model=SparseSlowModel(patch_sz=16,N=10240,T=48,sparse_cost='l1',slow_cost=None,lam_sparse=lam_sparse,perc_var=99.0),
            datasource='berkeleysegmentation',batchsize=48,save_every=100000,display_every=40000,
            ipython_profile=profile)

    go(l)

# TIMING
#l.learn(iterations=100)
#from time import time as now
#t0 = now()
#l.learn(iterations=1000)
#print 'time = ', now() - t0
# Using 4 gpu GeForce GTX 480
# l = SGD(model=SparseSlowModel(patch_sz=16,N=2560,T=16,sparse_cost='l1',slow_cost=None,lam_sparse=1.0),datasource='berkeleysegmentation',batchsize=16,save_every=20000,display_every=20000)
# l.learn(iterations=1000)
# took 27.38 seconds
