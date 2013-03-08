import argparse

parser = argparse.ArgumentParser(description='Initialize Theano on IPython Cluster GPUs')
parser.add_argument('--profile',type=str,default='nodb',
                    help='profile name of IPython Cluster')
args = parser.parse_args()
profile = args.profile
print 'Using IPython Cluster Profile:', profile

import hdl
reload(hdl)

from hdl.models import ConvSparseSlowModel
from hdl.parallel_learners import SGD

whitenpatches = 400

convwhitenfiltershp=(7,7)
perc_var = 100.
N = 256
kshp = (8,8)
stride = (4,4)
imsz = kshp[0]*6
imshp=(2,1,imsz,imsz)

print 'Init...'
l = SGD(model=ConvSparseSlowModel(imshp=imshp,convwhitenfiltershp=convwhitenfiltershp,perc_var=perc_var,N=N,kshp=kshp,stride=stride,sparse_cost='l1',slow_cost=None,lam_sparse=1.),
    datasource='berkeleysegmentation',batchsize=imshp[0],save_every=20000,display_every=20000,
    ipython_profile=profile)

print 'Estimate whitening...'
databatch = l.get_databatch(whitenpatches)
l.model.learn_whitening(databatch)
l.model.setup()

l.learn(iterations=100)

from time import time as now

t0 = now()
l.learn(iterations=100)
print 'time = ', now() - t0

#l.change_target(.5)
#l.learn(iterations=2000)
#l.change_target(.5)
#l.learn(iterations=2000)
#
#from hdl.display import display_final
#display_final(l.model)

# Using EC2 1 nodes (cc1.4xlarge), 14 engines
#l = SGD(model=ConvSparseSlowModel(imshp=imshp,convwhitenfiltershp=convwhitenfiltershp,perc_var=perc_var,N=N,kshp=kshp,stride=stride,sparse_cost='l1',slow_cost=None,lam_sparse=1.),
#    datasource='berkeleysegmentation',batchsize=imshp[0],save_every=20000,display_every=20000,
#    ipython_profile=profile)
# l.learn(iterations=100)
# took 5.90

# OLD:::

# Using 4 gpu GeForce GTX 480, 1 node
# l = SGD(model=SparseSlowModel(patch_sz=16,N=2560,T=16,sparse_cost='l1',slow_cost=None,lam_sparse=1.0),datasource='berkeleysegmentation',batchsize=16,save_every=20000,display_every=20000)
# l.learn(iterations=1000)
# took 27.38 seconds

# Using 16 gpus (GTX 480), 4 nodes
# l = SGD(model=SparseSlowModel(patch_sz=16,N=2560,T=16,sparse_cost='l1',slow_cost=None,lam_sparse=1.0),datasource='berkeleysegmentation',batchsize=16,save_every=20000,display_every=20000)
# l.learn(iterations=1000)
# took 14.92 seconds

# Using 16 gpus + active coefficient selection and transmission:
# l = SGD(model=SparseSlowModel(patch_sz=16,N=2560,T=16,sparse_cost='l1',slow_cost=None,lam_sparse=1.0),datasource='berkeleysegmentation',batchsize=16,save_every=20000,display_every=20000)
# l.learn(iterations=1000)
# took 11.37

# Using EC2 6 nodes (cg1.4xlarge), 2 GPUs each, and running from master node
# l = SGD(model=SparseSlowModel(patch_sz=16,N=2560,T=16,sparse_cost='l1',slow_cost=None,lam_sparse=1.0),datasource='berkeleysegmentation',batchsize=16,save_every=20000,display_every=20000)
# l.learn(iterations=1000)
# took 56.08

# Using EC2 2 nodes (cg1.4xlarge), 12 engines each, and running from master node
# l = SGD(model=SparseSlowModel(patch_sz=16,N=2560,T=16,sparse_cost='l1',slow_cost=None,lam_sparse=1.0),datasource='berkeleysegmentation',batchsize=16,save_every=20000,display_every=20000)
# l.learn(iterations=1000)
# took 23.58

# Using EC2 2 nodes (cc2.8xlarge), 8 engines each, and running from master node
# l = SGD(model=SparseSlowModel(patch_sz=16,N=2560,T=16,sparse_cost='l1',slow_cost=None,lam_sparse=1.0),datasource='berkeleysegmentation',batchsize=16,save_every=20000,display_every=20000)
# l.learn(iterations=1000)
# took 30.01

# Using EC2 2 nodes (cc2.8xlarge), 16 engines each, and running from master node
# l = SGD(model=SparseSlowModel(patch_sz=16,N=2560,T=16,sparse_cost='l1',slow_cost=None,lam_sparse=1.0),datasource='berkeleysegmentation',batchsize=16,save_every=20000,display_every=20000)
# l.learn(iterations=1000)
# took 20.00

# Using EC2 2 nodes (cc2.8xlarge), 24 engines each, and running from master node
# l = SGD(model=SparseSlowModel(patch_sz=16,N=2560,T=16,sparse_cost='l1',slow_cost=None,lam_sparse=1.0),datasource='berkeleysegmentation',batchsize=16,save_every=20000,display_every=20000)
# l.learn(iterations=1000)
# took 17.32

# Using EC2 2 nodes (cc2.8xlarge), 30 engines each, and running from master node
# l = SGD(model=SparseSlowModel(patch_sz=16,N=2560,T=16,sparse_cost='l1',slow_cost=None,lam_sparse=1.0),datasource='berkeleysegmentation',batchsize=16,save_every=20000,display_every=20000)
# l.learn(iterations=1000)
# took 15.34

# Using EC2 4 nodes (cc2.8xlarge), 24 engines each, and running from master node
# l = SGD(model=SparseSlowModel(patch_sz=16,N=2560,T=16,sparse_cost='l1',slow_cost=None,lam_sparse=1.0),datasource='berkeleysegmentation',batchsize=16,save_every=20000,display_every=20000)
# l.learn(iterations=1000)
# took 8.02

# Using EC2 4 nodes (cc2.8xlarge), 30 engines each, and running from master node
# l = SGD(model=SparseSlowModel(patch_sz=16,N=2560,T=16,sparse_cost='l1',slow_cost=None,lam_sparse=1.0),datasource='berkeleysegmentation',batchsize=16,save_every=20000,display_every=20000)
# l.learn(iterations=1000)
# took 7.09
