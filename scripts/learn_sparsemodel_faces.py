import hdl
reload(hdl)

from hdl.models import SparseSlowModel
from hdl.learners import SGD

whitenpatches = 160000

#l = SGD(model=SparseSlowModel(patch_sz=64,N=1024,T=64,sparse_cost='l1',slow_cost=None,perc_var=99),datasource='YouTubeFaces_aligned')
#l = SGD(model=SparseSlowModel(patch_sz=32,N=512,T=64,sparse_cost='l1',slow_cost=None,perc_var=99),datasource='YouTubeFaces_aligned',display_every=20000)
#l = SGD(model=SparseSlowModel(patch_sz=64,N=1024,T=64,sparse_cost='l1',slow_cost=None,perc_var=99),datasource='YouTubeFaces_aligned',display_every=20000)
#l = SGD(model=SparseSlowModel(patch_sz=48,N=768,T=64,sparse_cost='l1',slow_cost=None,perc_var=99),datasource='YouTubeFaces_aligned',display_every=20000)
#l = SGD(model=SparseSlowModel(patch_sz=48,N=1024,T=64,sparse_cost='l1',slow_cost=None,perc_var=99.25),datasource='YouTubeFaces_aligned',display_every=20000)
#l = SGD(model=SparseSlowModel(patch_sz=48,N=768,T=64,sparse_cost='l1',slow_cost=None,perc_var=99.),datasource='YouTubeFaces_aligned',display_every=20000)
#l = SGD(model=SparseSlowModel(patch_sz=48,N=1024,T=64,sparse_cost='subspacel1',slow_cost=None,perc_var=99.),datasource='YouTubeFaces_aligned',display_every=20000)
#l = SGD(model=SparseSlowModel(patch_sz=48,N=4096,T=64,lam_sparse=.4,sparse_cost='l1',slow_cost=None,perc_var=99.),datasource='YouTubeFaces_aligned',display_every=20000)

#l = SGD(model=SparseSlowModel(patch_sz=48,N=1024,T=64,sparse_cost='l1',slow_cost=None,perc_var=99.),datasource='TorontoFaces48',display_every=20000)
#l = SGD(model=SparseSlowModel(patch_sz=48,N=512,T=48,sparse_cost='l1',slow_cost=None,perc_var=99.),datasource='TorontoFaces48',display_every=20000)
#l = SGD(model=SparseSlowModel(patch_sz=48,N=256,T=48,sparse_cost='l1',slow_cost=None,perc_var=99.),datasource='TorontoFaces48',display_every=20000)
#l = SGD(model=SparseSlowModel(patch_sz=96,N=1024,T=48,sparse_cost='l1',slow_cost=None,perc_var=99.),datasource='TorontoFaces96',display_every=20000)
#l = SGD(model=SparseSlowModel(patch_sz=48,N=512,T=48,sparse_cost='subspacel1',slow_cost=None,perc_var=99.),datasource='TorontoFaces48',display_every=20000)

#l = SGD(model=SparseSlowModel(patch_sz=48,N=1024,T=48,sparse_cost='l1',slow_cost=None,perc_var=99.),datasource='YouTubeFaces_aligned_asymmetric',display_every=20000)

l = SGD(model=SparseSlowModel(patch_sz=48,N=4096,T=48,sparse_cost='l1',slow_cost=None,lam_sparse=1.6,perc_var=99.),datasource='YouTubeFaces_aligned',display_every=20000,batchsize=48)

databatch = l.get_databatch(whitenpatches)
l.model.learn_whitening(databatch)
l.model.setup()

l.learn(iterations=500000)
#l.batchsize *= 2
#l.learn(iterations=100000)
l.change_target(.5)
#l.batchsize *= 2
l.learn(iterations=100000)
l.change_target(.5)
#l.batchsize *= 2
l.learn(iterations=100000)

from hdl.display import display_final
display_final(l.model)
