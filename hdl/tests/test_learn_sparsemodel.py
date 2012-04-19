import hdl
reload(hdl)

from hdl.models import SparseSlowModel
from hdl.learners import SGD

whitenpatches = 10000

l = SGD(model=SparseSlowModel(patch_sz=8,N=64,T=64,sparse_cost='l1',slow_cost=None),datasource='vid075-chunks',display_every=500,save_every=500)

databatch = l.get_databatch(whitenpatches)
l.model.learn_whitening(databatch)
l.model.setup()

l.learn()
