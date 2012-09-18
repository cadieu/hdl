import hdl
reload(hdl)

from hdl.models import ConvSparseSlowModel
from hdl.learners import SGD

whitenpatches = 200

patch_sz = 48
kshp = (16,16)
N = 256
model = ConvSparseSlowModel(imshp=(2,1,patch_sz,patch_sz),convwhitenfiltershp=(7,7),N=N,kshp=kshp,
    perc_var=99.5,lam_sparse=1.0)

l = SGD(model=model,datasource='YouTubeFaces_aligned',display_every=1000,batchsize=model.imshp[0])

databatch = l.get_databatch(whitenpatches)
l.model.learn_whitening(databatch)
l.model.setup()

l.learn(iterations=100000)
#l.batchsize *= 2
#l.learn(iterations=100000)
l.change_target(.5)
#l.batchsize *= 2
l.learn(iterations=100000)
l.change_target(.5)
#l.batchsize *= 2
l.learn(iterations=100000)
l.change_target(.5)
#l.batchsize *= 2
l.learn(iterations=100000)

from hdl.display import display_final
display_final(l.model)
