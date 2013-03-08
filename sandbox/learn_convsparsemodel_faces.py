import argparse

parser = argparse.ArgumentParser(description='Initialize Theano on IPython Cluster GPUs')
parser.add_argument('--profile', type=str, default='nodb',
    help='profile name of IPython Cluster')
args = parser.parse_args()
profile = args.profile
print 'Using IPython Cluster Profile:', profile

import hdl

reload(hdl)

from hdl.models import ConvSparseSlowModel
from hdl.parallel_learners import SGD
#from hdl.learners import SGD

whitenpatches = 2000

convwhitenfiltershp = (15, 15)
perc_var = 99.9
N = 1024
kshp = (16, 16)
stride = (8, 8)
imsz = 48
imshp = (10, 1, imsz, imsz)

print 'Init...'
l = SGD(
        model=ConvSparseSlowModel(imshp=imshp, convwhitenfiltershp=convwhitenfiltershp, perc_var=perc_var, N=N, kshp=kshp,
                                  stride=stride, center_basis_functions=False,mask=True,
                                  sparse_cost='l1', slow_cost=None, lam_sparse=1.),
        datasource='YouTubeFaces_aligned', new_patch_size_fraction=1.5,pixels_from_center=80,
        batchsize=imshp[0],
        save_every=20000,
        display_every=10000,
        eta_target_maxupdate=.05,
        ipython_profile=profile)

print 'Estimate whitening...'
databatch = l.get_databatch(whitenpatches)
l.model.learn_whitening(databatch)
l.model.setup()

print 'kshp:', l.model.kshp
print 'imshp:', l.model.imshp
print 'featshp:', l.model.featshp

l.learn(iterations=160000)
l.change_target(.5)
l.learn(iterations=5000)
l.change_target(.5)
l.learn(iterations=5000)

from hdl.display import display_final

display_final(l.model)

l.model.save()

#ConvSparseSlowModel_patchsz064_ksz008_nchannels001_stride004_N031_NN031_convl2_subspacel1mean_None_2012-09-21_20-37-38
# took about 10k iterations to see anything.
