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
#from hdl.learners import SGD

whitenpatches = 400

convwhitenfiltershp=(11,11)
perc_var = 99.
N = 641
kshp = (16,16)
stride = (8,8)
imsz = kshp[0]*6
imshp=(2,1,imsz,imsz)

print 'Init...'
l = SGD(model=ConvSparseSlowModel(imshp=imshp,convwhitenfiltershp=convwhitenfiltershp,perc_var=perc_var,N=N,kshp=kshp,stride=stride,
        sparse_cost='subspacel1mean',slow_cost=None,lam_sparse=1.,center_basis_functions=False),
        datasource='berkeleysegmentation',batchsize=imshp[0],save_every=20000,display_every=1000,
        ipython_profile=profile)

print 'Estimate whitening...'
databatch = l.get_databatch(whitenpatches)
l.model.learn_whitening(databatch)
l.model.setup()

l.learn(iterations=40000)
l.change_target(.5)
l.learn(iterations=5000)
l.change_target(.5)
l.learn(iterations=5000)

from hdl.display import display_final
display_final(l.model)
