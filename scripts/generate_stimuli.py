import os
import numpy as np
from scipy.misc.pilutil import toimage
import hdl
reload(hdl)

from hdl.models import SparseSlowModel
from hdl.config import state_dir, fig_dir
from hdl.display import display_patches


def generate_whitenedspace(m,numstimuli,fig_num=1):

    rvals = np.random.randn(m.M,numstimuli)
    patches = np.dot(m.dewhitenmatrix,rvals)

    array = display_patches(patches,m.patch_sz,fig_num=fig_num)

    savepath = os.path.join(fig_dir,m.model_name + '_' + m.tstring)
    if not os.path.isdir(savepath): os.makedirs(savepath)
    fname = os.path.join(savepath, 'Whitened_patches.png')
    toimage(np.floor(.5*(array+1)*255)).save(fname)

def generate_sparsespace(m,numstimuli,sparsity=5.,fig_num=2):

    binomial_p = float(sparsity)/m.NN

    rvals = np.random.randn(m.NN,numstimuli)
    rvals *= np.random.binomial(1,binomial_p,size=rvals.shape)

    patches = np.dot(m.dewhitenmatrix,np.dot(m.A,rvals))

    array = display_patches(patches,m.patch_sz,fig_num=fig_num)

    savepath = os.path.join(fig_dir,m.model_name + '_' + m.tstring)
    if not os.path.isdir(savepath): os.makedirs(savepath)
    fname = os.path.join(savepath, 'Sparse_patches_%d.png'%int(sparsity))
    toimage(np.floor(.5*(array+1)*255)).save(fname)

if __name__ == '__main__':

    #model_name = 'SparseSlowModel_patchsz064_N2048_NN2048_l2_l1_None_2012-02-05_15-29-08/SparseSlowModel_patchsz064_N2048_NN2048_l2_l1_None.model'

    # faces
    model_name = 'SparseSlowModel_patchsz048_N1024_NN1024_l2_l1_None_2012-02-13_16-12-17/SparseSlowModel_patchsz048_N1024_NN1024_l2_l1_None.model'

    fname = os.path.join(state_dir,model_name)
    m = SparseSlowModel()
    m.load(fname, reset_theano=False)

    numstimuli = 20**2

    stimuli = generate_whitenedspace(m,numstimuli)

    for sparsity in [1, 2, 3, 4, 5, 10, 20, 40]:
        stimuli = generate_sparsespace(m,numstimuli,sparsity=sparsity)