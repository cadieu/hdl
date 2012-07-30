import os
import numpy as np
from scipy.misc import toimage

from models import SparseSlowModel
from config import state_dir
from matplotlib import pyplot as plt

def display_patches(patches,psz,fig_num=None,normalize=True):

    # plot the vectors in A
    NN = patches.shape[1]
    buf = 1
    sz = int(np.sqrt(NN))
    hval = np.max(np.abs(patches))
    array = -np.ones(((psz+buf)*sz+buf,(psz+buf)*sz+buf))
    pind = 0
    for r in range(sz):
        for c in range(sz):
            if pind >= NN:
                continue
            if normalize:
                hval = np.max(np.abs(patches[:,pind]))
                if hval == 0.: hval = 1
            patchesvalues = patches[:,pind].reshape(psz,psz)/hval
            array[buf+(psz+buf)*c:buf+(psz+buf)*c+psz,buf+(psz+buf)*r:buf+(psz+buf)*r+psz] = patchesvalues
            pind += 1
    hval = 1.
    if fig_num is None:
        plt.figure()
    else:
        plt.figure(fig_num)
    plt.clf()
    plt.imshow(array,vmin=-hval,vmax=hval,interpolation='nearest',cmap=plt.cm.gray)
    plt.colorbar()

    return array

def display_binoc_color_patches(binoc_patches,psz,fig_num=None,normalize=True):

    NN = binoc_patches.shape[1]
    patches = np.zeros((psz,psz,3,2*NN))
    count = 0
    for nn in range(NN):
        patches[...,count  ] = binoc_patches[...,nn].reshape(6,psz,psz)[:3,...].T
        patches[...,count+1] = binoc_patches[...,nn].reshape(6,psz,psz)[3:,...].T
        count += 2
    patches = patches.reshape((psz*psz*3,2*NN))

    return display_color_patches(patches=patches,psz=psz,fig_num=fig_num,normalize=normalize)

def display_color_patches(patches,psz,fig_num=None,normalize=True):
    # plot the vectors in A
    NN = patches.shape[1]
    buf = 1
    sz = int(np.sqrt(NN))
    if sz%2:
        sz += 1
    array = np.zeros(((psz+buf)*sz+buf,(psz+buf)*sz+buf,3))
    pind = 0
    for r in range(sz):
        for c in range(sz):
            if pind >= NN:
                continue
            if normalize:
                hval = np.max(np.abs(patches[:,pind]))
                if hval == 0.: hval = 1.
                patchesvalues = 127.5*patches[:,pind].reshape(psz,psz,3)/hval + 127.5
            else:
                patchesvalues = patches[:,pind].reshape(psz,psz,3)
            array[buf+(psz+buf)*c:buf+(psz+buf)*c+psz,buf+(psz+buf)*r:buf+(psz+buf)*r+psz,:] = patchesvalues
            pind += 1
    array = np.clip(array,0.,255.).astype(np.uint8)

    if fig_num is None:
        plt.figure()
    else:
        plt.figure(fig_num)
    plt.clf()
    plt.imshow(array,interpolation='nearest')

    return array

def display_final(m,save_string='final'):
    savepath = os.path.join(state_dir,m.model_name + '_' + m.tstring)
    if not os.path.exists(savepath): os.makedirs(savepath)

    repr_string = m.__repr__()
    model_details_fname = os.path.join(savepath,'model_details_final.txt')
    with open(model_details_fname,'w') as fh:
        fh.write(repr_string)

    max_factors = m.D

    d = m.display_whitening(save_string=save_string,max_factors=max_factors,zerophasewhiten=False)
    fname = os.path.join(savepath, 'whitenmatrix_hires_' + save_string + '.png')
    if d['whitenmatrix'].ndim == 2:
        toimage(np.floor(.5*(d['whitenmatrix']+1)*255)).save(fname)
    else:
        toimage(d['whitenmatrix']).save(fname)

    d = m.display_whitening(save_string=save_string,max_factors=max_factors)
    fname = os.path.join(savepath, 'whitenmatrix_hires_zerophase_' + save_string + '.png')
    if d['whitenmatrix'].ndim == 2:
        toimage(np.floor(.5*(d['whitenmatrix']+1)*255)).save(fname)
    else:
        toimage(d['whitenmatrix']).save(fname)

    if hasattr(m,'NN'):
        max_factors = m.NN

        d = m.display(save_string=save_string,max_factors=max_factors,zerophasewhiten=False)
        fname = os.path.join(savepath, 'A_hires_' + save_string + '.png')
        if d['A'].ndim == 2:
            toimage(np.floor(.5*(d['A']+1)*255)).save(fname)
        else:
            toimage(d['A']).save(fname)

        d = m.display(save_string=save_string,max_factors=max_factors)
        fname = os.path.join(savepath, 'A_hires_zerophase_' + save_string + '.png')
        if d['A'].ndim == 2:
            toimage(np.floor(.5*(d['A']+1)*255)).save(fname)
        else:
            toimage(d['A']).save(fname)


if __name__ == '__main__':


    #model_name = 'SparseSlowModel_patchsz064_N2048_NN2048_l2_l1_None_2012-02-05_15-29-08/SparseSlowModel_patchsz064_N2048_NN2048_l2_l1_None.model'

    # faces
    #model_name = 'SparseSlowModel_patchsz032_N512_NN512_l2_l1_None_2012-02-09_11-40-37/SparseSlowModel_patchsz032_N512_NN512_l2_l1_None.model'
    #model_name = 'SparseSlowModel_patchsz064_N1024_NN1024_l2_l1_None_2012-02-09_11-47-43/SparseSlowModel_patchsz064_N1024_NN1024_l2_l1_None.model'
    #model_name = 'SparseSlowModel_patchsz032_N512_NN512_l2_l1_None_2012-02-09_16-38-31/SparseSlowModel_patchsz032_N512_NN512_l2_l1_None.model'
    #model_name = 'SparseSlowModel_patchsz032_N512_NN512_l2_l1_None_2012-02-09_19-05-45/SparseSlowModel_patchsz032_N512_NN512_l2_l1_None.model'
    model_name = 'SparseSlowModel_patchsz048_N768_NN768_l2_l1_None_2012-02-09_19-07-07/SparseSlowModel_patchsz048_N768_NN768_l2_l1_None.model'
    #model_name = 'SparseSlowModel_patchsz064_N1024_NN1024_l2_l1_None_2012-02-09_19-08-50/SparseSlowModel_patchsz064_N1024_NN1024_l2_l1_None.model'
    model_name = 'SparseSlowModel_patchsz048_N1024_NN1024_l2_l1_None_2012-02-10_16-23-20/SparseSlowModel_patchsz048_N1024_NN1024_l2_l1_None.model'
    model_name = 'SparseSlowModel_patchsz048_N1024_NN1024_l2_subspacel1_None_2012-02-10_17-22-55/SparseSlowModel_patchsz048_N1024_NN1024_l2_subspacel1_None.model'

    fname = os.path.join(state_dir,model_name)
    m = SparseSlowModel()
    m.load(fname)

    display_final(m)

