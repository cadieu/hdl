import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.misc.pilutil import toimage
import hdl
reload(hdl)

from hdl.models import SparseSlowModel, BinocColorModel
from hdl.config import state_dir
from hdl.learners import BaseLearner
from hdl.display import display_patches, display_binoc_color_patches


def explore_marginals_color():
    small_value = .001
    phase_small_value = np.exp(-3.)

    print 'Loading model'
    model_name = 'BinocColorModel_patchsz016_N512_NN512_l2_subspacel1_dist_2012-03-13_15-24-27/BinocColorModel_patchsz016_N512_NN512_l2_subspacel1_dist.model'
    model_name = 'BinocColorModel_patchsz016_N512_NN512_l2_subspacel1_dist_2012-03-15_15-56-39/BinocColorModel_patchsz016_N512_NN512_l2_subspacel1_dist.model'
    datasource = '3Dvideo_color'

    fname = os.path.join(state_dir,model_name)
    m = BinocColorModel()
    m.load(fname)
    #m.inference_params['u_init_method'] = 'proj'
    #m.inference_params['FISTAargs']['maxiter'] = 40
    #m.inference_params['FISTAargs']['maxline'] = 40
    #m.inference_params['FISTAargs']['errthres'] = 1e-8
    #m.inference_params['FISTAargs']['verbose'] = True
    m.lam_sparse.set_value(getattr(np,hdl.models.theano.config.floatX)(m.lam_sparse.get_value()*.1))
    #m.lam_sparse.set_value(getattr(np,hdl.models.theano.config.floatX)(.2))
    m.reset_functions()
    l = BaseLearner(datasource=datasource,model=m)
    l.get_databatch()

    from hdl.config import fig_dir
    savepath = os.path.join(fig_dir,m.model_name + '_' + m.tstring,'explore_distribution')
    if not os.path.isdir(savepath): os.makedirs(savepath)

    print 'Get data'
    display_batches = True
    batch_size = 32
    batches = 100
    u_list = []
    snr_list = []
    A = m.A.get_value()
    for bind in range(batches):
        batch = l.get_databatch(batch_size)
        u_batch = m.inferlatent(m.preprocess(batch.copy()))
        batchhat = np.dot(m.dewhitenmatrix,np.dot(A,u_batch)) + m.inputmean
        error = batch - batchhat
        snr = -10.*np.log10(np.var(error,0)/np.var(batch,0))
        u_list.append(u_batch)
        snr_list.append(snr)
        print '%d->%d'%(bind*batch_size,(bind+1)*batch_size)
        print 'Max SNR: %f, Min SNR: %f'%(snr.max(), snr.min())
        #print '%2.2e %2.2e %2.2e'%(error[:,0].mean(), batch[:,0].mean(), batchhat[:,0].mean())

        if display_batches:
            if not os.path.isdir(os.path.join(savepath,'batch_rec')): os.makedirs(os.path.join(savepath,'batch_rec'))
            arr = display_binoc_color_patches(batch,m.patch_sz,1,normalize=False)
            fname = os.path.join(savepath,'batch_rec','batch_%d_%d.png'%(bind,batch_size))
            toimage(arr).save(fname)
            arr = display_binoc_color_patches(batchhat,m.patch_sz,1,normalize=False)
            fname = os.path.join(savepath,'batch_rec','batch_%d_%d_rec.png'%(bind,batch_size))
            toimage(arr).save(fname)

    import sys
    sys.exit()
    u = np.hstack(u_list)
    snr = np.hstack(snr_list)
    amp = np.sqrt(u[::2,:]**2 + u[1::2,:]**2)
    phase = np.arctan2(u[::2,:], u[1::2,:])
    print 'num indices above %f, %d'%(small_value,np.sum(amp>small_value))
    print 'num indices above %f, %d'%(small_value,np.sum(amp>phase_small_value))

    print 'Save distributions...'
    pinds = range(amp.shape[0])

    plt.figure(1)
    plt.clf()
    snr_valid = np.isreal(snr) & np.isfinite(snr)
    plt.hist(snr[snr_valid].ravel(),101)
    plt.title('SNR of reconstructions')
    fname = os.path.join(savepath,'SNR_rec_lam_%2.2e_uinit_%s.png'%(m.lam_sparse.get_value(),m.inference_params['u_init_method']))
    plt.savefig(fname)

    plt.figure(1)
    plt.clf()
    plt.subplot(1,2,1)
    plt.hist(amp.ravel(),101)
    plt.title('amp values')
    plt.subplot(1,2,2)
    plt.hist(np.log(amp.ravel()[amp.ravel() > 0.]),101)
    plt.title('log(amp) values')
    fname = os.path.join(savepath,'amp_lam_%2.2e_uinit_%s.png'%(m.lam_sparse.get_value(),m.inference_params['u_init_method']))
    plt.savefig(fname)

    savedir = os.path.join(savepath,'marginals')
    if not os.path.isdir(savedir): os.makedirs(savedir)

    for iind, pind in enumerate(pinds):

        plt.figure(1)
        plt.clf()

        valind = amp[pind,:] > small_value

        plt.subplot(1,3,1)
        hval = np.max(amp[pind,:])*.1
        Htemp, xedges, yedges = np.histogram2d(amp[pind,valind]*np.cos(phase[pind,valind]),amp[pind,valind]*np.sin(phase[pind,valind]),bins=np.arange(-hval,hval,2.*hval/31))
        Hzero = Htemp == 0
        if np.sum(Hzero):
            Hmin = Htemp[Hzero].min()
            H = np.zeros_like(Htemp)
            H[Hzero] = np.log(Hmin)
            H[~Hzero] = np.log(Htemp[~Hzero])
        else:
            H = np.log(Htemp)
        plt.imshow(H,interpolation='nearest')
        plt.title('real vs. imag %d'%pind)

        plt.subplot(1,3,2)
        plt.hist(np.log(amp[pind,valind]),bins=16)
        plt.title('amp %d dist'%pind)

        plt.subplot(1,3,3)
        valind = amp[pind,:] > phase_small_value
        plt.hist(phase[pind,valind],bins=16)
        plt.title('phase %d dist'%pind)

        fname = os.path.join(savedir, 'dist_%d.png'%pind)
        plt.savefig(fname)

        print 'Done with %d, %d/%d'%(pind,iind,len(pinds))


def explore_pairwise_ampphase():

    small_value = .001
    phase_small_value = np.exp(-1.)

    print 'Loading model'
    # faces YouTube
    model_name = 'SparseSlowModel_patchsz048_N1024_NN1024_l2_subspacel1_None_2012-02-21_12-37-25/SparseSlowModel_patchsz048_N1024_NN1024_l2_subspacel1_None.model'
    datasource = 'YouTubeFaces_aligned'

    # faces TFD
    #model_name = 'SparseSlowModel_patchsz048_N512_NN512_l2_subspacel1_None_2012-03-05_11-42-48/SparseSlowModel_patchsz048_N512_NN512_l2_subspacel1_None.model'
    datasource = 'TorontoFaces48'

    fname = os.path.join(state_dir,model_name)
    m = SparseSlowModel()
    m.load(fname)
    #m.inference_params['u_init_method'] = 'proj'
    #m.inference_params['FISTAargs']['maxiter'] = 40
    #m.inference_params['FISTAargs']['maxline'] = 40
    #m.inference_params['FISTAargs']['errthres'] = 1e-8
    #m.inference_params['FISTAargs']['verbose'] = True
    #m.lam_sparse.set_value(getattr(np,hdl.models.theano.config.floatX)(m.lam_sparse.get_value()*.1))
    #m.lam_sparse.set_value(getattr(np,hdl.models.theano.config.floatX)(.2))
    #m.reset_functions()
    l = BaseLearner(datasource=datasource,model=m)
    l.get_databatch()

    from hdl.config import fig_dir
    savepath = os.path.join(fig_dir,m.model_name + '_' + m.tstring,'explore_distribution')
    if not os.path.isdir(savepath): os.makedirs(savepath)

    print 'Get data'
    display_batches = False
    batch_size = 1000
    num_images = l.images.shape[0]
    batches = int(np.ceil(num_images//batch_size))
    u_list = []
    snr_list = []
    A = m.A.get_value()
    for bind in range(batches):
        batch = l.images[bind*batch_size:(bind+1)*batch_size,:,:]
        batch = np.double(batch.reshape((batch.shape[0],batch.shape[1]**2)).T)
        u_batch = m.inferlatent(m.preprocess(batch.copy()))
        batchhat = np.dot(m.dewhitenmatrix,np.dot(A,u_batch)) + m.inputmean
        error = batch - batchhat
        snr = -10.*np.log10(np.var(error,0)/np.var(batch,0))
        u_list.append(u_batch)
        snr_list.append(snr)
        print '%d->%d'%(bind*batch_size,(bind+1)*batch_size)

        if display_batches:
            arr = display_patches(batch-127.5,m.patch_sz,1,normalize=False)
            fname = os.path.join(savepath,'batch_%d_%d.png'%(bind,batch_size))
            toimage(np.floor(.5*(arr+1)*255)).save(fname)
            arr = display_patches(batchhat-127.5,m.patch_sz,1,normalize=False)
            fname = os.path.join(savepath,'batch_rec_%d_%d.png'%(bind,batch_size))
            toimage(np.floor(.5*(arr+1)*255)).save(fname)

    u = np.hstack(u_list)
    snr = np.hstack(snr_list)
    amp = np.sqrt(u[::2,:]**2 + u[1::2,:]**2)
    phase = np.arctan2(u[::2,:], u[1::2,:])
    print 'num indices above %f, %d'%(small_value,np.sum(amp>small_value))
    print 'num indices above %f, %d'%(small_value,np.sum(amp>phase_small_value))

    print 'Save distributions...'
    pinds = range(10) + [10,12,13,48,49] + range(100,130)

    plt.figure(1)
    plt.clf()
    plt.hist(snr.ravel(),101)
    plt.title('SNR of reconstructions')
    fname = os.path.join(savepath,'SNR_rec_lam_%2.2e_uinit_%s.png'%(m.lam_sparse.get_value(),m.inference_params['u_init_method']))
    plt.savefig(fname)

    plt.figure(1)
    plt.clf()
    plt.subplot(1,2,1)
    plt.hist(amp.ravel(),101)
    plt.title('amp values')
    plt.subplot(1,2,2)
    plt.hist(np.log(amp.ravel()[amp.ravel() > 0.]),101)
    plt.title('log(amp) values')
    fname = os.path.join(savepath,'amp_lam_%2.2e_uinit_%s.png'%(m.lam_sparse.get_value(),m.inference_params['u_init_method']))
    plt.savefig(fname)

    for iind, pind in enumerate(pinds):
        savedir = os.path.join(savepath,str(pind))
        if not os.path.isdir(savedir): os.makedirs(savedir)

        for m in range(amp.shape[0]):
            if m == pind: continue
            #if m > 10: continue
            plt.figure(1)
            plt.clf()

            valind = (amp[pind,:] > small_value) & (amp[m,:] > small_value)

            plt.subplot(1,2,1)
            H, xedges, yedges = np.histogram2d(np.log(amp[pind,valind]),np.log(amp[m,valind]),bins=16)
            plt.imshow(H,interpolation='nearest')
            plt.title('amp %d amp %d'%(pind,m))

            valind = (amp[pind,:] > phase_small_value) & (amp[m,:] > phase_small_value)

            plt.subplot(1,2,2)
            H, xedges, yedges = np.histogram2d(phase[pind,valind],phase[m,valind],bins=16)
            plt.imshow(H,interpolation='nearest')
            plt.title('phase %d phase %d'%(pind,m))

            fname = os.path.join(savedir, 'dist_%d_%d.png'%(pind,m))
            plt.savefig(fname)
        print 'Done with %d, %d/%d'%(pind,iind,len(pinds))


if __name__ == '__main__':
    explore_marginals_color()
