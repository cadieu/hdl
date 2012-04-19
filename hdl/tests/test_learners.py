import os
import numpy as np
import hdl
reload(hdl)

from hdl.models import SparseSlowModel
from hdl.learners import SGD

def test_vid075():
    from matplotlib import pyplot as plt

    test_name = 'vid075'
    l = SGD(model=SparseSlowModel(patch_sz=20,N=400,T=64),datasource='vid075-chunks',batchsize=64)

    batchsize = 64
    databatch = l.get_databatch(batchsize)

    batchsize = 1000
    databatch = l.get_databatch(batchsize)

    from hdl.config import tests_dir, tstring
    savepath = os.path.join(tests_dir,test_name,tstring())
    if not os.path.isdir(savepath): os.makedirs(savepath)

    vidind = np.floor(np.random.rand()*l.nvideos)
    hval = np.abs(l.videos[vidind,...]).max()
    for tt in range(l.videot):

        plt.figure(1)
        plt.clf()
        plt.imshow(l.videos[vidind,:,:,tt],cmap=plt.cm.gray,vmin=-hval,vmax=hval,interpolation='nearest')
        fname = os.path.join(savepath, 'vid_' + str(vidind) + '_frame_'+ str(tt) + '.png')
        plt.savefig(fname)

    batchsize = 64
    databatch = l.get_databatch(batchsize)
    hval = np.abs(databatch).max()
    for tt in range(batchsize):

        plt.figure(1)
        plt.clf()
        plt.imshow(databatch[...,tt].reshape(l.model.patch_sz,l.model.patch_sz),cmap=plt.cm.gray,vmin=-hval,vmax=hval,interpolation='nearest')
        fname = os.path.join(savepath, 'databatch_' + str(vidind) + '_frame_'+ str(tt) + '.png')
        plt.savefig(fname)

def test_YouTubeFaces():
    from matplotlib import pyplot as plt

    test_name = 'YouTubeFaces'
    l = SGD(model=SparseSlowModel(patch_sz=48,N=400,T=64),datasource='YouTubeFaces_aligned',batchsize=64)

    batchsize = 64
    databatch = l.get_databatch(batchsize)

    batchsize = 1000
    databatch = l.get_databatch(batchsize)

    from hdl.config import tests_dir, tstring
    savepath = os.path.join(tests_dir,test_name,tstring())
    if not os.path.isdir(savepath): os.makedirs(savepath)

    vidind = int(np.floor(np.random.rand()*l.YouTubeInfo['num_videos']))
    video = l.YouTubeInfo['videos'][vidind]
    hval = np.abs(video).max()
    for tt in range(video.shape[2]):

        plt.figure(1)
        plt.clf()
        plt.imshow(video[:,:,tt],cmap=plt.cm.gray,vmin=-hval,vmax=hval,interpolation='nearest')
        fname = os.path.join(savepath, 'vid_' + str(vidind) + '_frame_%04d'%tt + '.png')
        plt.savefig(fname)

    batchsize = 64
    databatch = l.get_databatch(batchsize)
    hval = np.abs(databatch).max()
    for tt in range(batchsize):

        plt.figure(1)
        plt.clf()
        plt.imshow(databatch[...,tt].reshape(l.model.patch_sz,l.model.patch_sz),cmap=plt.cm.gray,vmin=-hval,vmax=hval,interpolation='nearest')
        fname = os.path.join(savepath, 'databatch_frame_%04d'%tt + '.png')
        plt.savefig(fname)

    # test multiple loads
    for tt in range(100):
        databatch = l.get_databatch(batchsize)

def test_3Dvideo():
    from matplotlib import pyplot as plt

    test_name = '3Dvideo'
    psz = 16
    l = SGD(model=SparseSlowModel(patch_sz=psz,D=6*psz*psz,N=400,T=48),datasource='3Dvideo_color',batchsize=48)

    batchsize = 48
    databatch = l.get_databatch(batchsize)

    batchsize = 1000
    databatch = l.get_databatch(batchsize)

    from hdl.config import tests_dir, tstring
    savepath = os.path.join(tests_dir,test_name,tstring())
    if not os.path.isdir(savepath): os.makedirs(savepath)

    vidind = np.random.randint(l.video_buffer)
    video = l.videos[vidind]
    for tt in range(video.shape[0]):

        plt.figure(1)
        plt.clf()
        plt.subplot(1,2,1)
        frame = np.uint8(video[tt,:3,:,:].T)
        plt.imshow(frame,interpolation='nearest')
        plt.subplot(1,2,2)
        frame = np.uint8(video[tt,3:,:,:].T)
        plt.imshow(frame,interpolation='nearest')
        fname = os.path.join(savepath, 'vid_' + str(vidind) + '_frame_%04d'%tt + '.png')
        plt.savefig(fname)

    batchsize = 48
    databatch = l.get_databatch(batchsize)
    for tt in range(batchsize):

        plt.figure(1)
        plt.clf()
        plt.subplot(1,2,1)
        frame_both = np.uint8(databatch[...,tt].reshape(6,l.model.patch_sz,l.model.patch_sz)).T
        frame = frame_both[...,:3]
        plt.imshow(frame,interpolation='nearest')
        plt.subplot(1,2,2)
        frame = frame_both[...,3:]
        plt.imshow(frame,interpolation='nearest')
        fname = os.path.join(savepath, 'databatch_frame_%04d'%tt + '.png')
        plt.savefig(fname)

    # test multiple loads
    for tt in range(1000):
        if not tt%100: print tt,
        databatch = l.get_databatch(batchsize)

if __name__ == '__main__':
    #test_YouTubeFaces()
    test_3Dvideo()