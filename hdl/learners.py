import os
import numpy as np
from config import verbose, verbose_timing
from models import BaseModel
from time import time


class BaseLearner(object):

    def __init__(self,**kargs):

        self.model = kargs.get('model',BaseModel())
        self.save_every = kargs.get('save_every',10000)
        self.display_every = kargs.get('display_every',1000)
        self.iter   = 0

        self.datasource = kargs.get('datasource','berkeleysegmentation')
        self.batchsize = kargs.get('batchsize',128)

    def learn(self,iterations=1000):

        for i in range(iterations):
            pass
            # get a batch

    def crop_videos(self,batchsize):
        patch_sz = self.model.patch_sz

        rind = np.floor(self.nvideos*np.random.rand())
        rt = np.floor((self.videot-batchsize)*np.random.rand())
        ry = self.topmargin + np.floor((self.videoheight - patch_sz - self.BUFF - self.topmargin)*np.random.rand())
        rx = self.BUFF + np.floor((self.videowidth - patch_sz - 2*self.BUFF)*np.random.rand())
        return self.videos[rind,ry:ry+patch_sz,rx:rx+patch_sz,rt:rt+batchsize]

    def crop_single_video(self,video,batchsize,BUFF=0,topmargin=0):
        patch_sz = self.model.patch_sz
        videoheight, videowidth, videot = video.shape

        rt = np.floor((videot-batchsize)*np.random.rand())
        ry = topmargin + np.floor((videoheight - patch_sz - BUFF - topmargin)*np.random.rand())
        rx = BUFF + np.floor((videowidth - patch_sz - 2*BUFF)*np.random.rand())
        return video[ry:ry+patch_sz,rx:rx+patch_sz,rt:rt+batchsize]

    def crop_single_binoc_video(self,video,batchsize,BUFF=0,topmargin=0):
        patch_sz = self.model.patch_sz
        mse_reject = self.model.binoc_movie_mse_reject
        videot, videoc, videowidth, videoheight = video.shape

        rt = np.floor((videot-batchsize)*np.random.rand())
        ry = topmargin + np.floor((videoheight - patch_sz - BUFF - topmargin)*np.random.rand())
        rxl = BUFF + np.floor((videowidth - patch_sz - 2*BUFF - np.abs(self.binocular_offset))*np.random.rand())
        if self.binocular_offset < 0:
            rxl -= self.binocular_offset
        video_crop = np.zeros((batchsize,videoc,patch_sz,patch_sz))
        video_crop[:,:3,:,:] = video[rt:rt+batchsize,:3,rxl:rxl+patch_sz,ry:ry+patch_sz]
        rxr = rxl + self.binocular_offset
        video_crop[:,3:,:,:] = video[rt:rt+batchsize,3:,rxr:rxr+patch_sz,ry:ry+patch_sz]

        mse = np.mean((video_crop[:,3:,:,:].ravel() - video_crop[:,:3,:,:].ravel())**2)

        if mse_reject is None:
            return video_crop
        elif mse > mse_reject:
            return video_crop
        else:
            print 'reject video, mse = ', mse
            return None

    def get_databatch(self,batchsize=None,testing=False):
        if batchsize is None: batchsize = self.batchsize

        if self.datasource == 'berkeleysegmentation':
            patch_sz = self.model.patch_sz

            if not hasattr(self,'images'):
                from config import data_dir
                from scipy import misc
                if testing:
                    img_dir = os.path.join(data_dir,'BSR/BSDS500/data/images/test')
                    nimg = 200
                else:
                    img_dir = os.path.join(data_dir,'BSR/BSDS500/data/images/train')
                    nimg = 200
                img_files = [os.path.join(img_dir,img_name) for img_name in os.listdir(img_dir)]
                if len(img_files) != nimg:
                    raise IOError('image files missing - found %d of %d in %s'%(len(img_files),nimg,img_dir))
                self.images = np.zeros((481,321,nimg),dtype=np.uint8)
                for i,fname in enumerate(img_files):
                    a = misc.imread(fname,flatten=True)
                    if a.shape[0] == 321: a = a.T
                    self.images[:,:,i] = a

            batch = np.zeros((self.model.D,batchsize))
            for i in xrange(batchsize):
                x,y,t = self.images.shape
                rt = np.random.randint(t)
                rx = np.random.randint(x-patch_sz+1)
                ry = np.random.randint(y-patch_sz+1)
                batch[:,i] = self.images[rx:rx+patch_sz,ry:ry+patch_sz,rt].ravel()

            return batch

        elif self.datasource == 'vid075-chunks':

            if not hasattr(self,'videos'):
                from config import data_dir

                nvideos = 56
                videoheight = 128
                videowidth = 128
                videot = 64

                self.BUFF = 4
                self.topmargin = 15

                self.videos = np.zeros((nvideos, videoheight, videowidth, videot))
                for video_ind in range(1,nvideos+1):
                    video_fname = os.path.join(data_dir,'vid075-chunks','chunk' + str(video_ind))
                    video_cstring = np.fromfile(video_fname,dtype='>f4').reshape(videot,videowidth,videoheight).astype(np.double).tostring('F')
                    self.videos[video_ind-1,...] = np.fromstring(video_cstring).reshape(videoheight,videowidth,videot)

                self.nvideos = nvideos
                self.videoheight = videoheight
                self.videowidth = videowidth
                self.videot = videot

            if batchsize > self.videot:
                batch = np.zeros((self.model.D,batchsize))
                done = False
                batch_remaining = batchsize
                t0 = 0
                while not done:
                    tsz = min(self.videot,batch_remaining)
                    batch0 = self.crop_videos(tsz)
                    batch[:,t0:t0+tsz] = batch0.reshape(self.model.D,tsz)
                    t0 += tsz
                    batch_remaining -= tsz
                    if batch_remaining <= 0: done = True
            else:
                batch = self.crop_videos(batchsize).reshape(self.model.D,batchsize)
            return batch

        elif self.datasource == 'YouTubeFaces_aligned':
            import cPickle
            from scipy import misc
            from config import scratch_local_dir, scratch_dir, public_dir
            patch_sz = self.model.patch_sz

            if not hasattr(self,'YouTubeInfo'):
                pcrop = 80 # crop pixels from center
                newsize = int(np.ceil(patch_sz*1.5))
                TOTAL_FRAMES = 1000000

                cache_dir = os.path.join(scratch_dir,'hdl','YouTubeFaces/aligned_images_DB')
                cache_name = os.path.join(cache_dir,'newsize_%d_pcrop_%d_TOTAL_FRAMES_%d.npz'%(newsize,pcrop,TOTAL_FRAMES))
                if not os.path.exists(cache_name):
                    self.YouTubeInfo = {'videos':[],'video_weights':[]}
                    base_dir = os.path.join(scratch_local_dir,'YouTubeFaces/aligned_images_DB')
                    if not os.path.exists(base_dir): base_dir = os.path.join(public_dir,'YouTubeFaces/YouTubeFaces/aligned_images_DB')
                    identities = os.listdir(base_dir)
                    total_frames = 0
                    for identity in identities:
                        if total_frames > TOTAL_FRAMES: continue
                        video_numbers = os.listdir(os.path.join(base_dir,identity))
                        for video_number in video_numbers:
                            video_number_path = os.path.join(base_dir,identity,video_number)
                            frame_partial_names = os.listdir(video_number_path)
                            frame_partial_index = sorted([int(item.split('.')[1]) for item in frame_partial_names])
                            first_part, dummy, last_part = frame_partial_names[0].split('.')
                            frame_fnames = [os.path.join(video_number_path,first_part + '.' + str(img_index) + '.' + last_part) for img_index in frame_partial_index]
                            frames = len(frame_fnames)
                            video_array = np.zeros((newsize,newsize,frames),dtype=np.uint8)
                            for frame_index, frame_fname in enumerate(frame_fnames):
                                image = misc.imread(frame_fname,flatten=True)
                                sxy, szx = image.shape
                                cent = int(np.floor(np.float(szx)/2))
                                video_array[...,frame_index] = misc.imresize(image[(cent-pcrop):(cent+pcrop),(cent-pcrop):(cent+pcrop)],(newsize,newsize))
                            self.YouTubeInfo['videos'].append(video_array)
                            self.YouTubeInfo['video_weights'].append(frames)
                            total_frames += frames
                            print '\rtotal_frames loaded=%d'%total_frames
                    num_videos = len(self.YouTubeInfo['videos'])
                    video_weights = np.zeros((num_videos,))
                    for ind, val in enumerate(self.YouTubeInfo['video_weights']):
                        video_weights[ind] = float(val)/total_frames
                    self.YouTubeInfo['video_weights'] = video_weights
                    self.YouTubeInfo['num_videos'] = num_videos

                    print '\nSaving cache file:', cache_name, '...',
                    if not os.path.exists(cache_dir): os.makedirs(cache_dir)
                    with open(cache_name,'wb') as fh:
                        cPickle.dump(dict(YouTubeInfo=self.YouTubeInfo),fh)
                        print 'Done'
                else:
                    print 'Loading cache file:', cache_name, '...'
                    with open(cache_name,'rb') as fh:
                        ldict = cPickle.load(fh)
                        self.YouTubeInfo = ldict['YouTubeInfo']
                        print 'Done'

            video_index = np.where(np.random.multinomial(1,self.YouTubeInfo['video_weights']))[0][0]
            video_array = self.YouTubeInfo['videos'][video_index]
            num_frames = video_array.shape[2]

            if batchsize > num_frames:
                batch = np.zeros((self.model.D,batchsize))
                done = False
                batch_remaining = batchsize
                t0 = 0
                while not done:
                    video_index = np.where(np.random.multinomial(1,self.YouTubeInfo['video_weights']))[0][0]
                    video_array = self.YouTubeInfo['videos'][video_index]
                    num_frames = video_array.shape[2]

                    tsz = min(num_frames,batch_remaining)
                    batch0 = self.crop_single_video(video_array,tsz)
                    batch[:,t0:t0+tsz] = batch0.reshape(self.model.D,tsz)
                    t0 += tsz
                    batch_remaining -= tsz
                    if batch_remaining <= 0: done = True
            else:
                batch = self.crop_single_video(video_array,batchsize).reshape(self.model.D,batchsize)
            return batch.astype(np.single)

        elif self.datasource == 'YouTubeFaces_aligned_asymmetric':
            import cPickle
            from scipy import misc
            from config import scratch_local_dir, scratch_dir, public_dir
            patch_sz = self.model.patch_sz

            if not hasattr(self,'YouTubeInfo'):
                normcrop = 80
                xcrop = 60 # crop pixels from center
                ycrop = 100
                newxsize = int(np.ceil( float(xcrop)/normcrop * patch_sz*1.5) )
                newysize = int(np.ceil( float(ycrop)/normcrop * patch_sz*1.5) )
                TOTAL_FRAMES = 1000000

                cache_dir = os.path.join(scratch_dir,'hdl','YouTubeFaces/aligned_images_DB')
                cache_name = os.path.join(cache_dir,'newxsize_%d_newysize_%d_xcrop_%d_ycrop_%d_normcrop_%d_TOTAL_FRAMES_%d.npz'%(newxsize,newysize,xcrop,ycrop,normcrop,TOTAL_FRAMES))
                if not os.path.exists(cache_name):
                    self.YouTubeInfo = {'videos':[],'video_weights':[]}
                    base_dir = os.path.join(scratch_local_dir,'YouTubeFaces/aligned_images_DB')
                    if not os.path.exists(base_dir): base_dir = os.path.join(public_dir,'YouTubeFaces/YouTubeFaces/aligned_images_DB')
                    identities = os.listdir(base_dir)
                    total_frames = 0
                    for identity in identities:
                        if total_frames > TOTAL_FRAMES: continue
                        video_numbers = os.listdir(os.path.join(base_dir,identity))
                        for video_number in video_numbers:
                            video_number_path = os.path.join(base_dir,identity,video_number)
                            frame_partial_names = os.listdir(video_number_path)
                            frame_partial_index = sorted([int(item.split('.')[1]) for item in frame_partial_names])
                            first_part, dummy, last_part = frame_partial_names[0].split('.')
                            frame_fnames = [os.path.join(video_number_path,first_part + '.' + str(img_index) + '.' + last_part) for img_index in frame_partial_index]
                            frames = len(frame_fnames)
                            video_array = np.zeros((newysize,newxsize,frames),dtype=np.uint8)
                            for frame_index, frame_fname in enumerate(frame_fnames):
                                image = misc.imread(frame_fname,flatten=True)
                                sxy, szx = image.shape
                                cent = int(np.floor(np.float(szx)/2))
                                video_array[...,frame_index] = misc.imresize(image[(cent-ycrop):(cent+ycrop),(cent-xcrop):(cent+xcrop)],(newysize,newxsize))
                            self.YouTubeInfo['videos'].append(video_array)
                            self.YouTubeInfo['video_weights'].append(frames)
                            total_frames += frames
                            print '\rtotal_frames loaded=%d'%total_frames
                    num_videos = len(self.YouTubeInfo['videos'])
                    video_weights = np.zeros((num_videos,))
                    for ind, val in enumerate(self.YouTubeInfo['video_weights']):
                        video_weights[ind] = float(val)/total_frames
                    self.YouTubeInfo['video_weights'] = video_weights
                    self.YouTubeInfo['num_videos'] = num_videos

                    print '\nSaving cache file:', cache_name, '...',
                    if not os.path.exists(cache_dir): os.makedirs(cache_dir)
                    with open(cache_name,'wb') as fh:
                        cPickle.dump(dict(YouTubeInfo=self.YouTubeInfo),fh)
                        print 'Done'
                else:
                    print 'Loading cache file:', cache_name, '...'
                    with open(cache_name,'rb') as fh:
                        ldict = cPickle.load(fh)
                        self.YouTubeInfo = ldict['YouTubeInfo']
                        print 'Done'

            video_index = np.where(np.random.multinomial(1,self.YouTubeInfo['video_weights']))[0][0]
            video_array = self.YouTubeInfo['videos'][video_index]
            num_frames = video_array.shape[2]

            if batchsize > num_frames:
                batch = np.zeros((self.model.D,batchsize))
                done = False
                batch_remaining = batchsize
                t0 = 0
                while not done:
                    video_index = np.where(np.random.multinomial(1,self.YouTubeInfo['video_weights']))[0][0]
                    video_array = self.YouTubeInfo['videos'][video_index]
                    num_frames = video_array.shape[2]

                    tsz = min(num_frames,batch_remaining)
                    batch0 = self.crop_single_video(video_array,tsz)
                    batch[:,t0:t0+tsz] = batch0.reshape(self.model.D,tsz)
                    t0 += tsz
                    batch_remaining -= tsz
                    if batch_remaining <= 0: done = True
            else:
                batch = self.crop_single_video(video_array,batchsize).reshape(self.model.D,batchsize)
            return batch.astype(np.single)

        elif self.datasource == 'TorontoFaces48':
            from scipy.io import loadmat
            from config import public_dir
            patch_sz = self.model.patch_sz

            if not hasattr(self,'images'):
                loadfile = os.path.join(public_dir,'TorontoFaces','TFD_ranzato_48x48.mat')
                mfile = loadmat(loadfile)
                self.images = mfile['images']

            batch = np.zeros((self.model.D,batchsize))
            for i in xrange(batchsize):
                t,x,y = self.images.shape
                rt = np.random.randint(t)
                rx = np.random.randint(x-patch_sz+1)
                ry = np.random.randint(y-patch_sz+1)
                batch[:,i] = self.images[rt,rx:rx+patch_sz,ry:ry+patch_sz].ravel()

            return batch

        elif self.datasource == 'TorontoFaces96':
            from scipy.io import loadmat
            from config import public_dir
            patch_sz = self.model.patch_sz

            if not hasattr(self,'images'):
                loadfile = os.path.join(public_dir,'TorontoFaces','TFD_ranzato_96x96.mat')
                mfile = loadmat(loadfile)
                self.images = mfile['images']

            batch = np.zeros((self.model.D,batchsize))
            for i in xrange(batchsize):
                t,x,y = self.images.shape
                rt = np.random.randint(t)
                rx = np.random.randint(x-patch_sz+1)
                ry = np.random.randint(y-patch_sz+1)
                batch[:,i] = self.images[rt,rx:rx+patch_sz,ry:ry+patch_sz].ravel()

            return batch

        elif self.datasource == '3Dvideo_color':
            from config import data_dir
            import tables

            if not hasattr(self,'videos'):
                print 'Loading 3Dvideo_color'
                self.video_buffer = 100
                self.video_reload = 200
                self.video_counter = 0
                self.binocular_offset = +13

                videopath = os.path.join(data_dir,'3Dvideo','processed')
                self.video_list = [os.path.join(videopath,item) for item in filter(lambda x: x.count('.h5'),os.listdir(videopath))]

                self.videos = []
                for video_ind in range(self.video_buffer):
                    rind = np.random.randint(len(self.video_list))
                    h5file = tables.openFile(self.video_list[rind])
                    binocvideo = np.hstack((h5file.root.left.read(),h5file.root.right.read()))
                    self.videos.append(binocvideo)
                    h5file.close()

            self.video_counter += 1
            if not self.video_counter%self.video_reload:
                rind = np.random.randint(len(self.video_list))
                print 'Load video', self.video_list[rind]
                h5file = tables.openFile(self.video_list[rind])
                binocvideo = np.hstack((h5file.root.left.read(),h5file.root.right.read()))
                h5file.close()
                self.videos.pop()
                self.videos = [binocvideo] + self.videos

            vidind = np.random.randint(self.video_buffer)
            video = self.videos[vidind]

            if batchsize > video.shape[0]:
                batch = np.zeros((self.model.D,batchsize))
                done = False
                batch_remaining = batchsize
                t0 = 0
                while not done:
                    batch0 = None
                    tsz = 0
                    while batch0 is None:
                        vidind = np.random.randint(self.video_buffer)
                        video = self.videos[vidind]

                        tsz = min(video.shape[0],batch_remaining)
                        batch0 = self.crop_single_binoc_video(video,tsz)

                    batch[:,t0:t0+tsz] = batch0.reshape(tsz,self.model.D).T
                    t0 += tsz
                    batch_remaining -= tsz
                    if batch_remaining <= 0: done = True
            else:
                batch0 = None
                while batch0 is None:
                    batch0 = self.crop_single_binoc_video(video,batchsize)
                batch = np.double(batch0.reshape(batchsize,self.model.D).T)
            return batch

        elif self.datasource == 'randn':
            return np.random.randn(self.model.D,batchsize)
        else:
            assert NotImplementedError, self.datasource

class SGD(BaseLearner):

    def __init__(self, **kargs):
        super(SGD, self).__init__(**kargs)

        self.eta = kargs.get('eta', .0001)

        self.adapt_eta = kargs.get('adapt_eta', True)
        self.eta_target_maxupdate = kargs.get('eta_target_maxupate', .05)
        self.eta_adapt_upfactor = kargs.get('eta_adapt_upfactor', 1.01)
        self.eta_adapt_downfactor = kargs.get('eta_adapt_downfactor', .95)

        if kargs.has_key('get_databatch'):
            self.get_databatch = kargs['get_databatch']

    def _adapt_eta(self,max_update):

        if not self.adapt_eta: return

        if max_update > self.eta_target_maxupdate:
            self.eta *= self.eta_adapt_downfactor
            #if verbose: print 'max_update (%2.2e) above target (%2.2e), new eta = %2.2e'%(max_update, self.eta_target_maxupdate, self.eta)
        else:
            self.eta *= self.eta_adapt_upfactor

    def change_target(self,fraction):
        """change the eta_target_maxupdate by fraction:
        eta_target_maxupdate *= fraction"""
        self.eta_target_maxupdate *= fraction

    def learn(self,iterations=1000):

        for it in range(iterations):

            t0 = time()
            x = self.get_databatch()
            if verbose_timing: print 'self.get_databatch() time %f'%(time() - t0)

            t0 = time()
            grad_dict = self.model.gradient(x)
            if verbose_timing: print 'self.model.gradient time %f'%(time() - t0)

            t0 = time()
            for key in grad_dict:
                grad_dict[key] *= self.eta
            max_update = self.model.update_model(grad_dict)
            self._adapt_eta(max_update)
            if verbose_timing: print 'self.model.update_model time %f'%(time() - t0)

            self.iter += 1

            if not self.iter%self.save_every:
                print 'Saving model at iteration %d'%self.iter
                self.model.save()

            if not self.iter%self.display_every:
                print 'Displaying model at iteration %d'%self.iter
                self.model.display(save_string='learning_update=%07d'%self.iter)

            if not it%100: print 'Update %d of %d, total updates %d'%(it, iterations, self.iter)


class SGD_layer(SGD):

    def __init__(self,**kargs):

        super(SGD_layer,self).__init__(**kargs)

        self.model_sequence = kargs['model_sequence']
        self.layer_params = kargs['layer_params']

    def get_databatch(self,batchsize=None,testing=False):
        batch = super(SGD_layer,self).get_databatch(batchsize=batchsize,testing=testing)

        for mind, m in enumerate(self.model_sequence):
            if 'output_function' in self.layer_params[mind]:
                batch = m.output(m.preprocess(batch),self.layer_params[mind]['output_function'])
            else:
                batch = m.output(m.preprocess(batch))

        return batch


