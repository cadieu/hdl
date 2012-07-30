import numpy as np

import theano
from theano import tensor as T
from theano import function, Param, shared, In, Out, sandbox
from theano.tensor.nnet import conv2d

from scipy.signal import correlate2d, convolve2d

def test_conv():

    imshp = (1,1,20,20)
    kshp = (10,1,10,10)

    theano_convolve2d = get_theano_convolve2d(imshp,kshp)

    image = np.random.randn(*imshp)
    kernel = np.random.randn(*kshp)

    output_sz = (imshp[0],kshp[0],imshp[2] - kshp[2] + 1, imshp[3] - kshp[3] + 1)
    scipy_output = np.zeros(output_sz)
    for im_i in range(imshp[0]):
        for k_i in range(kshp[0]):
            for im_j in range(imshp[1]):
                scipy_output[im_i,k_i,:,:] += convolve2d(np.squeeze(image[im_i,im_j,:,:]),np.squeeze(kernel[k_i,im_j,:,:]),mode='valid')

    theano_output = theano_convolve2d(image,kernel)

    print 'scipy:', scipy_output.shape
    print 'theano:', theano_output.shape

    np.testing.assert_allclose(scipy_output,theano_output)


def get_theano_convolve2d(imshp,kshp):
    image = T.tensor4(dtype=theano.config.floatX)
    kernel = T.tensor4(dtype=theano.config.floatX)

    output = conv2d(image,kernel,image_shape=imshp,filter_shape=kshp)

    theano_convolve2d = function(inputs=[image,kernel],outputs=output)

    return theano_convolve2d


def test_corr():

    #imshp = (3,2,20,20) # num images, channels, szy, szx
    kshp = (10,2,5,10) # features, channels, szy, szx
    featshp = (3,10,11,11) # num images, features, szy, szx

    theano_correlate2d = get_theano_correlate2d(kshp=kshp,featshp=featshp)

    features = np.random.randn(*featshp)
    kernel = np.random.randn(*kshp)

    output_sz = (featshp[0], kshp[1], kshp[2] + featshp[2] - 1, kshp[3] + featshp[3] - 1)

    scipy_output = np.zeros(output_sz)
    for im_i in range(featshp[0]):
        for im_j in range(kshp[1]):
            for k_i in range(kshp[0]):
                scipy_output[im_i,im_j,:,:] += correlate2d(np.squeeze(features[im_i,k_i,:,:]),np.squeeze(kernel[k_i,im_j,:,:]),mode='full')

    theano_output = theano_correlate2d(features,kernel)

    print 'scipy:', scipy_output.shape
    print 'theano:', theano_output.shape

    np.testing.assert_allclose(scipy_output,theano_output)

def get_theano_correlate2d(kshp,featshp):

    features = T.tensor4(dtype=theano.config.floatX)
    kernel = T.tensor4(dtype=theano.config.floatX)

    # Need to transpose first two dimensions of kernel, and reverse index kernel image dims (for correlation)
    kernel_rotated = T.transpose(kernel[:,:,::-1,::-1],axes=[1,0,2,3])

    image = conv2d(features,kernel_rotated,border_mode='full')

    gen_image = function(inputs=[features,kernel],outputs=image)

    return gen_image

def test_derivative():

    imshp = (3,2,20,20) # num images, channels, szy, szx
    kshp = (10,2,5,5) # features, channels, szy, szx
    featshp = (imshp[0],kshp[0],imshp[2] - kshp[2] + 1,imshp[3] - kshp[3] + 1) # num images, features, szy, szx

    feature_derivative = theano_derivative()

    features = np.random.randn(*featshp)
    kernel = np.random.randn(*kshp)
    image = np.random.randn(*imshp)

    T_derivative = feature_derivative(image,features,kernel)
    print 'derivative of features:', T_derivative.shape, ', expected:', featshp

    def scipy_feature_derivative(image,features,kernel):

        scipy_image_estimate = scipy_correlate4d(features,kernel)
        scipy_error = image-scipy_image_estimate
        scipy_derivative = -scipy_convolve4d(scipy_error,kernel)
        return scipy_derivative

    scipy_derivative = scipy_feature_derivative(image,features,kernel)

    print 'scipy:', scipy_derivative.shape
    print 'theano:', T_derivative.shape

    np.testing.assert_allclose(scipy_derivative,T_derivative)

    import IPython; ipshell = IPython.embed; ipshell(banner1='ipshell')

def theano_derivative():

    features = T.tensor4(dtype=theano.config.floatX)
    kernel = T.tensor4(dtype=theano.config.floatX)
    image = T.tensor4(dtype=theano.config.floatX)

    # Need to transpose first two dimensions of kernel, and reverse index kernel image dims (for correlation)
    kernel_rotated = T.transpose(kernel[:,:,::-1,::-1],axes=[1,0,2,3])

    image_estimate = conv2d(features,kernel_rotated,border_mode='full')
    mse = .5*T.sum((image - image_estimate)**2)
    feature_grad = T.grad(mse,features)

    return function(inputs=[image,features,kernel],outputs=feature_grad)


def scipy_convolve4d(image,kernel):

    imshp = image.shape
    kshp = kernel.shape
    featshp = (imshp[0],kshp[0],imshp[2] - kshp[2] + 1,imshp[3] - kshp[3] + 1) # num images, features, szy, szx

    scipy_output = np.zeros(featshp)
    for im_i in range(imshp[0]):
        for k_i in range(kshp[0]):
            for im_j in range(imshp[1]):
                scipy_output[im_i,k_i,:,:] += convolve2d(np.squeeze(image[im_i,im_j,:,:]),np.squeeze(kernel[k_i,im_j,:,:]),mode='valid')

    return scipy_output

def scipy_correlate4d(features,kernel):

    featshp = features.shape
    kshp = kernel.shape

    output_sz = (featshp[0], kshp[1], kshp[2] + featshp[2] - 1, kshp[3] + featshp[3] - 1)

    scipy_output = np.zeros(output_sz)
    for im_i in range(featshp[0]):
        for im_j in range(kshp[1]):
            for k_i in range(kshp[0]):
                scipy_output[im_i,im_j,:,:] += correlate2d(np.squeeze(features[im_i,k_i,:,:]),np.squeeze(kernel[k_i,im_j,:,:]),mode='full')

    return scipy_output

from hdl.models import BaseModel
import os
from copy import copy
import numpy as np
from hdl.fista import Fista
import theano

from hdl.config import state_dir, verbose, tstring, verbose_timing
from time import time
from scipy import ndimage

from matplotlib import pyplot as plt

class ConvWhitenInputModel(BaseModel):
    _params = copy(BaseModel._params)
    _params.extend(['imshp','nchannels'])
    _params.extend(['M','whiten','convinputmean','convwhitenfilter','convdewhitenfilter','convwhitenfiltershp','inputmean','whitenmatrix', 'dewhitenmatrix','zerophasewhitenmatrix','zerophasedewhitenmatrix','num_eigs','perc_var'])

    def __init__(self, **kargs):
        super(ConvWhitenInputModel, self).__init__(**kargs)

        self.whiten = kargs.get('whiten',True)

        self.imshp = kargs.get('imshp',(32,1,32,32))
        self.nchannels = self.imshp[1]
        self.patch_sz = self.imshp[2]
        assert self.imshp[2] == self.imshp[3], 'only supports square patches now'

        if self.whiten:

            self.convinputmean = None
            self.convwhitenfilter = None
            self.convwhitenfiltershp = kargs.get('convwhitenfiltershp',(5,5))
            if len(self.convwhitenfiltershp) == 2:
                self.convwhitenfiltershp = (self.nchannels, self.nchannels, self.convwhitenfiltershp[0], self.convwhitenfiltershp[1])
            assert self.convwhitenfiltershp[2]%2, 'Whitening convolution filter should be odd (to pick center tap)'
            assert self.convwhitenfiltershp[3]%2, 'Whitening convolution filter should be odd (to pick center tap)'

            self.inputmean = None
            self.whitenmatrix = None
            self.dewhitenmatrix = None
            self.zerophasewhitenmatrix = None
            self.num_eigs = kargs.get('num_eigs',None)
            self.perc_var = kargs.get('perc_var', 99.)
            self.M = None
        else:
            self.M = self.D

        self.D = np.prod(self.imshp[1:])

    def _reshape_input(self,patches):

        imshp = (patches.shape[1],self.imshp[1],self.imshp[2],self.imshp[3])
        return np.reshape(np.transpose(patches),imshp)

    def _reshape_output(self,output):

        shp = output.shape
        return np.transpose(np.reshape(output,(shp[0],np.prod(shp[1:]))))

    def _scipy_convolve4d(self,image,kernel,mode='valid'):

        imshp = image.shape
        kshp = kernel.shape
        if mode=='valid':
            featshp = (imshp[0],kshp[0],imshp[2] - kshp[2] + 1,imshp[3] - kshp[3] + 1) # num images, features, szy, szx
        elif mode == 'same':
            featshp = (imshp[0],kshp[0],imshp[2],imshp[3]) # num images, features, szy, szx
        elif mode == 'full':
            featshp = (imshp[0],kshp[0],imshp[2] + kshp[2] - 1,imshp[3] + kshp[3] - 1) # num images, features, szy, szx
        else:
            raise NotImplemented, 'Unkonwn mode %s'%mode

        scipy_output = np.zeros(featshp)
        for im_i in range(imshp[0]):
            for k_i in range(kshp[0]):
                for im_j in range(imshp[1]):
                    scipy_output[im_i,k_i,:,:] += convolve2d(np.squeeze(image[im_i,im_j,:,:]),np.squeeze(kernel[k_i,im_j,:,:]),mode=mode)

        return scipy_output

    def learn_whitening(self,patches):

        from hdl.utils import whiten_var

        imshp = (patches.shape[1],self.imshp[1],self.imshp[2],self.imshp[3])
        image = self._reshape_input(patches)

        szx,szy = self.convwhitenfiltershp[2:]
        list_patches = []
        for imind in range(imshp[0]):
            for yoffset in range(imshp[2] - szy + 1):
                for xoffset in range(imshp[3] - szx + 1):
                    list_patches.append(image[imind,:,yoffset:yoffset+szy,xoffset:xoffset+szx].ravel())

        array_patches = np.vstack(list_patches).T
        wpatches, inputmean, whitenmatrix, dewhitenmatrix, zerophasewhitenmatrix, zerophasedewhitenmatrix = whiten_var(array_patches,num_eigs=self.num_eigs,perc_var=self.perc_var)

        self.inputmean = inputmean
        self.whitenmatrix = whitenmatrix
        self.dewhitenmatrix = dewhitenmatrix
        self.zerophasewhitenmatrix = zerophasewhitenmatrix
        self.zerophasedewhitenmatrix = zerophasedewhitenmatrix

        # crop out the central tap
        ycnt = int(np.floor(szy/2))
        xcnt = int(np.floor(szx/2))

        # select the inputmean for the central tap for each channel:
        self.convinputmean = np.reshape(inputmean,(self.nchannels,szy,szx))[:,ycnt,xcnt].reshape(1,self.nchannels,1,1)

        # let's do this one step at a time: first reshape
        zerofilters = zerophasewhitenmatrix.reshape(zerophasewhitenmatrix.shape[0],self.nchannels,szy,szx)[:,:,ycnt,xcnt]
        zerodefilters = zerophasedewhitenmatrix.reshape(zerophasedewhitenmatrix.shape[0],self.nchannels,szy,szx)[:,:,ycnt,xcnt]
        # create target array:
        self.convwhitenfilter = np.zeros(self.convwhitenfiltershp)
        self.convdewhitenfilter = np.zeros(self.convwhitenfiltershp)
        # crop out each filter one at a time, and reshape it:
        for nfilt in range(self.convwhitenfiltershp[0]):
            self.convwhitenfilter[nfilt,...] = zerofilters[:,nfilt].reshape(self.nchannels,szy,szx)
            self.convdewhitenfilter[nfilt,...] = zerodefilters[:,nfilt].reshape(self.nchannels,szy,szx)

        if self.D is None:
            self.D = patches.shape[0]

    def preprocess(self,batch):

        if self.whiten:
            batch = self._reshape_input(batch)

            batch -= self.convinputmean
            # perform convolution (Naive implementation for now):
            output = self._scipy_convolve4d(batch,self.convwhitenfilter,mode='same')
            output = self._reshape_output(output)
            return output.astype(theano.config.floatX)
        else:
            return batch.astype(theano.config.floatX)

    def display_whitening(self,save_string=None,save=True,normalize_A=True,max_factors=256,zerophasewhiten=True):
        output = {}

        nfilt, nchannels, szy, szx = self.convwhitenfiltershp

        if zerophasewhiten:
            A = self.zerophasewhitenmatrix
            if hasattr(A,'get_value'):
                A = A.get_value()
            Afilter = self.convwhitenfilter.reshape((nfilt*nchannels,szy*szx))
            Adefilter = self.convdewhitenfilter.reshape((nfilt*nchannels,szy*szx))
            A = np.vstack((Afilter,Adefilter,A))
        else:
            A = self.whitenmatrix
            if hasattr(A,'get_value'):
                A = A.get_value()

        A = A.T # make the columns the filters

        psz = int(np.ceil(np.sqrt(A.shape[0])))
        if not psz**2 == A.shape[0]:
            A = np.vstack((A,np.zeros((psz**2 - A.shape[0],A.shape[1]))))

        # plot the vectors in A
        NN = min(A.shape[1],max_factors)
        buf = 1
        sz = int(np.sqrt(NN))
        hval = np.max(np.abs(A))
        array = -np.ones(((psz+buf)*sz+buf,(psz+buf)*sz+buf))
        Aind = 0
        for r in range(sz):
            for c in range(sz):
                if normalize_A:
                    hval = np.max(np.abs(A[:,Aind]))
                Avalues = A[:,Aind].reshape(psz,psz)/hval
                array[buf+(psz+buf)*c:buf+(psz+buf)*c+psz,buf+(psz+buf)*r:buf+(psz+buf)*r+psz] = Avalues
                Aind += 1
        hval = 1.
        plt.figure(1)
        plt.clf()
        plt.imshow(array,vmin=-hval,vmax=hval,interpolation='nearest',cmap=plt.cm.gray)
        plt.colorbar()

        output['whitenmatrix'] = array
        plt.draw()

        if save:
            from hdl.config import state_dir, tstring
            savepath = os.path.join(state_dir,self.model_name + '_' + self.tstring)
            if not os.path.isdir(savepath): os.makedirs(savepath)

            if save_string is None:
                save_string = tstring()
            plt.figure(1)
            fname = os.path.join(savepath, 'whitenmatrix_' + save_string + '.png')
            plt.savefig(fname)

        return output

class ConvSparseSlowModel(ConvWhitenInputModel):

    _params = copy(ConvWhitenInputModel._params)
    _params.extend(['N','NN','A','T','K','kshp','featshp','inference_method','inference_params','lam_sparse','lam_slow','lam_l2','rec_cost','sparse_cost','slow_cost'])

    def __init__(self, **kargs):

        super(ConvSparseSlowModel, self).__init__(**kargs)

        self.N = kargs.get('N',8)
        self.NN = self.N
        self.rec_cost = kargs.get('rec_cost','convl2')
        self.sparse_cost = kargs.get('sparse_cost','l1')
        self.slow_cost = kargs.get('slow_cost',None)

        self.T = self.imshp[0]

        self.kshp = kargs.get('kshp', (self.NN,self.nchannels,8,8))
        if len(self.kshp) == 2:
            self.kshp = (self.NN, self.nchannels, self.kshp[0], self.kshp[1])
        self.featshp = (self.imshp[0],self.kshp[0],self.imshp[2] - self.kshp[2] + 1,self.imshp[3] - self.kshp[3] + 1) # num images, features, szy, szx
        self.center_basis_functions = kargs.get('center_basis_functions',True)

        self.M = int(np.prod(self.kshp[1:]))

        self.lam_sparse = theano.shared(getattr(np,theano.config.floatX)(kargs.get('lam_sparse',.1)))
        self.lam_slow = theano.shared(getattr(np,theano.config.floatX)(kargs.get('lam_slow',.1)))
        self.lam_l2 = theano.shared(getattr(np,theano.config.floatX)(kargs.get('lam_l2',.1)))

        self.K = kargs.get('K',8)

        self.inference_method = 'FISTA'
        self.inference_params = {'u_init_method': kargs.get('u_init_method','rand'),
                                 'FISTAargs': {'maxiter': kargs.get('fista_maxiter', 20),
                                               'maxline': kargs.get('fista_maxline', 10),
                                               'errthres': 1e-5,
                                               #'L':1., # default .1
                                               'verbose': False}}

        self.setup_complete = False

        if self.patch_sz:
            self.model_name = kargs.get('model_name','ConvSparseSlowModel_patchsz%03d_ksz%03d_nchannels%03d_N%03d_NN%03d_%s_%s_%s'%(self.patch_sz, self.kshp[2], self.nchannels, self.N, self.NN, self.rec_cost, self.sparse_cost, self.slow_cost))
        else:
            self.model_name = kargs.get('model_name','ConvSparseSlowModel_ksz%03d_nchannels%03d_N%03d_NN%03d_%s_%s_%s'%(self.kshp[2], self.nchannels, self.N, self.NN, self.rec_cost, self.sparse_cost, self.slow_cost))

        if self.slow_cost == 'dist' and self.T < 2:
            raise ValueError, 'self.T must be greater than or equal to 2 with slow_cost = dist'
        #self.setup()

    def _reset_on_load(self):
        self.lam_sparse = theano.shared(getattr(np,theano.config.floatX)(self.lam_sparse))
        self.lam_slow = theano.shared(getattr(np,theano.config.floatX)(self.lam_slow))
        self.lam_l2 = theano.shared(getattr(np,theano.config.floatX)(self.lam_l2))
        self.A = theano.shared(self.A.astype(theano.config.floatX))

        self.setup(init=False)

    def reset_functions(self):
        self.setup(init=False)

    def setup(self,init=True):
        """Setup model functions"""

        if init:
            A = self.normalize_A(np.random.randn(self.M,self.NN))
            self.A = theano.shared(A.astype(theano.config.floatX))

        # Inference
        if self.rec_cost == 'convl2' and self.sparse_cost == 'subspacel1' and self.slow_cost == 'dist':
            print 'Use l2subspacel1slow problem for Fista'
            #setup theano inputs to pass to setup:
            self.x = theano.shared(np.random.randn(self.M, self.T).astype(theano.config.floatX))
            self.u = np.random.randn(self.NN,self.T).astype(theano.config.floatX)

            self._fista = Fista(xinit=self.u,A=self.A,lam_sparse=self.lam_sparse,lam_slow=self.lam_slow,x=self.x,problem_type='convl2subspacel1slow')

        elif self.rec_cost == 'convl2' and self.sparse_cost == 'subspacel1':
            print 'Use l2subspacel1 problem for Fista'
            #setup theano inputs to pass to setup:
            self.x = theano.shared(np.random.randn(self.M, self.T).astype(theano.config.floatX))
            self.u = np.random.randn(self.NN,self.T).astype(theano.config.floatX)

            self._fista = Fista(xinit=self.u,A=self.A,lam_sparse=self.lam_sparse,x=self.x,problem_type='convl2subspacel1')

        elif self.rec_cost == 'convl2' and self.sparse_cost == 'Ksubspacel1':
            print 'Use l2Ksubspacel1 problem for Fista'
            #setup theano inputs to pass to setup:
            self.x = theano.shared(np.random.randn(self.M, self.T).astype(theano.config.floatX))
            self.u = np.random.randn(self.NN,self.T).astype(theano.config.floatX)

            self._fista = Fista(xinit=self.u,A=self.A,lam_sparse=self.lam_sparse,x=self.x,K=self.K,problem_type='convl2Ksubspacel1')

        elif self.rec_cost == 'convl2' and self.sparse_cost == 'elastic':
            print 'Use l2l1 problem for Fista'
            #setup theano inputs to pass to setup:
            self.x = theano.shared(np.random.randn(self.M, self.T).astype(theano.config.floatX))
            self.u = np.random.randn(self.NN,self.T).astype(theano.config.floatX)

            self._fista = Fista(xinit=self.u,A=self.A,lam_sparse=self.lam_sparse,lam_l2=self.lam_l2,x=self.x,problem_type='convl2elastic')

        elif self.rec_cost == 'convl2' and self.sparse_cost == 'l1':
            print 'Use l2l1 problem for Fista'
            #setup theano inputs to pass to setup:
            self.x = theano.shared(np.random.randn(self.M, self.T).astype(theano.config.floatX))
            self.u = np.random.randn(self.NN,self.T).astype(theano.config.floatX)

            self._fista = Fista(xinit=self.u,A=self.A,lam=self.lam_sparse,imshp=self.imshp,kshp=self.kshp,x=self.x,problem_type='convl2l1')

        else:
            raise NotImplementedError, 'Inference for rec_cost %s, sparse_cost %s, slow_cost %s'%(self.rec_cost, self.sparse_cost, self.slow_cost)

        # Update gradients:
        if self.rec_cost == 'convl2':
            from hdl.theano_methods import T_l2_cost_conv
            x = theano.tensor.matrix('x')
            u = theano.tensor.matrix('u')
            grad_A = theano.tensor.grad(T_l2_cost_conv(x,u,self.A,self.imshp,self.kshp,mask=True),self.A)
            if theano.config.device == 'cpu':
                self._df_dA = theano.function([x,u],grad_A)
            else:
                from theano.sandbox.cuda import host_from_gpu
                self._df_dA = theano.function([x,u],host_from_gpu(grad_A))
        else:
            raise NotImplementedError, 'rec_cost unknown %s'%self.rec_cost

        self.setup_complete = True

    def gradient(self,batch):

        if not self.setup_complete: raise AssertionError, 'Please call model.setup() before processing'

        t0 = time()
        batch = self.preprocess(batch)
        if verbose_timing: print 'self.preprocess time %f'%(time() - t0)

        t0 = time()
        u = self.inferlatent(batch)
        if verbose_timing: print 'self.inferlatent time %f'%(time() - t0)

        t0 = time()
        modelgradient = self.calc_modelgradient(batch,u)
        if verbose_timing: print 'self.calc_modelgradient time %f'%(time() - t0)

        return modelgradient

    def calc_modelgradient(self,batch,u):

        grad_dict = dict(dA=self._df_dA(batch,u))

        return grad_dict

    def update_model(self,update_dict,eta=None):
        """ update model parameters with update_dict['dA']
        returns the maximum update percentage max(dA/A)
        """

        A = self.A.get_value()
        param_max = np.max(np.abs(A),axis=0)

        if eta is None:
            update_max = np.max(np.abs(update_dict['dA']),axis=0)
            A -= update_dict['dA']
        else:
            update_max = eta*np.max(np.abs(update_dict['dA']),axis=0)
            A -= eta*update_dict['dA']

        update_max = np.max(update_max/param_max)

        A = self.normalize_A(A)
        self.A.set_value(A)

        return update_max

    def normalize_A(self,A):

        if self.center_basis_functions:
            A = np.reshape(A.T,self.kshp)
            Atemp = np.zeros_like(A)
            cm_y0 = self.kshp[2]/2
            cm_x0 = self.kshp[3]/2
            for n in range(A.shape[0]):
                im = np.sum(np.abs(A[n,...]),axis=0)
                cm_y, cm_x = np.round(ndimage.measurements.center_of_mass(im))
                cm_y = int(cm_y)
                cm_x = int(cm_x)
                if cm_y > cm_y0:
                    y_delta = cm_y - cm_y0
                    y0slice = slice(y_delta,self.kshp[2])
                    yslice = slice(0,self.kshp[2] - y_delta)
                else:
                    y_delta = cm_y0 - cm_y
                    y0slice = slice(0,self.kshp[2] - y_delta)
                    yslice = slice(y_delta,self.kshp[2])
                if cm_x > cm_x0:
                    x_delta = cm_x - cm_x0
                    x0slice = slice(x_delta,self.kshp[2])
                    xslice = slice(0,self.kshp[2] - x_delta)
                else:
                    x_delta = cm_x0 - cm_x
                    x0slice = slice(0,self.kshp[2] - x_delta)
                    xslice = slice(x_delta,self.kshp[2])

                Atemp[n,:,yslice,xslice] = A[n,:,y0slice,x0slice]
            A = np.reshape(Atemp.T,(np.prod(self.kshp[1:]),self.kshp[0]))

        Anorm = np.sqrt((A**2).sum(axis=0)).reshape(1,self.NN)
        return A/Anorm

    def __call__(self,batch,output_function='infer_abs',chunk_size=None):

        return self.output(self.preprocess(batch),
            output_function=output_function,chunk_size=chunk_size)

    def output(self,batch,output_function='infer_abs',chunk_size=None):

        bsz = batch.shape[1]
        if chunk_size is None:
            chunk_size = self.T

        if bsz > chunk_size:
            u_out = None
            done = False
            t0 = 0
            while not done:
                u_minibatch = self._output_minibatch(batch[:,t0:t0+chunk_size],output_function)
                if u_out is None:
                    u_out = np.zeros((u_minibatch.shape[0],bsz))
                u_out[:,t0:t0+chunk_size] = u_minibatch
                t0 += chunk_size
                if t0 >= bsz: done = True
            return u_out
        else:
            return self._output_minibatch(batch,output_function)

    def _output_minibatch(self,batch,output_function):
        if output_function == 'infer':
            return self.inferlatent(batch)
        elif output_function == 'proj':
            raise NotImplementedError
            #return np.dot(self.A.get_value().T,batch)
        elif output_function == 'infer_abs':
            u = self.inferlatent(batch)
            return np.sqrt(u[::2,:]**2 + u[1::2,:]**2)
        elif output_function == 'proj_abs':
            raise NotImplementedError
            #u = np.dot(self.A.get_value().T,batch)
            #return np.sqrt(u[::2,:]**2 + u[1::2,:]**2)
        elif output_function == 'infer_loga':
            u = self.inferlatent(batch)
            amp = np.sqrt(u[::2,:]**2 + u[1::2,:]**2)
            return np.log(amp + .01)
        elif output_function == 'proj_loga':
            raise NotImplementedError
            #u = np.dot(self.A.get_value().T,batch)
            #amp = np.sqrt(u[::2,:]**2 + u[1::2,:]**2)
            #return np.log(amp + .01)
        else:
            assert NotImplemented, 'Unknown output_function %s'%output_function

    def __initu(self,batch):

        if self.inference_params['u_init_method'] == 'proj':
            raise NotImplemented
            #return np.dot(self.A.get_value().T,batch)
        elif self.inference_params['u_init_method'] == 'rand':
            return .001*np.random.randn(np.prod(self.featshp[1:]), batch.shape[1]).astype(theano.config.floatX)
        else:
            raise NotImplementedError

    def inferlatent(self,batch):
        """

        """
        if not self.inference_method == 'FISTA':
            raise NotImplementedError

        # set shared variables
        t0 = time()
        u0 = self.__initu(batch)
        if verbose_timing: print 'self.__initu time %f'%(time() - t0)

        # run inference
        uhat, fista_history = self._fista(u0, batch, **self.inference_params['FISTAargs'])

        # return
        return uhat

    def display(self,save_string=None,save=True,normalize_A=True,max_factors=256,zerophasewhiten=True):
        output = {}
        if hasattr(self.A,'get_value'):
            A = self.A.get_value()
        else:
            A = self.A

        if not zerophasewhiten:
            Areshaped = np.reshape(A.T,self.kshp)
            Aoutput = self._scipy_convolve4d(Areshaped,self.convdewhitenfilter,mode='same')
            A = np.reshape(Aoutput,(self.kshp[0],np.prod(self.kshp[1:]))).T

        psz = int(np.ceil(np.sqrt(A.shape[0])))
        if not psz**2 == A.shape[0]:
            A = np.vstack((A,np.zeros((psz**2 - A.shape[0],A.shape[1]))))

        # plot the vectors in A
        NN = min(self.NN,max_factors)
        buf = 1
        sz = int(np.sqrt(NN))
        hval = np.max(np.abs(A))
        array = -np.ones(((psz+buf)*sz+buf,(psz+buf)*sz+buf))
        Aind = 0
        for r in range(sz):
            for c in range(sz):
                if normalize_A:
                    hval = np.max(np.abs(A[:,Aind]))
                Avalues = A[:,Aind].reshape(psz,psz)/hval
                array[buf+(psz+buf)*c:buf+(psz+buf)*c+psz,buf+(psz+buf)*r:buf+(psz+buf)*r+psz] = Avalues
                Aind += 1
        hval = 1.
        plt.figure(1)
        plt.clf()
        plt.imshow(array,vmin=-hval,vmax=hval,interpolation='nearest',cmap=plt.cm.gray)
        plt.colorbar()

        output['A'] = array

        plt.draw()

        if save:
            from hdl.config import state_dir, tstring
            savepath = os.path.join(state_dir,self.model_name + '_' + self.tstring)
            if not os.path.isdir(savepath): os.makedirs(savepath)

            if save_string is None:
                save_string = tstring()
            plt.figure(1)
            fname = os.path.join(savepath, 'A_' + save_string + '.png')
            plt.savefig(fname)

        return output

def test_imageshape():

    from hdl.learners import SGD

    whitenpatches = 100

    model = ConvSparseSlowModel(imshp=(whitenpatches,1,64,64),convwhitenfiltershp=(7,7),N=16,kshp=(5,5),perc_var=100.)

    l = SGD(model=model,datasource='vid075-chunks',display_every=100,save_every=10000,batchsize=model.imshp[0])

    print 'whitenpatches', whitenpatches
    print 'model.imshp', model.imshp
    print 'model.convwhitenfiltershp', model.convwhitenfiltershp

    databatch = l.get_databatch(whitenpatches)

    from matplotlib import pyplot as plt

    images = np.transpose(databatch).reshape(l.model.imshp)
    plt.figure(1)
    for ind in range(100):
        plt.subplot(10,10,ind)
        im = np.squeeze(images[ind,0,:,:])
        plt.imshow(im,interpolation='nearest',cmap=plt.cm.gray)
        plt.axis('off')
        plt.draw()
    plt.show()

def test_convsparsenet(lam_sparse=.1,N=16,perc_var=100.):

    from hdl.models import SparseSlowModel
    from hdl.learners import SGD

    whitenpatches = 1000

    #model = ConvWhitenInputModel(imshp=(10,1,32,32),convwhitenfiltershp=(7,7),perc_var=100.)
    model = ConvSparseSlowModel(imshp=(10,1,28,28),convwhitenfiltershp=(7,7),N=N,kshp=(7,7),perc_var=perc_var,lam_sparse=lam_sparse)

    l = SGD(model=model,datasource='vid075-chunks',display_every=1000,save_every=10000,batchsize=model.imshp[0])

    print 'whitenpatches', whitenpatches
    print 'model.imshp', model.imshp
    print 'model.convwhitenfiltershp', model.convwhitenfiltershp

    databatch = l.get_databatch(whitenpatches)
    l.model.learn_whitening(databatch)

    l.model.setup()

    l.learn(iterations=20000)
    l.change_target(.5)
    l.learn(iterations=5000)
    l.change_target(.5)
    l.learn(iterations=5000)

    #l.learn(iterations=160000)
    #l.change_target(.5)
    #l.learn(iterations=20000)
    #l.change_target(.5)
    #l.learn(iterations=20000)

    from hdl.display import display_final
    display_final(l.model)

def sweep_lambda():

    #lam_sparses = [0.01, 0.02, 0.04, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.8, 1.0, 2.0, 4.0, 10.0]
    #lam_sparses = [1.0, 2.0, 4.0, 8.0, 10.0]
    lam_sparses = [1.0]
    #Ns = [32, 48, 64, 80, 96, 112, 128]
    Ns = [16]
    #perc_vars = [80., 95., 99., 99.5, 99.9]
    perc_vars = [100.]
    for lam_sparse in lam_sparses:
        for N in Ns:
            for perc_var in perc_vars:
                test_convsparsenet(lam_sparse=lam_sparse,N=N,perc_var=perc_var)

if __name__ == '__main__':

    #test_corr()
    #test_derivative()
    #test_imageshape()
    #test_convsparsenet()
    sweep_lambda()
