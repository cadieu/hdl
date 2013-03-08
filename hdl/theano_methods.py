from theano import tensor as T
from theano.tensor.nnet import conv2d

def T_l2_cost_norm(x,a,A):
    Alength = T.sqrt((A**2).sum(axis=0))
    #Alengthreshape = Alength.dimshuffle(('x',0))
    #Alengthreshape = T.addbroadcast(Alength,0)
    Anorm = A/Alength
    xhat = T.dot(Anorm,a)
    error = xhat - x
    return .5*T.sum(error**2)

def T_l2_cost(x,a,A):
    xhat = T.dot(A,a)
    error = xhat - x
    return .5*T.sum(error**2)

def T_gl2_cost(x,a,A):
    _l2_cost = T_l2_cost(x,a,A)
    return T.grad(_l2_cost,a)

def T_l1_cost(a,lam):
    return lam*T.sum(T.abs_(a))

def T_gl1_cost(a,lam):
    _l1_cost = T_l1_cost(a,lam)
    return T.grad(_l1_cost,a)

def T_a_shrinkage(a,L,lam):
    a_shrinkage = T.abs_(a) - lam/L
    #a_sign = T.switch(T.gt(a,0.),1.,0.) + T.switch(T.lt(a,0.),-1.,0.)
    return T.switch(T.gt(a_shrinkage,0.),a_shrinkage*T.sgn(a),0.)

def T_subspacel1_cost(a,lam_sparse,small_value=.001):
    amp = T.sqrt(a[::2,:]**2 + a[1::2,:]**2 + small_value)
    # subspace l1 cost
    return lam_sparse*T.sum(amp)

def T_gsubspacel1_cost(a,lam_sparse,small_value=.001):
    _subspacel1_cost = T_subspacel1_cost(a,lam_sparse,small_value)
    return T.grad(_subspacel1_cost,a)

def T_subspacel1_shrinkage(a,L,lam_sparse,small_value=.001):
    amp = T.sqrt(a[::2,:]**2 + a[1::2,:]**2 + small_value)

    # subspace l1 shrinkage
    amp_shrinkage = 1. - (lam_sparse/L)/amp
    amp_value = T.switch(T.gt(amp_shrinkage,0.),amp_shrinkage,0.)
    subspacel1_prox = T.zeros_like(a)
    subspacel1_prox = T.set_subtensor(subspacel1_prox[ ::2,:],amp_value*a[ ::2,:])
    subspacel1_prox = T.set_subtensor(subspacel1_prox[1::2,:],amp_value*a[1::2,:])
    return subspacel1_prox

def T_Ksubspacel1_cost(a,lam_sparse,K=8,small_value=.001):
    a_list = [a[k::K,:]**2 for k in range(K)]
    amp = T.sqrt(T.add(*a_list)+small_value)
    # subspace l1 cost
    return lam_sparse*T.sum(amp)

def T_gKsubspacel1_cost(a,lam_sparse,K=8,small_value=.001):
    _subspacel1_cost = T_Ksubspacel1_cost(a,lam_sparse,K,small_value)
    return T.grad(_subspacel1_cost,a)

def T_Ksubspacel1_shrinkage(a,L,lam_sparse,K=8,small_value=.001):
    a_list = [a[k::K,:]**2 for k in range(K)]
    amp = T.sqrt(T.add(*a_list)+small_value)

    # subspace l1 shrinkage
    amp_shrinkage = 1. - (lam_sparse/L)/amp
    amp_value = T.switch(T.gt(amp_shrinkage,0.),amp_shrinkage,0.)
    subspacel1_prox = T.zeros_like(a)
    for k in range(K):
        subspacel1_prox = T.set_subtensor(subspacel1_prox[k::K,:],amp_value*a[k::K,:])
    return subspacel1_prox

def T_subspacel1_slow_cost(a,lam_sparse,lam_slow,small_value=.001):
    amp = T.sqrt(a[::2,:]**2 + a[1::2,:]**2 + small_value)
    damp = amp[:,1:] - amp[:,:-1]
    # slow cost
    _slow_cost = (.5*lam_slow)*T.sum(damp**2)
    # subspace l1 cost
    _subspacel1_cost = lam_sparse*T.sum(amp)

    return _slow_cost + _subspacel1_cost

def T_gsubspacel1_slow_cost(a,lam_sparse,lam_slow,small_value=.001):
    _subspacel1_slow_cost = T_subspacel1_slow_cost(a,lam_sparse,lam_slow,small_value)
    return T.grad(_subspacel1_slow_cost,a)

def T_subspacel1_slow_shrinkage(a,L,lam_sparse,lam_slow,small_value=.001):
    amp = T.sqrt(a[::2,:]**2 + a[1::2,:]**2 + small_value)
    #damp = amp[:,1:] - amp[:,:-1]

    # compose slow shrinkage with subspace l1 shrinkage

    # slow shrinkage
    div = T.zeros_like(amp)
    d1 = amp[:,1:] - amp[:,:-1]
    d2 = d1[:,1:] - d1[:,:-1]
    div = T.set_subtensor(div[:,1:-1],-d2)
    div = T.set_subtensor(div[:,0], -d1[:,0])
    div = T.set_subtensor(div[:,-1], d1[:,-1])
    slow_amp_shrinkage = 1 - (lam_slow/L)*(div/amp)
    slow_amp_value = T.switch(T.gt(slow_amp_shrinkage,0),slow_amp_shrinkage,0)
    slow_shrinkage_prox_a = slow_amp_value*a[::2,:]
    slow_shrinkage_prox_b = slow_amp_value*a[1::2,:]

    # subspace l1 shrinkage
    amp_slow_shrinkage_prox = T.sqrt(slow_shrinkage_prox_a**2 + slow_shrinkage_prox_b**2)
    #amp_shrinkage = 1. - (lam_slow*lam_sparse/L)*amp_slow_shrinkage_prox
    amp_shrinkage = 1. - (lam_sparse/L)/amp_slow_shrinkage_prox
    amp_value = T.switch(T.gt(amp_shrinkage,0.),amp_shrinkage,0.)
    subspacel1_prox = T.zeros_like(a)
    subspacel1_prox = T.set_subtensor(subspacel1_prox[ ::2,:],amp_value*slow_shrinkage_prox_a)
    subspacel1_prox = T.set_subtensor(subspacel1_prox[1::2,:],amp_value*slow_shrinkage_prox_b)
    return subspacel1_prox

def T_l2_vector_cost(a,lam):
    return .5*lam*T.sum(a**2)

def T_elastic_cost(a,lam_sparse,lam_l2):
    _l1_cost = T_l1_cost(a,lam_sparse)
    _l2_cost = T_l2_vector_cost(a,lam_l2)
    return _l1_cost + _l2_cost

def T_gelastic_cost(a,lam_sparse,lam_l2):
    _elastic_cost = T_elastic_cost(a,lam_sparse,lam_l2)
    return T.grad(_elastic_cost,a)

def T_elastic_shrinkage(a,L,lam_sparse,lam_l2):

    # compose l2 shrinkage with l1 shrinkage
    prox_l1 = T_a_shrinkage(a,L,lam_sparse)
    prox_elastic = prox_l1/(1 + lam_l2*lam_sparse/L)
    return prox_elastic

def T_l2_cost_conv(x,a,A,imshp,kshp,featshp,stride=(1,1),mask=True):
    """
    xsz*ysz*nchannels, nimages = x.shape
    xsz*ysz*nfeat, nimages = a.shape
    xsz*ysz*nchannels, nfeat = A.shape
    """

    #imshp = num images, channels, szy, szx
    #kshp = features, channels, szy, szx
    #featshp = num images, features, szy, szx

    image_error, kernel, features = helper_T_l2_cost_conv(x=x,a=a,A=A,imshp=imshp,kshp=kshp,featshp=featshp,stride=stride,mask=mask)

    return .5*T.sum(image_error **2)

def T_gl2_cost_conv(x,a,A,imshp,kshp,featshp,stride=(1,1),mask=True):
    image_error, kernel, features = helper_T_l2_cost_conv(x=x,a=a,A=A,imshp=imshp,kshp=kshp,featshp=featshp,stride=stride,mask=mask)
    feature_grad = -conv2d(image_error,kernel,image_shape=imshp,filter_shape=kshp,subsample=stride)
    feature_grad = feature_grad[:,:,:featshp[2],:featshp[3]]
    reshaped_feature_grad = T.transpose(T.reshape(feature_grad,(featshp[0],featshp[1]*featshp[2]*featshp[3]),ndim=2))
    return reshaped_feature_grad

def helper_T_l2_cost_conv(x,a,A,imshp,kshp,featshp,stride=(1,1),mask=True):
    """
    xsz*ysz*nchannels, nimages = x.shape
    xsz*ysz*nfeat, nimages = a.shape
    xsz*ysz*nchannels, nfeat = A.shape
    """

    #imshp = num images, channels, szy, szx
    #kshp = features, channels, szy, szx
    #featshp = num images, features, szy, szx

    image = T.reshape(T.transpose(x),imshp,ndim=4)
    kernel = T.reshape(T.transpose(A),kshp,ndim=4)
    features = T.reshape(T.transpose(a),featshp,ndim=4)

    if stride == (1,1):
        #Need to transpose first two dimensions of kernel, and reverse index kernel image dims (for correlation)
        kernel_rotated = T.transpose(kernel[:,:,::-1,::-1],axes=[1,0,2,3])
        featshp_logical = (featshp[0],featshp[1],featshp[2]*stride[0],featshp[3]*stride[1])
        kshp_rotated = (kshp[1], kshp[0], kshp[2], kshp[3])
        image_estimate = conv2d(features,kernel_rotated,border_mode='full',
                                image_shape=featshp,filter_shape=kshp_rotated,
                                imshp_logical=featshp_logical[1:],kshp_logical=kshp[2:])
    else:
        my_corr2d = MyCorr(strides=stride,imshp=imshp)
        image_estimate = my_corr2d(features,kernel)

    if mask:
        image_error_temp = image - image_estimate
        image_error = T.zeros_like(image_error_temp)
        #image_error = T.set_subtensor(image_error[:,:,(kshp[2]-1):(imshp[2]-kshp[2]+1),(kshp[3]-1):(imshp[3]-kshp[3]+1)],
        #                         image_error_temp[:,:,(kshp[2]-1):(imshp[2]-kshp[2]+1),(kshp[3]-1):(imshp[3]-kshp[3]+1)])
        image_error = T.set_subtensor(image_error[:,:,(kshp[2]-stride[0]):(imshp[2]-kshp[2]+stride[0]),(kshp[3]-stride[1]):(imshp[3]-kshp[3]+stride[1])],
                                 image_error_temp[:,:,(kshp[2]-stride[0]):(imshp[2]-kshp[2]+stride[0]),(kshp[3]-stride[1]):(imshp[3]-kshp[3]+stride[1])])
    else:
        image_error = image - image_estimate

    return image_error, kernel, features

def T_l2_cost_conv_dA(x,a,A,imshp,kshp,featshp,stride=(1,1),mask=True):
    image_error, kernel, features = helper_T_l2_cost_conv(x=x,a=a,A=A,imshp=imshp,kshp=kshp,featshp=featshp,stride=stride,mask=mask)

    if stride == (1,1):

        image_error_rot = T.transpose(image_error,[1,0,2,3])[:,:,::-1,::-1]
        imshp_rot = (imshp[1],imshp[0],imshp[2],imshp[3])
        featshp_rot = (featshp[1],featshp[0],featshp[2],featshp[3])
        features_rot = T.transpose(features,[1,0,2,3])

        featshp_rot_logical = (featshp_rot[0],
                               featshp_rot[1],
                               imshp[2] - kshp[2] + 1,
                               imshp[3] - kshp[3] + 1)
        kernel_grad_rot = -1.*conv2d(image_error_rot,features_rot,
                                  image_shape=imshp_rot,filter_shape=featshp_rot,
                                  imshp_logical=imshp_rot[1:],kshp_logical=featshp_rot_logical[2:])
        kernel_grad = T.transpose(kernel_grad_rot,[1,0,2,3])

        reshape_kernel_grad = T.transpose(T.reshape(kernel_grad,(kshp[0],kshp[1]*kshp[2]*kshp[3]),ndim=2))

        return reshape_kernel_grad

    else:
        my_conv = MyConv_view(strides=stride,kshp=kshp)
        kernel_grad = my_conv(image_error,features)

        reshape_kernel_grad = T.transpose(T.reshape(kernel_grad, (kshp[0], kshp[1] * kshp[2] * kshp[3]), ndim=2))

        return reshape_kernel_grad

def T_l2_cost_conv_dA_old(x, a, A, imshp, kshp, featshp, stride=(1, 1), mask=True):
    image_error, kernel, features = helper_T_l2_cost_conv(x=x, a=a, A=A, imshp=imshp, kshp=kshp, featshp=featshp,
        stride=stride, mask=mask)

    image_error_rot = T.transpose(image_error, [1, 0, 2, 3])[:, :, ::-1, ::-1]
    imshp_rot = (imshp[1], imshp[0], imshp[2], imshp[3])
    featshp_rot = (featshp[1], featshp[0], featshp[2], featshp[3])
    features_rot = T.transpose(features, [1, 0, 2, 3])

    featshp_rot_logical = (featshp_rot[0],
                           featshp_rot[1],
                           imshp[2] - kshp[2] + 1,
                           imshp[3] - kshp[3] + 1)
    kernel_grad_rot = -1. * conv2d(image_error_rot, features_rot,
        image_shape=imshp_rot, filter_shape=featshp_rot,
        imshp_logical=imshp_rot[1:], kshp_logical=featshp_rot_logical[2:])
    kernel_grad = T.transpose(kernel_grad_rot, [1, 0, 2, 3])

    reshape_kernel_grad = T.transpose(T.reshape(kernel_grad, (kshp[0], kshp[1] * kshp[2] * kshp[3]), ndim=2))

    return reshape_kernel_grad

def T_subspacel1_cost_conv(a,lam_sparse,imshp,kshp,featshp,stride=(1,1),small_value=.001):
    featshp = (imshp[0],kshp[0],featshp[2],featshp[3]) # num images, features, szy, szx
    features = T.reshape(T.transpose(a),featshp,ndim=4)

    amp = T.sqrt(features[:,::2,:,:]**2 + features[:,1::2,:,:]**2 + small_value)
    # subspace l1 cost
    return lam_sparse*T.sum(amp)

def T_gsubspacel1_cost_conv(a,lam_sparse,imshp,kshp,featshp,stride=(1,1),small_value=.001):

    _subspacel1_cost = T_subspacel1_cost_conv(a,lam_sparse,imshp,kshp,featshp,stride=stride,small_value=small_value)
    return T.grad(_subspacel1_cost,a)

def T_subspacel1_shrinkage_conv(a,L,lam_sparse,imshp,kshp,featshp,stride=(1,1),small_value=.001):
    featshp = (imshp[0],kshp[0],featshp[2],featshp[3]) # num images, features, szy, szx
    features = T.reshape(T.transpose(a),featshp,ndim=4)

    amp = T.sqrt(features[:,::2,:,:]**2 + features[:,1::2,:,:]**2 + small_value)

    # subspace l1 shrinkage
    amp_shrinkage = 1. - (lam_sparse/L)/amp
    amp_value = T.switch(T.gt(amp_shrinkage,0.),amp_shrinkage,0.)
    subspacel1_prox = T.zeros_like(features)
    subspacel1_prox = T.set_subtensor(subspacel1_prox[:, ::2,:,:],amp_value*features[:, ::2,:,:])
    subspacel1_prox = T.set_subtensor(subspacel1_prox[:,1::2,:,:],amp_value*features[:,1::2,:,:])

    reshape_subspacel1_prox = T.transpose(T.reshape(subspacel1_prox,(featshp[0],featshp[1]*featshp[2]*featshp[3]),ndim=2))

    return reshape_subspacel1_prox


def T_subspacel1_slow_cost_conv(a, lam_sparse, lam_slow, imshp,kshp,featshp,stride=(1,1), small_value=.001):
    featshp = (imshp[0], kshp[0], featshp[2], featshp[3]) # num images, features, szy, szx
    features = T.reshape(T.transpose(a), featshp, ndim=4)

    amp = T.sqrt(features[:,::2,:,:]**2 + features[:,1::2,:,:]**2 + small_value)
    damp = amp[1:,:,:,:] - amp[:-1,:,:,:]
    # slow cost
    _slow_cost = (.5 * lam_slow) * T.sum(damp ** 2)
    # subspace l1 cost
    _subspacel1_cost = lam_sparse * T.sum(amp)

    return _slow_cost + _subspacel1_cost

def T_gsubspacel1_slow_cost_conv(a, lam_sparse, lam_slow, imshp,kshp,featshp,stride=(1,1),small_value=.001):

    _subspacel1_slow_cost = T_subspacel1_slow_cost_conv(a, lam_sparse, lam_slow, imshp, kshp, featshp,stride=stride,small_value=small_value)
    return T.grad(_subspacel1_slow_cost, a)

def T_subspacel1_slow_shrinkage_conv(a, L, lam_sparse, lam_slow, imshp,kshp,featshp,stride=(1,1),small_value=.001):
    featshp = (imshp[0],kshp[0],featshp[2],featshp[3]) # num images, features, szy, szx
    features = T.reshape(T.transpose(a),featshp,ndim=4)

    amp = T.sqrt(features[:,::2,:,:]**2 + features[:,1::2,:,:]**2 + small_value)
    #damp = amp[:,1:] - amp[:,:-1]

    # compose slow shrinkage with subspace l1 shrinkage

    # slow shrinkage
    div = T.zeros_like(amp)
    d1 = amp[1:,:,:,:] - amp[:-1,:,:,:]
    d2 = d1[1:,:,:,:] - d1[:-1,:,:,:]
    div = T.set_subtensor(div[1:-1,:,:,:], -d2)
    div = T.set_subtensor(div[0,:,:,:], -d1[0,:,:,:])
    div = T.set_subtensor(div[-1,:,:,:], d1[-1,:,:,:])
    slow_amp_shrinkage = 1 - (lam_slow / L) * (div / amp)
    slow_amp_value = T.switch(T.gt(slow_amp_shrinkage, 0), slow_amp_shrinkage, 0)
    slow_shrinkage_prox_a = slow_amp_value * features[:, ::2, :,:]
    slow_shrinkage_prox_b = slow_amp_value * features[:,1::2, :,:]

    # subspace l1 shrinkage
    amp_slow_shrinkage_prox = T.sqrt(slow_shrinkage_prox_a ** 2 + slow_shrinkage_prox_b ** 2)
    #amp_shrinkage = 1. - (lam_slow*lam_sparse/L)*amp_slow_shrinkage_prox
    amp_shrinkage = 1. - (lam_sparse / L) / amp_slow_shrinkage_prox
    amp_value = T.switch(T.gt(amp_shrinkage, 0.), amp_shrinkage, 0.)
    subspacel1_prox = T.zeros_like(features)
    subspacel1_prox = T.set_subtensor(subspacel1_prox[:, ::2, :,:], amp_value * slow_shrinkage_prox_a)
    subspacel1_prox = T.set_subtensor(subspacel1_prox[:,1::2, :,:], amp_value * slow_shrinkage_prox_b)

    reshape_subspacel1_prox = T.transpose(T.reshape(subspacel1_prox,(featshp[0],featshp[1]*featshp[2]*featshp[3]),ndim=2))
    return reshape_subspacel1_prox


def T_subspacel1mean_cost_conv(a,lam_sparse,imshp,kshp,featshp,stride=(1,1),small_value=.001):
    featshp = (imshp[0],kshp[0],featshp[2],featshp[3]) # num images, features, szy, szx
    features = T.reshape(T.transpose(a),featshp,ndim=4)

    amp = T.sqrt(features[:,:-1:2,:,:]**2 + features[:,1:-1:2,:,:]**2 + small_value)
    # subspace l1 cost
    return lam_sparse*T.sum(amp)

def T_gsubspacel1mean_cost_conv(a,lam_sparse,imshp,kshp,featshp,stride=(1,1),small_value=.001):

    _subspacel1_cost = T_subspacel1_cost_conv(a,lam_sparse,imshp,kshp,featshp,stride=stride,small_value=small_value)
    return T.grad(_subspacel1_cost,a)

def T_subspacel1mean_shrinkage_conv(a,L,lam_sparse,imshp,kshp,featshp,stride=(1,1),small_value=.001):
    featshp = (imshp[0],kshp[0],featshp[2],featshp[3]) # num images, features, szy, szx
    features = T.reshape(T.transpose(a),featshp,ndim=4)

    amp = T.sqrt(features[:,:-1:2,:,:]**2 + features[:,1:-1:2,:,:]**2 + small_value)

    # subspace l1 shrinkage
    amp_shrinkage = 1. - (lam_sparse/L)/amp
    amp_value = T.switch(T.gt(amp_shrinkage,0.),amp_shrinkage,0.)
    subspacel1_prox = T.zeros_like(features)
    subspacel1_prox = T.set_subtensor(subspacel1_prox[:, :-1:2,:,:],amp_value*features[:, :-1:2,:,:])
    subspacel1_prox = T.set_subtensor(subspacel1_prox[:,1:-1:2,:,:],amp_value*features[:,1:-1:2,:,:])
    subspacel1_prox = T.set_subtensor(subspacel1_prox[:,-1,:,:],              features[:,-1,:,:])

    reshape_subspacel1_prox = T.transpose(T.reshape(subspacel1_prox,(featshp[0],featshp[1]*featshp[2]*featshp[3]),ndim=2))

    return reshape_subspacel1_prox

# amp phase generative model?
#def T_l2_amp_phase_cost(x,a,A):
#    N = a.shape[0]/2
#    loga = T.dot(A[:,:N],a[:N,:])
#    archat = T.dot(A[:,N:],a[N:,:])
#    xhat = T.dot(A,a)
#    error = xhat - x
#    return .5*T.sum(error**2)
#
#def T_gl2_amp_phase_cost(x,a,A):
#    _l2_cost = T_l2_cost(x,a,A)
#    return T.grad(_l2_cost,a)

from theano.gof import Op, Apply
from skimage.util.shape import view_as_windows

import numpy as np

class MyCorr(Op):

    def __init__(self, strides, imshp):
        super(MyCorr, self).__init__()
        self.strides = strides
        self.imshp = imshp

    def __eq__(self, other):
        return type(self) == type(other)
    def __hash__(self):
        return hash(type(self))
    def make_node(self, features, kernel):
        return Apply(self, [features, kernel],[features.type()])
    def perform(self, node, inputs, output_storage):
        features, kernel = inputs
        strides = self.strides

        featshp = features.shape
        kshp = kernel.shape

        produced_output_sz = (featshp[0], kshp[1], kshp[2] + strides[0] * (featshp[2] - 1), kshp[3] + strides[1] * (featshp[3] - 1))
        returned_output_sz = (featshp[0], kshp[1], self.imshp[2], self.imshp[3])

        k_rot = kernel[:,:,::-1,::-1]

        scipy_output = np.zeros(returned_output_sz,dtype=node.out.dtype)
        for im_i in range(featshp[0]):

            im_out = np.zeros(produced_output_sz[1:],dtype=node.out.dtype)
            im_outr = view_as_windows(im_out,(kshp[1],kshp[2],kshp[3]))[0,::strides[0],::strides[1],...]

            im_hatr = np.tensordot(features[im_i,...],k_rot,axes=((0,),(0,)))

            for a in range(im_hatr.shape[0]):
                for b in range(im_hatr.shape[1]):
                    im_outr[a,b,...] += im_hatr[a,b,...]

            if produced_output_sz[2] <= returned_output_sz[2]:
                scipy_output[im_i,:,:im_out.shape[1],:im_out.shape[2]] = im_out
            else:
                scipy_output[im_i,:,:,:] = im_out[:,:returned_output_sz[2],:returned_output_sz[3]]

        #print 'MyCorr, output.shape:', scipy_output.shape, 'strides', strides, 'featshp', featshp, 'kshp', kshp, 'imshp', self.imshp,\
        #'produced_output_sz', produced_output_sz, 'returned_output_sz', returned_output_sz, \
        #'im_outr.shape', im_outr.shape, 'im_hatr.shape', im_hatr.shape

        output_storage[0][0] = scipy_output


def scipy_convolve4d(image, kernel, mode='valid', stride=(1, 1)):
    from scipy.signal import convolve2d

    imshp = image.shape
    kshp = kernel.shape
    if mode == 'valid':
        featshp = (imshp[0], kshp[0], imshp[2] - kshp[2] + 1, imshp[3] - kshp[3] + 1) # num images, features, szy, szx
    elif mode == 'same':
        featshp = (imshp[0], kshp[0], imshp[2], imshp[3]) # num images, features, szy, szx
    elif mode == 'full':
        featshp = (imshp[0], kshp[0], imshp[2] + kshp[2] - 1, imshp[3] + kshp[3] - 1) # num images, features, szy, szx
    else:
        raise NotImplemented, 'Unkonwn mode %s' % mode

    scipy_output = np.zeros(featshp)
    for im_i in range(imshp[0]):
        for k_i in range(kshp[0]):
            for im_j in range(imshp[1]):
                scipy_output[im_i, k_i, :, :] += convolve2d(np.squeeze(image[im_i, im_j, :, :]),
                    np.squeeze(kernel[k_i, im_j, :, :]), mode=mode)

    if not stride == (1, 1):
        scipy_output = scipy_output[:, :, ::stride[0], ::stride[1]]

    return scipy_output

from hdl.operations import convolve4d_view

class MyConv(Op):
    def __init__(self, strides, kshp):
        super(MyConv, self).__init__()
        self.strides = strides
        self.kshp = kshp

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def make_node(self, features, kernel):
        return Apply(self, [features, kernel], [features.type()])

    def perform(self, node, inputs, output_storage):
        image_error, features = inputs
        strides = self.strides

        scipy_error_rot = np.transpose(image_error, [1, 0, 2, 3])
        features_rot = np.transpose(features, [1, 0, 2, 3])

        feat_expand = np.zeros((features_rot.shape[0],
                                features_rot.shape[1],
                                image_error.shape[2] - self.kshp[2] + 1,
                                image_error.shape[3] - self.kshp[3] + 1), dtype=features.dtype)
        feat_expand[:, :, ::strides[0], ::strides[1]] = features_rot

        #scipy_derivative_rot = -scipy_convolve4d(scipy_error_rot[:, :, ::-1, ::-1], feat_expand)
        scipy_derivative_rot = -convolve4d_view(scipy_error_rot[:, :, ::-1, ::-1], feat_expand)
        scipy_derivative = np.transpose(scipy_derivative_rot, [1, 0, 2, 3])

        output_storage[0][0] = scipy_derivative

class MyConv_view(Op):
    def __init__(self, strides, kshp):
        super(MyConv_view, self).__init__()
        self.strides = strides
        self.kshp = kshp

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def make_node(self, features, kernel):
        return Apply(self, [features, kernel], [features.type()])

    def perform(self, node, inputs, output_storage):
        image_error, features = inputs
        kshp = self.kshp
        imshp = image_error.shape
        strides = self.strides

#        scipy_error_rot = np.transpose(image_error, [1, 0, 2, 3])
#        features_rot = np.transpose(features, [1, 0, 2, 3])
#
#        feat_expand = np.zeros((features_rot.shape[0],
#                                features_rot.shape[1],
#                                image_error.shape[2] - self.kshp[2] + 1,
#                                image_error.shape[3] - self.kshp[3] + 1), dtype=features.dtype)
#        feat_expand[:, :, ::strides[0], ::strides[1]] = features_rot
#
#        #scipy_derivative_rot = -scipy_convolve4d(scipy_error_rot[:, :, ::-1, ::-1], feat_expand)
#
#        feat_flipped = features_rot[:, :, ::-1, ::-1]

        from skimage.util.shape import view_as_windows

        image_error_view = view_as_windows(image_error, (imshp[0], kshp[1], kshp[2], kshp[3]))[0,0,::strides[0],::strides[1],...]
        # image_error_view.shape = (featszr, featszc, num_im, channels, ksz[2], ksz[3])
        # features.shape = (num_im, num_filters, featszr, featszc)
        kernel_derivative = - np.tensordot(features,image_error_view, axes=((0, 2, 3), (2, 0, 1)))
        # kernel_derivative_temp.shape = (num_filters, channels, ksz[2], ksz[3])

        output_storage[0][0] = kernel_derivative[:,:,::-1,::-1]

#        output = np.zeros(kshp,dtype=image_error.dtype)
#        this_image = None
#        for im_i in range(imshp[0]):
#            this_image = image_error[im_i, :, ::-1, ::-1]
#            imager = view_as_windows(this_image, (kshp[1], kshp[2], kshp[3]))[0, ::strides[0], ::strides[1], ...]
#            # imager.shape = (featszr, featszc, channels, ksz[2], ksz[3])
#            feat = np.tensordot(feat_flipped, imager, axes=((1, 2, 3), (2, 3, 4)))
#
#            output[im_i, ...] = feat
#
#        scipy_derivative_rot = -convolve4d_view(scipy_error_rot[:, :, ::-1, ::-1], feat_expand)
#        scipy_derivative = np.transpose(scipy_derivative_rot, [1, 0, 2, 3])
#
#        output_storage[0][0] = scipy_derivative
