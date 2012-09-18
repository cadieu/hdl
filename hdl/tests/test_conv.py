import numpy as np

import theano
from theano import tensor as T
from theano import function, Param, shared, In, Out, sandbox
from theano.tensor.nnet import conv2d

from scipy.signal import correlate2d, convolve2d

from hdl.models import ConvWhitenInputModel, ConvSparseSlowModel

DEBUG = True

def check_grad(func, grad, x0, args, dtype=float, epsilon=None, **kargs):
    """from scipy.optimize
    """

    _dtype = dtype
    if epsilon is None:
        _epsilon = np.sqrt(np.finfo(_dtype).eps)
    else:
        _epsilon = epsilon

    def approx_fprime(xk,f,epsilon,*args):
        f0 = f(*((xk,)+args))
        grad = np.zeros_like(xk)
        ei = np.zeros_like(xk)
        it = np.nditer(xk,flags=['multi_index'])
        while not it.finished:
            ei[it.multi_index] = epsilon
            grad[it.multi_index] = (f(*((xk+ei,)+args)) - f0)/epsilon
            ei[it.multi_index] = 0.0
            it.iternext()
        return grad

    function_grad = grad(x0,*args)
    numeric_grad = approx_fprime(x0,func,_epsilon,*args)
    difference = np.mean((function_grad-numeric_grad)**2)

    return difference, function_grad, numeric_grad

def _test_feature_derivative_imshp_kshp(imshp=(2,1,16,16),kshp=(1,2,5,5),stride=(1,1),mask=False):

    assert imshp[1] == kshp[1]

    featshp = (imshp[0],kshp[0],(imshp[2] - kshp[2]) / stride[0] + 1, (imshp[3] - kshp[3]) / stride[1] + 1)

    _test_feature_derivative_base(imshp=imshp,kshp=kshp,featshp=featshp,stride=stride,mask=mask)

def _test_feature_derivative_kshp_featshp(kshp=(10,2,10,10),featshp=(3,10,11,11),stride=(3,3),mask=False):

    #stride = (3,3)
    #kshp = (10,2,10,10) # features, channels, szy, szx
    #featshp = (3,10,11,11) # num images, features, szy, szx
    imshp = (featshp[0],kshp[1],featshp[2]*stride[0] + kshp[2] - 1,featshp[3]*stride[1] + kshp[3] - 1) # num images, channels, szy, szx

    _test_feature_derivative_base(imshp=imshp,kshp=kshp,featshp=featshp,stride=stride,mask=mask)

def _test_feature_derivative_base(imshp,kshp,featshp,stride,mask=False,tol=1e-4):

    #stride = (3,3)
    #kshp = (10,2,10,10) # features, channels, szy, szx
    #featshp = (3,10,11,11) # num images, features, szy, szx
    #imshp = (featshp[0],kshp[1],featshp[2]*stride[0] + kshp[2] - 1,featshp[3]*stride[1] + kshp[3] - 1) # num images, channels, szy, szx

    print 'Testing feature derivative with'
    print '    imshp  =', imshp
    print '    kshp   =', kshp
    print '    featshp=', featshp
    print '    strides=', stride
    print '    mask   =', mask

    if isinstance(imshp,list): imshp = tuple(imshp)
    if isinstance(kshp,list): kshp = tuple(kshp)
    if isinstance(featshp,list): featshp = tuple(featshp)

    assert imshp[1] == kshp[1]
    assert kshp[0] == featshp[1]
    assert (imshp[2] - kshp[2])/stride[0] + 1 == featshp[2]
    assert (imshp[3] - kshp[3])/stride[1] + 1 == featshp[3]

    features = np.random.randn(*featshp)
    kernel = np.random.randn(*kshp)
    image = np.random.randn(*imshp)

    theano_feature_derivative = get_theano_feature_derivative(imshp,kshp,featshp,stride=stride,mask=mask)
    scipy_derivative = scipy_feature_derivative(image,features,kernel,stride=stride)

    T_derivative = theano_feature_derivative(image,features,kernel)
    assert T_derivative.shape == featshp
    assert scipy_derivative.shape == T_derivative.shape

    np.testing.assert_allclose(scipy_derivative,T_derivative)

    theano_function = get_theano_function(imshp,kshp,featshp,stride=stride,mask=mask)
    def theano_function_features(features,image,kernel):
        return theano_function(image,features,kernel)

    def theano_grad_features(features,image,kernel):
        return theano_feature_derivative(image,features,kernel)

    scipy_function_features = lambda features, image, kernel: scipy_function(image,features,kernel,stride=stride)
    scipy_grad_features = lambda features, image, kernel: scipy_feature_derivative(image,features,kernel,stride=stride)

    difference, function_grad, numeric_grad = check_grad(theano_function_features,theano_grad_features,features,(image,kernel))
    assert difference < tol
    print 'Theano/Theano difference', difference

    difference, function_grad, numeric_grad = check_grad(scipy_function_features,scipy_grad_features,features,(image,kernel))
    assert difference < tol
    print 'Scipy/Scipy difference', difference

    difference, function_grad, numeric_grad = check_grad(theano_function_features,scipy_grad_features,features,(image,kernel))
    assert difference < tol
    print 'Theano/Scipy difference', difference

    difference, function_grad, numeric_grad = check_grad(scipy_function_features,theano_grad_features,features,(image,kernel))
    assert difference < tol
    print 'Scipy/Theano difference', difference

def get_theano_feature_derivative(imshp,kshp,featshp,stride,mask=True):

    from hdl.theano_methods import T_gl2_cost_conv

    image = T.tensor4()
    features = T.tensor4()
    kernel = T.tensor4()

    x = T.transpose(T.reshape(image,(imshp[0],np.prod(imshp[1:])),ndim=2))
    a = T.transpose(T.reshape(features,(featshp[0],np.prod(featshp[1:])),ndim=2))
    A = T.transpose(T.reshape(kernel, (kshp[0],np.prod(kshp[1:])),ndim=2))

    output = T_gl2_cost_conv(x,a,A,imshp,kshp,featshp,stride=stride,mask=mask)

    routput = T.reshape(T.transpose(output),featshp,ndim=4)

    return function(inputs=[image,features,kernel],outputs=routput)

def get_theano_function(imshp,kshp,featshp,stride,mask=True):

    from hdl.theano_methods import T_l2_cost_conv

    image = T.tensor4()
    features = T.tensor4()
    kernel = T.tensor4()

    x = T.transpose(T.reshape(image,(imshp[0],np.prod(imshp[1:])),ndim=2))
    a = T.transpose(T.reshape(features,(featshp[0],np.prod(featshp[1:])),ndim=2))
    A = T.transpose(T.reshape(kernel, (kshp[0],np.prod(kshp[1:])),ndim=2))

    output = T_l2_cost_conv(x,a,A,imshp,kshp,featshp,stride=stride,mask=mask)

    return function(inputs=[image,features,kernel],outputs=output)

def scipy_function(image,features,kernel,stride=(1,1)):
    _scipy_image_estimate = scipy_correlate4d(features,kernel,stride=stride)
    if not image.shape == _scipy_image_estimate.shape:
        if _scipy_image_estimate.shape[2] > image.shape[2]:
            sz_r = image.shape[2]
            sz_c = image.shape[3]
            scipy_image_estimate = _scipy_image_estimate[:,:,:sz_r,:sz_c]
        else:
            scipy_image_estimate = np.zeros_like(image)
            sz_r = _scipy_image_estimate.shape[2]
            sz_c = _scipy_image_estimate.shape[3]
            scipy_image_estimate[:,:,:sz_r,:sz_c] = _scipy_image_estimate
    else:
        scipy_image_estimate = _scipy_image_estimate
    scipy_error = image-scipy_image_estimate
    return .5*np.sum(scipy_error**2)

def scipy_feature_derivative(image,features,kernel,stride=(1,1)):

    _scipy_image_estimate = scipy_correlate4d(features,kernel,stride=stride)
    if not image.shape == _scipy_image_estimate.shape:
        if _scipy_image_estimate.shape[2] > image.shape[2]:
            sz_r = image.shape[2]
            sz_c = image.shape[3]
            scipy_image_estimate = _scipy_image_estimate[:,:,:sz_r,:sz_c]
        else:
            scipy_image_estimate = np.zeros_like(image)
            sz_r = _scipy_image_estimate.shape[2]
            sz_c = _scipy_image_estimate.shape[3]
            scipy_image_estimate[:,:,:sz_r,:sz_c] = _scipy_image_estimate
    else:
        scipy_image_estimate = _scipy_image_estimate
    scipy_error = image-scipy_image_estimate
    scipy_derivative = -scipy_convolve4d(scipy_error,kernel,stride=stride)
    return scipy_derivative

def scipy_correlate4d(features,kernel,stride=(1,1)):

    featshp = features.shape
    kshp = kernel.shape

    output_sz = (featshp[0], kshp[1], kshp[2] + stride[0]*featshp[2] - 1, kshp[3] + stride[1]*featshp[3] - 1)

    scipy_output = np.zeros(output_sz)
    for im_i in range(featshp[0]):
        for im_j in range(kshp[1]):
            for k_i in range(kshp[0]):
                if stride == (1,1):
                    feature = np.squeeze(features[im_i,k_i,:,:])
                else:
                    feature = np.zeros((featshp[2]*stride[0],featshp[3]*stride[1]),dtype=features.dtype)
                    feature[::stride[0],::stride[1]] = np.squeeze(features[im_i,k_i,:,:])
                scipy_output[im_i,im_j,:,:] += correlate2d(feature,np.squeeze(kernel[k_i,im_j,:,:]),mode='full')

    return scipy_output

def scipy_convolve4d(image,kernel,mode='valid',stride=(1,1)):

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

    if not stride == (1,1):
        scipy_output = scipy_output[:,:,::stride[0],::stride[1]]

    return scipy_output

def convolve4d_view(image,kernel,mode='valid',stride=(1,1)):

    from skimage.util.shape import view_as_windows

    imshp = image.shape
    kshp = kernel.shape

    offset = None
    if mode=='valid':
        featshp = (imshp[0],kshp[0],(imshp[2] - kshp[2])/stride[0] + 1,(imshp[3] - kshp[3])/stride[1] + 1) # num images, features, szy, szx
    elif mode == 'same':
        assert stride == (1,1)
        featshp = (imshp[0],kshp[0],imshp[2],imshp[3]) # num images, features, szy, szx
        offset = (kshp[2]/2, kshp[3]/2)
    #elif mode == 'full':
    #    assert stride == (1,1)
    #    featshp = (imshp[0],kshp[0],imshp[2] + kshp[2] - 1,imshp[3] + kshp[3] - 1) # num images, features, szy, szx
    else:
        raise NotImplemented, 'Unkonwn mode %s'%mode

    kernel_flipped = kernel[:,:,::-1,::-1]

    output = np.zeros(featshp)
    this_image = None
    for im_i in range(imshp[0]):

        if mode == 'valid':
            this_image = image[im_i,...]
        elif mode == 'same':
            if this_image is None:
                this_image_shp = (imshp[1], imshp[2] + kshp[2] - 1, imshp[3] + kshp[3] - 1)
                this_image = np.zeros(this_image_shp,dtype=image.dtype)
            this_image[:,offset[0]:(offset[0]+imshp[2]),offset[1]:(offset[1]+imshp[3])] = image[im_i,...]
        else:
            raise NotImplemented
        imager = view_as_windows(this_image,(kshp[1],kshp[2],kshp[3]))[0,::stride[0],::stride[1],...]
        # imager.shape = (featszr, featszc, channels, ksz[2], ksz[3])
        feat = np.tensordot(kernel_flipped,imager,axes=((1,2,3),(2,3,4)))

        output[im_i,...] = feat

    return output

def _test_kernel_derivative_imshp_kshp(imshp=(2,1,16,16),kshp=(1,2,5,5),stride=(1,1),mask=False):

    assert imshp[1] == kshp[1]

    featshp = (imshp[0],kshp[0],(imshp[2] - kshp[2]) / stride[0] + 1, (imshp[3] - kshp[3]) / stride[1] + 1)

    _test_kernel_derivative_base(imshp=imshp,kshp=kshp,featshp=featshp,stride=stride,mask=mask)

def _test_kernel_derivative_kshp_featshp(kshp=(10,2,10,10),featshp=(3,10,11,11),stride=(3,3),mask=False):

    #stride = (3,3)
    #kshp = (10,2,10,10) # features, channels, szy, szx
    #featshp = (3,10,11,11) # num images, features, szy, szx
    imshp = (featshp[0],kshp[1],featshp[2]*stride[0] + kshp[2] - 1,featshp[3]*stride[1] + kshp[3] - 1) # num images, channels, szy, szx

    _test_kernel_derivative_base(imshp=imshp,kshp=kshp,featshp=featshp,stride=stride,mask=mask)

def _test_kernel_derivative_base(imshp,kshp,featshp,stride,mask=False,tol=1e-4):

    print 'Testing kernel derivative with'
    print '    imshp  =', imshp
    print '    kshp   =', kshp
    print '    featshp=', featshp
    print '    strides=', stride
    print '    mask   =', mask

    if isinstance(imshp,list): imshp = tuple(imshp)
    if isinstance(kshp,list): kshp = tuple(kshp)
    if isinstance(featshp,list): featshp = tuple(featshp)

    assert imshp[1] == kshp[1]
    assert kshp[0] == featshp[1]
    assert (imshp[2] - kshp[2])/stride[0] + 1 == featshp[2]
    assert (imshp[3] - kshp[3])/stride[1] + 1 == featshp[3]

    features = np.random.randn(*featshp)
    kernel = np.random.randn(*kshp)
    image = np.random.randn(*imshp)

    theano_kernel_derivative = get_theano_kernel_derivative(imshp,kshp,featshp,stride=stride,mask=mask)

    scipy_derivative = scipy_kernel_derivative(image,features,kernel,stride=stride)

    T_derivative = theano_kernel_derivative(image,features,kernel)
    assert T_derivative.shape == kshp
    assert scipy_derivative.shape == T_derivative.shape

    theano_function = get_theano_function(imshp,kshp,featshp,stride=stride,mask=mask)
    theano_function_kernel = lambda kernel, image, features: theano_function(image,features,kernel)
    theano_grad_kernel = lambda kernel, image, features: theano_kernel_derivative(image,features,kernel)

    scipy_function_kernel = lambda kernel, image, features: scipy_function(image,features,kernel,stride=stride)
    scipy_grad_kernel = lambda kernel, image, features: scipy_kernel_derivative(image,features,kernel,stride=stride)

    difference, function_grad, numeric_grad = check_grad(theano_function_kernel,theano_grad_kernel,kernel,(image,features))
    print 'Theano/Theano difference', difference
    assert difference < tol

    difference, function_grad, numeric_grad = check_grad(scipy_function_kernel,scipy_grad_kernel,kernel,(image,features))
    print 'Scipy/Scipy difference', difference
    assert difference < tol

    difference, function_grad, numeric_grad = check_grad(theano_function_kernel,scipy_grad_kernel,kernel,(image,features))
    print 'Theano/Scipy difference', difference
    #print 'function_grad\n', function_grad
    #print 'numeric_grad\n', numeric_grad
    assert difference < tol

    difference, function_grad, numeric_grad = check_grad(scipy_function_kernel,theano_grad_kernel,kernel,(image,features))
    print 'Scipy/Theano difference', difference
    assert difference < tol

def get_theano_kernel_derivative(imshp,kshp,featshp,stride,mask=True):

    image = T.tensor4()
    features = T.tensor4()
    kernel = T.tensor4()

    x = T.transpose(T.reshape(image,(imshp[0],np.prod(imshp[1:])),ndim=2))
    a = T.transpose(T.reshape(features,(featshp[0],np.prod(featshp[1:])),ndim=2))
    A = T.transpose(T.reshape(kernel, (kshp[0],np.prod(kshp[1:])),ndim=2))

    if stride == (1,1):
        from hdl.theano_methods import T_l2_cost_conv
        cost = T_l2_cost_conv(x,a,A,imshp=imshp,kshp=kshp,featshp=featshp,stride=stride,mask=mask)
        grad_A = T.grad(cost,A)
        rgrad_A = T.reshape(T.transpose(grad_A),kshp)
        return function(inputs=[image,features,kernel],outputs=rgrad_A)
    else:
        from hdl.theano_methods import T_l2_cost_conv_dA
        grad_A = T_l2_cost_conv_dA(x,a,A,imshp=imshp,kshp=kshp,featshp=featshp,stride=stride,mask=mask)
        rgrad_A = T.reshape(T.transpose(grad_A),kshp)
        return function(inputs=[image,features,kernel],outputs=rgrad_A)

def scipy_kernel_derivative(image,features,kernel,stride=(1,1)):

    _scipy_image_estimate = scipy_correlate4d(features,kernel,stride=stride)
    if not image.shape == _scipy_image_estimate.shape:
        if _scipy_image_estimate.shape[2] > image.shape[2]:
            sz_r = image.shape[2]
            sz_c = image.shape[3]
            scipy_image_estimate = _scipy_image_estimate[:,:,:sz_r,:sz_c]
        else:
            scipy_image_estimate = np.zeros_like(image)
            sz_r = _scipy_image_estimate.shape[2]
            sz_c = _scipy_image_estimate.shape[3]
            scipy_image_estimate[:,:,:sz_r,:sz_c] = _scipy_image_estimate
    else:
        scipy_image_estimate = _scipy_image_estimate

    scipy_error = image-scipy_image_estimate

    scipy_error_rot = np.transpose(scipy_error,[1,0,2,3])
    features_rot = np.transpose(features,[1,0,2,3])

    feat_expand = np.zeros((features_rot.shape[0],
                            features_rot.shape[1],
                            image.shape[2] - kernel.shape[2] + 1,
                            image.shape[3] - kernel.shape[3] + 1),dtype=features.dtype)
    feat_expand[:,:,::stride[0],::stride[1]] = features_rot

    scipy_derivative_rot = -scipy_convolve4d(scipy_error_rot[:,:,::-1,::-1],feat_expand)
    scipy_derivative = np.transpose(scipy_derivative_rot,[1,0,2,3])

    return scipy_derivative

def test_feature_derivative_driver():
    #kshp = features, channels, szy, szx
    #featshp = num images, features, szy, szx
    #imshp = num images, channels, szy, szx

    # TODO: test dims equal to 1

    strides = [
        (1,1),
        (2,2),
        (3,3)]
    num_features = [2,3,4]
    num_channels = [2,3]
    num_ims = [2]
    feat_szs = [(2,2), (3,3), (4,4)]
    k_szs = [(2,2), (3,3), (4,4), (5,5)]

    from itertools import product

    for stride, num_feature, num_channel, num_im, feat_sz, k_sz in product(strides,num_features,num_channels,num_ims,feat_szs,k_szs):
        kshp = (num_feature,num_channel,k_sz[0],k_sz[1])
        featshp = (num_im,num_feature,feat_sz[0],feat_sz[1])
        print '-'*30
        _test_feature_derivative_kshp_featshp(kshp=kshp,featshp=featshp,stride=stride)

    im_szs = [(10,10), (11,11), (12,12)]

    for stride, num_feature, num_channel, num_im, im_sz, k_sz in product(strides,num_features,num_channels,num_ims,im_szs,k_szs):
        imshp = (num_im,num_channel,im_sz[0],im_sz[1])
        kshp = (num_feature,num_channel,k_sz[0],k_sz[1])
        print '-'*30
        _test_feature_derivative_imshp_kshp(imshp=imshp,kshp=kshp,stride=stride)

def test_kernel_derivative_driver():
    #kshp = features, channels, szy, szx
    #featshp = num images, features, szy, szx
    #imshp = num images, channels, szy, szx

    # TODO: test dims equal to 1

    strides = [
        (1,1),
        (2,2),
        (3,3)]
    num_features = [2,3,4]
    num_channels = [2,3]
    num_ims = [2]
    feat_szs = [(2,2), (3,3), (4,4)]
    k_szs = [(2,2), (3,3), (4,4), (5,5)]

    from itertools import product

    for stride, num_feature, num_channel, num_im, feat_sz, k_sz in product(strides,num_features,num_channels,num_ims,feat_szs,k_szs):
        kshp = (num_feature,num_channel,k_sz[0],k_sz[1])
        featshp = (num_im,num_feature,feat_sz[0],feat_sz[1])
        print '-'*30
        _test_kernel_derivative_kshp_featshp(kshp=kshp,featshp=featshp,stride=stride)

    im_szs = [(10,10), (11,11), (12,12)]

    for stride, num_feature, num_channel, num_im, im_sz, k_sz in product(strides,num_features,num_channels,num_ims,im_szs,k_szs):
        imshp = (num_im,num_channel,im_sz[0],im_sz[1])
        kshp = (num_feature,num_channel,k_sz[0],k_sz[1])
        print '-'*30
        _test_kernel_derivative_imshp_kshp(imshp=imshp,kshp=kshp,stride=stride)

def test_convolve4d_view():

    modes = ['valid', 'same']
    strides = [
        (1,1),
        (2,2),
        (3,3),
        (4,4)
        ]
    num_features = [2,3,4]
    num_channels = [2,3]
    num_ims = [2]
    k_szs = [(2,2), (3,3), (4,4), (5,5), (7,7)]

    from itertools import product
    from time import time as now

    im_szs = [(10,10), (11,11), (12,12), (24,24)]

    for mode, stride, num_feature, num_channel, num_im, im_sz, k_sz in product(modes,strides,num_features,num_channels,num_ims,im_szs,k_szs):
        if mode == 'same' and not stride == (1,1): continue
        imshp = (num_im,num_channel,im_sz[0],im_sz[1])
        kshp = (num_feature,num_channel,k_sz[0],k_sz[1])
        image = np.random.randn(*imshp)
        kernel = np.random.randn(*kshp)
        t0 = now()
        a = scipy_convolve4d(image=image,kernel=kernel,stride=stride,mode=mode)
        a_time = now() - t0
        t0 = now()
        b = convolve4d_view(image=image,kernel=kernel,stride=stride,mode=mode)
        b_time = now() - t0
        print 'mode %s, scipy time: %2.3e, numpy view time: %2.3e, fraction = %2.2f'%(mode, a_time,b_time,a_time/b_time)

        np.testing.assert_almost_equal(a,b)

if __name__ == '__main__':
    #test_feature_derivative_driver()
    #test_kernel_derivative_driver()
    test_convolve4d_view()