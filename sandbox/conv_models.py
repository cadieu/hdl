import numpy as np

import theano
from theano import tensor as T
from theano import function, Param, shared, In, Out, sandbox
from theano.tensor.nnet import conv2d

from scipy.signal import correlate2d, convolve2d

from hdl.models import ConvWhitenInputModel, ConvSparseSlowModel

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

def test_conv():

    stride = 4
    imshp = (1,1,20,20)
    kshp = (10,1,10,10)

    theano_convolve2d = get_theano_convolve2d(imshp,kshp,stride)

    image = np.random.randn(*imshp)
    kernel = np.random.randn(*kshp)

    scipy_output = scipy_convolve4d(image,kernel,stride=stride)
    theano_output = theano_convolve2d(image,kernel)

    print 'scipy:', scipy_output.shape
    print 'theano:', theano_output.shape

    np.testing.assert_allclose(scipy_output,theano_output)


def get_theano_convolve2d(imshp,kshp,stride=1):
    image = T.tensor4(dtype=theano.config.floatX)
    kernel = T.tensor4(dtype=theano.config.floatX)

    output = conv2d(image,kernel,image_shape=imshp,filter_shape=kshp,subsample=(stride,stride))

    theano_convolve2d = function(inputs=[image,kernel],outputs=output)

    return theano_convolve2d


def test_corr():

    stride = 4
    #imshp = (3,2,20,20) # num images, channels, szy, szx
    kshp = (10,2,10,10) # features, channels, szy, szx
    featshp = (3,10,11,11) # num images, features, szy, szx

    theano_correlate2d = get_theano_correlate2d(kshp=kshp,featshp=featshp,stride=stride)

    features = np.random.randn(*featshp)
    kernel = np.random.randn(*kshp)

    scipy_output = scipy_correlate4d(features,kernel,stride=stride)
    theano_output = theano_correlate2d(features,kernel)

    print 'scipy:', scipy_output.shape
    print 'theano:', theano_output.shape

    np.testing.assert_allclose(scipy_output,theano_output)

def get_theano_correlate2d(kshp,featshp,stride=1):

    features = T.tensor4(dtype=theano.config.floatX)
    kernel = T.tensor4(dtype=theano.config.floatX)

    # Need to transpose first two dimensions of kernel, and reverse index kernel image dims (for correlation)
    kernel_rotated = T.transpose(kernel[:,:,::-1,::-1],axes=[1,0,2,3])

    # Method indexing:
    #strided_features = T.zeros((featshp[0],featshp[1],featshp[2]*stride,featshp[3]*stride),dtype=theano.config.floatX)
    #strided_features = T.set_subtensor(strided_features[:,:,::stride,::stride],features)
    #image = conv2d(strided_features,kernel_rotated,border_mode='full')

    # Method striding:
    featshp_logical = (featshp[0],featshp[1],featshp[2]*stride,featshp[3]*stride)
    kshp_rotated = (kshp[1], kshp[0], kshp[2], kshp[3])
    image = conv2d(features,kernel_rotated,border_mode='full',
        image_shape=featshp,filter_shape=kshp_rotated,
        imshp_logical=featshp_logical[1:],kshp_logical=kshp[2:])

    gen_image = function(inputs=[features,kernel],outputs=image)

    return gen_image

def _scipy_function(image,features,kernel,stride=1):
    scipy_image_estimate = scipy_correlate4d(features,kernel,stride=stride)
    scipy_error = image-scipy_image_estimate
    return .5*np.sum(scipy_error**2)

def test_derivative(kshp=(10,2,10,10),featshp=(3,10,11,11),stride=3):

    #stride = 3
    #kshp = (10,2,10,10) # features, channels, szy, szx
    #featshp = (3,10,11,11) # num images, features, szy, szx
    imshp = (featshp[0],kshp[1],featshp[2]*stride + kshp[2] - 1,featshp[3]*stride + kshp[3] - 1) # num images, channels, szy, szx

    feature_derivative = theano_derivative(imshp,kshp,featshp,stride=stride)

    features = np.random.randn(*featshp)
    kernel = np.random.randn(*kshp)
    image = np.random.randn(*imshp)

    print 'featshp formula:', (imshp[2] - kshp[2] + 1)/stride, featshp[2]

    def scipy_feature_derivative(image,features,kernel,stride=1):

        scipy_image_estimate = scipy_correlate4d(features,kernel,stride=stride)
        scipy_error = image-scipy_image_estimate
        scipy_derivative = -scipy_convolve4d(scipy_error,kernel,stride=stride)
        return scipy_derivative

    scipy_derivative = scipy_feature_derivative(image,features,kernel,stride=stride)

    T_derivative = feature_derivative(image,features,kernel)
    print 'derivative of features:', T_derivative.shape, ', expected:', featshp

    print 'scipy:', scipy_derivative.shape
    print 'theano:', T_derivative.shape

    np.testing.assert_allclose(scipy_derivative,T_derivative)

    theano_function = get_theano_function(imshp,kshp,featshp,stride=stride)
    theano_function_features = lambda features, image, kernel: theano_function(image,features,kernel)
    theano_grad_features = lambda features, image, kernel: feature_derivative(image,features,kernel)

    scipy_function_features = lambda features, image, kernel: _scipy_function(image,features,kernel,stride=stride)
    scipy_grad_features = lambda features, image, kernel: scipy_feature_derivative(image,features,kernel,stride=stride)

    difference, function_grad, numeric_grad = check_grad(theano_function_features,theano_grad_features,features,(image,kernel))
    print 'Theano/Theano difference', difference

    difference, function_grad, numeric_grad = check_grad(scipy_function_features,scipy_grad_features,features,(image,kernel))
    print 'Scipy/Scipy difference', difference

    difference, function_grad, numeric_grad = check_grad(theano_function_features,scipy_grad_features,features,(image,kernel))
    print 'Theano/Scipy difference', difference

    difference, function_grad, numeric_grad = check_grad(scipy_function_features,theano_grad_features,features,(image,kernel))
    print 'Scipy/Theano difference', difference

    #import IPython; ipshell = IPython.embed; ipshell(banner1='ipshell')

def theano_derivative(imshp,kshp,featshp,stride=1):

    features = T.tensor4(dtype=theano.config.floatX)
    kernel = T.tensor4(dtype=theano.config.floatX)
    image = T.tensor4(dtype=theano.config.floatX)

    # Need to transpose first two dimensions of kernel, and reverse index kernel image dims (for correlation)
    kernel_rotated = T.transpose(kernel[:,:,::-1,::-1],axes=[1,0,2,3])

    featshp_logical = (featshp[0],featshp[1],featshp[2]*stride,featshp[3]*stride)
    kshp_rotated = (kshp[1], kshp[0], kshp[2], kshp[3])
    image_estimate = conv2d(features,kernel_rotated,border_mode='full',
                            image_shape=featshp,filter_shape=kshp_rotated,
                            imshp_logical=featshp_logical[1:],kshp_logical=kshp[2:])

    image_error = image - image_estimate

    feature_grad = -conv2d(image_error,kernel,image_shape=imshp,filter_shape=kshp,subsample=(stride,stride))
    return function(inputs=[image,features,kernel],outputs=feature_grad)

def get_theano_function(imshp,kshp,featshp,stride=1):

    features = T.tensor4(dtype=theano.config.floatX)
    kernel = T.tensor4(dtype=theano.config.floatX)
    image = T.tensor4(dtype=theano.config.floatX)

    # Need to transpose first two dimensions of kernel, and reverse index kernel image dims (for correlation)
    kernel_rotated = T.transpose(kernel[:,:,::-1,::-1],axes=[1,0,2,3])

    featshp_logical = (featshp[0],featshp[1],featshp[2]*stride,featshp[3]*stride)
    kshp_rotated = (kshp[1], kshp[0], kshp[2], kshp[3])
    image_estimate = conv2d(features,kernel_rotated,border_mode='full',
                            image_shape=featshp,filter_shape=kshp_rotated,
                            imshp_logical=featshp_logical[1:],kshp_logical=kshp[2:])

    image_error = image - image_estimate

    f = .5*T.sum(image_error**2)

    return function(inputs=[image,features,kernel],outputs=f)

def test_derivative_driver():

    strides = [1,2,3,5,8]
    kshp_featshp = [((10, 2,10,10), ( 3,10,11,11)),
                    (( 1, 9,10,10), ( 3, 1,11,11)),
                    (( 3, 2, 5, 5), ( 3, 3, 8, 8))]

    for stride in strides:
        for kshp, featshp in kshp_featshp:
            test_derivative(kshp=kshp,featshp=featshp,stride=stride)

def test_kernel_derivative(kshp=(10,2,10,10),featshp=(3,10,11,11),stride=1,epsilon=1e-5):

    #stride = 3
    #kshp = (10,2,10,10) # features, channels, szy, szx
    #featshp = (3,10,11,11) # num images, features, szy, szx
    imshp = (featshp[0],kshp[1],featshp[2]*stride + kshp[2] - 1,featshp[3]*stride + kshp[3] - 1) # num images, channels, szy, szx
    print 'imshp', imshp, 'kshp', kshp, 'featshp', featshp, 'stride', stride

    kernel_derivative = theano_kernel_derivative(imshp,kshp,featshp,stride=stride)

    features = np.random.randn(*featshp)
    kernel = np.random.randn(*kshp)
    image = np.random.randn(*imshp)

    # debug:
    #features[:] = 0.
    #features[0,0,0,0] = 1.
    #image[:] = 0.
    #image[0,0,0,0] = 1.
    #print 'kernel\n', kernel[0,0,:,:]

    print 'featshp formula:', (imshp[2] - kshp[2] + 1)/stride, featshp[2]

    def scipy_kernel_derivative(image,features,kernel,stride=1):

        scipy_image_estimate = scipy_correlate4d(features,kernel,stride=stride)
        scipy_error = image-scipy_image_estimate

        scipy_error_rot = np.transpose(scipy_error,[1,0,2,3])
        features_rot = np.transpose(features,[1,0,2,3])

        feat_expand = np.zeros((features_rot.shape[0],features_rot.shape[1],image.shape[2] - kernel.shape[2] + 1,image.shape[3] - kernel.shape[3] + 1),dtype=features.dtype)
        feat_expand[:,:,::stride,::stride] = features_rot

        scipy_derivative_rot = -scipy_convolve4d(scipy_error_rot[:,:,::-1,::-1],feat_expand)
        scipy_derivative = np.transpose(scipy_derivative_rot,[1,0,2,3])

        print 'scipy_derivative.shape', scipy_derivative.shape, 'expected', kernel.shape
        #scipy_derivative = scipy_derivative[:,:,:kernel.shape[2],:kernel.shape[3]]

        return scipy_derivative

    scipy_derivative = scipy_kernel_derivative(image,features,kernel,stride=stride)

    T_derivative = kernel_derivative(image,features,kernel)
    assert T_derivative.shape == kshp
    print 'derivative of features:', T_derivative.shape, ', expected:', kshp

    print 'scipy_derivative.shape', scipy_derivative.shape, 'T_derivative', T_derivative.shape

    theano_function = get_theano_function(imshp,kshp,featshp,stride=stride)
    theano_function_kernel = lambda kernel, image, features: theano_function(image,features,kernel)
    theano_grad_kernel = lambda kernel, image, features: kernel_derivative(image,features,kernel)

    scipy_function_kernel = lambda kernel, image, features: _scipy_function(image,features,kernel,stride=stride)
    scipy_grad_kernel = lambda kernel, image, features: scipy_kernel_derivative(image,features,kernel,stride=stride)

    difference, function_grad, numeric_grad = check_grad(theano_function_kernel,theano_grad_kernel,kernel,(image,features),epsilon=epsilon)
    print 'Theano/Theano difference', difference

    difference, function_grad, numeric_grad = check_grad(scipy_function_kernel,scipy_grad_kernel,kernel,(image,features),epsilon=epsilon)
    print 'Scipy/Scipy difference', difference
    #print 'function_grad\n', function_grad
    #print 'numeric_grad\n', numeric_grad

    difference, function_grad, numeric_grad = check_grad(theano_function_kernel,scipy_grad_kernel,kernel,(image,features),epsilon=epsilon)
    print 'Theano/Scipy difference', difference

    difference, function_grad, numeric_grad = check_grad(scipy_function_kernel,theano_grad_kernel,kernel,(image,features),epsilon=epsilon)
    print 'Scipy/Theano difference', difference

    #np.testing.assert_allclose(scipy_derivative,T_derivative)

def theano_kernel_derivative(imshp,kshp,featshp,stride=1):

    features = T.tensor4(dtype=theano.config.floatX)
    kernel = T.tensor4(dtype=theano.config.floatX)
    image = T.tensor4(dtype=theano.config.floatX)

    # Need to transpose first two dimensions of kernel, and reverse index kernel image dims (for correlation)
    kernel_rotated = T.transpose(kernel[:,:,::-1,::-1],axes=[1,0,2,3])

    featshp_logical = (featshp[0],featshp[1],featshp[2]*stride,featshp[3]*stride)
    kshp_rotated = (kshp[1], kshp[0], kshp[2], kshp[3])
    image_estimate = conv2d(features,kernel_rotated,border_mode='full',
                            image_shape=featshp,filter_shape=kshp_rotated,
                            imshp_logical=featshp_logical[1:],kshp_logical=kshp[2:])

    image_error = image - image_estimate

    image_error_rot = T.transpose(image_error,[1,0,2,3])[:,:,::-1,::-1]
    imshp_rot = (imshp[1],imshp[0],imshp[2],imshp[3])
    featshp_rot = (featshp[1],featshp[0],featshp[2],featshp[3])
    features_rot = T.transpose(features,[1,0,2,3])

    featshp_rot_logical = (featshp_rot[0],featshp_rot[1],featshp_rot[2]*stride,featshp_rot[3]*stride)
    kernel_grad_rot = -conv2d(image_error_rot,features_rot,
                              image_shape=imshp_rot,filter_shape=featshp_rot,
                              imshp_logical=imshp_rot[1:],kshp_logical=featshp_rot_logical[2:])
    kernel_grad = T.transpose(kernel_grad_rot,[1,0,2,3])

    return function(inputs=[image,features,kernel],outputs=kernel_grad)

def test_kernel_derivative_driver():

    strides = [2,3,5,8]
    kshp_featshp = [((10, 2,10,10), ( 3,10,11,11)),
                    (( 1, 9,10,10), ( 3, 1,11,11)),
                    (( 3, 2, 5, 5), ( 3, 3, 8, 8))]

    for stride in strides:
        for kshp, featshp in kshp_featshp:
            print '-'*20
            print 'stride', stride, 'kshp', kshp, 'featshp', featshp
            test_kernel_derivative(kshp=kshp,featshp=featshp,stride=stride)

def scipy_convolve4d(image,kernel,stride=1):

    imshp = image.shape
    kshp = kernel.shape
    featshp = (imshp[0],kshp[0],imshp[2] - kshp[2] + 1,imshp[3] - kshp[3] + 1) # num images, features, szy, szx

    scipy_output = np.zeros(featshp)
    for im_i in range(imshp[0]):
        for k_i in range(kshp[0]):
            for im_j in range(imshp[1]):
                scipy_output[im_i,k_i,:,:] += convolve2d(np.squeeze(image[im_i,im_j,:,:]),np.squeeze(kernel[k_i,im_j,:,:]),mode='valid')

    if not stride == 1:
        scipy_output = scipy_output[:,:,::stride,::stride]

    return scipy_output

def scipy_correlate4d(features,kernel,stride=1):

    featshp = features.shape
    kshp = kernel.shape

    output_sz = (featshp[0], kshp[1], kshp[2] + stride*featshp[2] - 1, kshp[3] + stride*featshp[3] - 1)

    scipy_output = np.zeros(output_sz)
    for im_i in range(featshp[0]):
        for im_j in range(kshp[1]):
            for k_i in range(kshp[0]):
                if stride == 1:
                    feature = np.squeeze(features[im_i,k_i,:,:])
                else:
                    feature = np.zeros((featshp[2]*stride,featshp[3]*stride),dtype=features.dtype)
                    feature[::stride,::stride] = np.squeeze(features[im_i,k_i,:,:])
                scipy_output[im_i,im_j,:,:] += correlate2d(feature,np.squeeze(kernel[k_i,im_j,:,:]),mode='full')

    return scipy_output

def get_theano_correlate_view(strides=(1,1)):

    from myconv import MyCorr

    features = T.tensor4(dtype=theano.config.floatX)
    kernel = T.tensor4(dtype=theano.config.floatX)

    output = MyCorr(strides=strides)(features,kernel)

    f = function(inputs=[features,kernel],outputs=output)
    return f

def scipy_correlate4d_view(features,kernel,stride=1):

    #imshp = (3,2,20,20) # num images, channels, szy, szx
    #kshp = (10,2,10,10) # features, channels, szy, szx
    #featshp = (3,10,11,11) # num images, features, szy, szx

    from skimage.util.shape import view_as_windows

    featshp = features.shape
    kshp = kernel.shape

    output_sz = (featshp[0], kshp[1], kshp[2] + stride*featshp[2] - 1, kshp[3] + stride*featshp[3] - 1)

    k_rot = kernel[:,:,::-1,::-1]

    scipy_output = np.zeros(output_sz)
    for im_i in range(featshp[0]):

        im_out = np.zeros(output_sz[1:])
        im_outr = view_as_windows(im_out,(kshp[1],kshp[2],kshp[3]))[0,::stride,::stride,...]

        im_hatr = np.tensordot(np.squeeze(features[im_i,...]),k_rot,axes=((0,),(0,)))
        #import IPython; ipshell = IPython.embed; ipshell(banner1='ipshell')

#        for a in range(im_hatr.shape[0]):
#            im_outr[a,:,...] += im_hatr[a,:,...]

        for a in range(im_hatr.shape[0]):
            for b in range(im_hatr.shape[1]):
                im_outr[a,b,...] += im_hatr[a,b,...]

#        for d in range(im_hatr.shape[3]):
#            for e in range(im_hatr.shape[4]):
#                im_outr[:,:,:,d,e] += im_hatr[:,:,:,d,e]

#                for c in range(im_hatr.shape[2]):
#                    for d in range(im_hatr.shape[3]):
#                        for e in range(im_hatr.shape[4]):
#                            im_outr[a,b,c,d,e] += im_hatr[a,b,c,d,e]
#        im_outr += im_hatr

        scipy_output[im_i,...] = im_out

    return scipy_output

def test_scipy_view_method(kshp=(16,1,16,16),featshp=(2,16,8,8),stride=8):
    #stride = 3
    #kshp = (10,2,10,10) # features, channels, szy, szx
    #featshp = (3,10,11,11) # num images, features, szy, szx
    #imshp = (featshp[0],kshp[1],featshp[2]*stride + kshp[2] - 1,featshp[3]*stride + kshp[3] - 1) # num images, channels, szy, szx

    from time import time as now

    features = np.random.randn(*featshp)
    kernel = np.random.randn(*kshp)
    features *= 0.
    features[0,0,0,0] = 1.

    scipy_image_estimate = scipy_correlate4d(features,kernel,stride=stride)
    scipy_image_estimate_view = scipy_correlate4d_view(features,kernel,stride=stride)

    theano_correlate_view = get_theano_correlate_view(strides=(stride,stride))
    theano_image_estimate_view = theano_correlate_view(features,kernel)

    print scipy_image_estimate[0,0,:3,:3]
    print scipy_image_estimate_view[0,0,:3,:3]
    print theano_image_estimate_view[0,0,:3,:3]

    np.testing.assert_almost_equal(scipy_image_estimate,scipy_image_estimate_view,decimal=3)
    np.testing.assert_almost_equal(scipy_image_estimate,theano_image_estimate_view,decimal=3)

    theano_correlate2d = get_theano_correlate2d(kshp=kshp,featshp=featshp,stride=stride)

    reps = 3

#    t0 = now()
#    for rep in range(reps):
#        scipy_correlate4d(features,kernel,stride=stride)
#    print 'scipy_correlate4d', now() - t0

    t0 = now()
    for rep in range(reps):
        scipy_correlate4d_view(features,kernel,stride=stride)
    print 'scipy_correlate4d_view', now() - t0

    t0 = now()
    for rep in range(reps):
        theano_correlate_view(features,kernel)
    print 'theano_correlate_view', now() - t0

    t0 = now()
    for rep in range(reps):
        theano_correlate2d(features,kernel)
    print 'theano_correlate2d', now() - t0

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

def test_convolve4d_view():



def test_convsparsenet(lam_sparse=.1,N=16,perc_var=100.,stride=1,kshp=(7,7),batchsize=4):

    from hdl.learners import SGD

    whitenpatches = 1000
    if isinstance(kshp,int): kshp = (kshp,kshp)
    if isinstance(stride,int): stride = (stride,stride)
    imszr = 5*stride[0] + kshp[0] - 1
    imszc = 5*stride[1] + kshp[1] - 1

    #model = ConvWhitenInputModel(imshp=(10,1,32,32),convwhitenfiltershp=(7,7),perc_var=100.)
    model = ConvSparseSlowModel(imshp=(batchsize,1,imszr,imszc),convwhitenfiltershp=(7,7),N=N,kshp=kshp,stride=stride,
        perc_var=perc_var,lam_sparse=lam_sparse)

    l = SGD(model=model,datasource='vid075-chunks',display_every=50,save_every=10000,batchsize=model.imshp[0])

    print 'whitenpatches', whitenpatches
    print 'model.imshp', model.imshp
    print 'model.convwhitenfiltershp', model.convwhitenfiltershp

    databatch = l.get_databatch(whitenpatches)
    l.model.learn_whitening(databatch)

    l.model.setup()

    l.learn(iterations=5000)
    l.model.center_basis_functions = False
    l.learn(iterations=15000)
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
    lam_sparses = [.01, .1, 1.0]
    #Ns = [32, 48, 64, 80, 96, 112, 128]
    Ns = [32, 64]
    #perc_vars = [80., 95., 99., 99.5, 99.9]
    perc_vars = [99., 99.5, 100.]
    stride = 8
    kshp = (16,16)
    for lam_sparse in lam_sparses:
        for N in Ns:
            for perc_var in perc_vars:
                test_convsparsenet(lam_sparse=lam_sparse,N=N,perc_var=perc_var,kshp=kshp,stride=stride)

def test_convsparsenet_subspace(lam_sparse=1.,lam_slow=1.,N=8,perc_var=100.,stride=(1,1)):

    from hdl.models import ConvSparseSlowModel
    from hdl.learners import SGD

    whitenpatches = 1000
    psz = 48
    ksz = 16
    convwhitenfiltershp=(15,15)

    #model = ConvWhitenInputModel(imshp=(10,1,32,32),convwhitenfiltershp=(7,7),perc_var=100.)
    model = ConvSparseSlowModel(imshp=(4,1,psz,psz),convwhitenfiltershp=convwhitenfiltershp,N=N,kshp=(ksz,ksz),stride=stride,
        sparse_cost='subspacel1',
        perc_var=perc_var,
        lam_sparse=lam_sparse,
        lam_slow=lam_slow,
        mask=True,
        force_subspace_orthogonal=True)

    l = SGD(model=model,datasource='vid075-chunks',display_every=50,save_every=10000,
            eta_target_maxupdate=0.5,
            batchsize=model.imshp[0])

    print 'whitenpatches', whitenpatches
    print 'model.imshp', model.imshp
    print 'model.convwhitenfiltershp', model.convwhitenfiltershp

    databatch = l.get_databatch(whitenpatches)
    l.model.learn_whitening(databatch)

    l.model.setup()

    l.learn(iterations=1000)
    l.model.center_basis_functions=False
    l.learn(iterations=9000)
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

def test_sparsenet_subspace(lam_sparse=1.,lam_slow=1.,N=8,perc_var=100.):

    from hdl.models import SparseSlowModel
    from hdl.learners import SGD

    whitenpatches = 10000
    psz = 16

    #model = ConvWhitenInputModel(imshp=(10,1,32,32),convwhitenfiltershp=(7,7),perc_var=100.)
    model = SparseSlowModel(patch_sz=psz,N=N,
        sparse_cost='subspacel1',
        perc_var=perc_var,
        lam_sparse=lam_sparse,
        lam_slow=lam_slow,T=48)

    l = SGD(model=model,datasource='vid075-chunks',display_every=100,save_every=10000,batchsize=model.T)

    print 'whitenpatches', whitenpatches

    databatch = l.get_databatch(whitenpatches)
    l.model.learn_whitening(databatch)

    l.model.setup()

    l.learn(iterations=5000)
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

def orig_helper_T_l2_cost_conv(x,a,A,imshp,kshp,featshp,stride=(1,1),mask=True):
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

    # Need to transpose first two dimensions of kernel, and reverse index kernel image dims (for correlation)
    kernel_rotated = T.transpose(kernel[:,:,::-1,::-1],axes=[1,0,2,3])

    #kernel_fill = T.zeros_like(kernel_rotated)
    #kernel_rotated = T.fill(kernel_fill,kernel_rotated)

    featshp_logical = (featshp[0],featshp[1],featshp[2]*stride[0],featshp[3]*stride[1])
    kshp_rotated = (kshp[1], kshp[0], kshp[2], kshp[3])
    image_estimate = conv2d(features,kernel_rotated,border_mode='full',
                            image_shape=featshp,filter_shape=kshp_rotated,
                            imshp_logical=featshp_logical[1:],kshp_logical=kshp[2:])

    if mask:
        image_error_temp = image - image_estimate
        image_error = T.zeros_like(image_error_temp)
        image_error = T.set_subtensor(image_error[:,:,(kshp[2]-1):(imshp[2]-kshp[2]+1),(kshp[3]-1):(imshp[3]-kshp[3]+1)],
                                 image_error_temp[:,:,(kshp[2]-1):(imshp[2]-kshp[2]+1),(kshp[3]-1):(imshp[3]-kshp[3]+1)])
    else:
        image_error = image - image_estimate

    return image_error, kernel, features

def helper_T_l2_cost_conv(image,features,kernel,imshp,kshp,featshp,stride=(1,1),mask=False):
    """
    imshp = num images, channels, szy, szx
    kshp = features, channels, szy, szx
    featshp = num images, features, szy, szx

    """


    if True:
        from myconv import MyCorr
        my_corr2d = MyCorr(strides=stride)
        image_estimate = my_corr2d(features,kernel)

    else:
        # Need to transpose first two dimensions of kernel, and reverse index kernel image dims (for correlation)
        kernel_rotated = T.transpose(kernel[:,:,::-1,::-1],axes=[1,0,2,3])
        featshp_logical = (featshp[0],featshp[1],featshp[2]*stride[0],featshp[3]*stride[1])
        kshp_rotated = (kshp[1], kshp[0], kshp[2], kshp[3])
        image_estimate = conv2d(features,kernel_rotated,border_mode='full',
                                image_shape=featshp,filter_shape=kshp_rotated,
                                imshp_logical=featshp_logical[1:],kshp_logical=kshp[2:])

    image_error = image - image_estimate

    return image_error

def fix_gpu_transfer():

    kshp=(10,2,10,10)
    featshp=(3,10,11,11)
    stride=8
    mask = False
    imshp = (featshp[0],kshp[1],featshp[2]*stride + kshp[2] - 1,featshp[3]*stride + kshp[3] - 1) # num images, channels, szy, szx

    from theano import tensor as T
    x = T.tensor4()
    a = T.tensor4()
    A = T.tensor4()

    image_error = helper_T_l2_cost_conv(x,a,A,imshp,kshp,featshp,stride=(stride,stride),mask=mask)
    cost = .5*T.sum(image_error **2)

    func = function([x,a,A],cost)

    import theano
    theano.printing.debugprint(func)

    x_in = np.random.randn(*imshp).astype(np.float32)
    a_in = np.random.randn(*featshp).astype(np.float32)
    A_in = np.random.randn(*kshp).astype(np.float32)

    from time import time as now
    repeats = 10
    t0 = now()
    for i in range(repeats):
        output = func(x_in,a_in,A_in)
    t = now() - t0
    print 'time / iter = %f' % (t/repeats)

    #| | |ConvOp{('imshp', (10, 11, 11)),('kshp', (10, 10)),('nkern', 2),('bsize', 3),('dx', 1),('dy', 1),('out_mode', 'full'),('unroll_batch', 3),('unroll_kern', 2),('unroll_patch', False),('imshp_logical', (10, 11, 11)),('kshp_logical', (10, 10)),('kshp_logical_top_aligned', True)} [@69066448] ''   8
    #| | |ConvOp{('imshp', (10, 11, 11)),('kshp', (10, 10)),('nkern', 2),('bsize', 3),('dx', 1),('dy', 1),('out_mode', 'full'),('unroll_batch', 3),('unroll_kern', 2),('unroll_patch', False),('imshp_logical', (10, 33, 33)),('kshp_logical', (10, 10)),('kshp_logical_top_aligned', True)} [@46739152] ''   8

def debug_convsparsenet(lam_sparse=.1,N=16,perc_var=100.,stride=1,kshp=(7,7),batchsize=4,imsz=(64,64)):

    from hdl.learners import SGD

    whitenpatches = 100
    if isinstance(kshp,int): kshp = (kshp,kshp)
    if isinstance(stride,int): stride = (stride,stride)
    imszr = imsz[0]
    imszc = imsz[1]

    #model = ConvWhitenInputModel(imshp=(10,1,32,32),convwhitenfiltershp=(7,7),perc_var=100.)
    model = ConvSparseSlowModel(imshp=(batchsize,1,imszr,imszc),convwhitenfiltershp=(7,7),N=N,kshp=kshp,stride=stride,
        perc_var=perc_var,lam_sparse=lam_sparse)

    l = SGD(model=model,datasource='vid075-chunks',display_every=50,save_every=10000,batchsize=model.imshp[0])

    print 'whitenpatches', whitenpatches
    print 'model.imshp', model.imshp
    print 'model.convwhitenfiltershp', model.convwhitenfiltershp

    databatch = l.get_databatch(whitenpatches)
    l.model.learn_whitening(databatch)

    l.model.setup()

    l.learn(iterations=10)
    l.model.center_basis_functions = False
    l.learn(iterations=10)

if __name__ == '__main__':

    #test_conv()
    #test_corr()
    #test_derivative(kshp=(2,1,3,3),featshp=(2,2,2,2),stride=1)
    #test_derivative_driver()
    #test_kernel_derivative(kshp=(2,2,3,3),featshp=(1,2,3,3),stride=1)
    #test_kernel_derivative(kshp=(2,2,3,3),featshp=(1,2,3,3),stride=3)
    #test_kernel_derivative_driver()
    #test_imageshape()
    #test_convsparsenet()
    #sweep_lambda()
    #test_convsparsenet(lam_sparse=1.0,stride=5)
    #test_convsparsenet_subspace(perc_var=99.,lam_sparse=1.,N=64,stride=(1,1))
    #test_sparsenet_subspace(perc_var=99.,lam_sparse=.1,N=256)
    #test_convsparsenet(lam_sparse=0.1,stride=8,kshp=16,N=64)
    #fix_gpu_transfer()
    #test_scipy_view_method()
    #fix_gpu_transfer()
    debug_convsparsenet(stride=8,batchsize=4,kshp=(16,16),imsz=(58,58))

