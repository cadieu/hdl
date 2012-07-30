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

def T_l2_cost_conv(x,a,A,imshp,kshp,mask=True):
    """
    xsz*ysz*nchannels, nimages = x.shape
    xsz*ysz*nfeat, nimages = a.shape
    xsz*ysz*nchannels, nfeat = A.shape
    """

    #imshp = num images, channels, szy, szx
    #kshp = features, channels, szy, szx
    #featshp = num images, features, szy, szx

    featshp = (imshp[0],kshp[0],imshp[2] - kshp[2] + 1,imshp[3] - kshp[3] + 1) # num images, features, szy, szx

    image = T.reshape(T.transpose(x),imshp)
    kernel = T.reshape(T.transpose(A),kshp)
    features = T.reshape(T.transpose(a),featshp)

    # Need to transpose first two dimensions of kernel, and reverse index kernel image dims (for correlation)
    kernel_rotated = T.transpose(kernel[:,:,::-1,::-1],axes=[1,0,2,3])

    image_estimate = conv2d(features,kernel_rotated,border_mode='full')

    if mask:
        image_error_temp = image - image_estimate
        image_error = T.zeros_like(image_error_temp)
        image_error = T.set_subtensor(image_error[:,:,(kshp[2]-1):(imshp[2]-kshp[2]+1),(kshp[3]-1):(imshp[3]-kshp[3]+1)],
                                 image_error_temp[:,:,(kshp[2]-1):(imshp[2]-kshp[2]+1),(kshp[3]-1):(imshp[3]-kshp[3]+1)])
    else:
        image_error = image - image_estimate

    return .5*T.sum(image_error **2)

def T_gl2_cost_conv(x,a,A,imshp,kshp,mask=True):
    _l2_cost_conv = T_l2_cost_conv(x,a,A,imshp,kshp,mask=mask)
    return T.grad(_l2_cost_conv,a)

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
