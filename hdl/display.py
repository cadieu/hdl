import os
import numpy as np
from scipy.misc import toimage

from config import state_dir
from matplotlib import pyplot as plt


def display_patches(patches, psz, fig_num=None, normalize=True):
    # plot the vectors in A
    NN = patches.shape[1]
    buf = 1
    sz = int(np.ceil(np.sqrt(NN)))
    hval = np.max(np.abs(patches))
    array = -np.ones(((psz + buf) * sz + buf, (psz + buf) * sz + buf))
    pind = 0
    for r in range(sz):
        for c in range(sz):
            if pind >= NN:
                continue
            if normalize:
                hval = np.max(np.abs(patches[:, pind]))
                if hval == 0.: hval = 1
            patchesvalues = patches[:, pind].reshape(psz, psz) / hval
            array[buf + (psz + buf) * c:buf + (psz + buf) * c + psz,
            buf + (psz + buf) * r:buf + (psz + buf) * r + psz] = patchesvalues
            pind += 1
    hval = 1.
    if fig_num is None:
        plt.figure()
    else:
        plt.figure(fig_num)
    plt.clf()
    plt.imshow(array, vmin=-hval, vmax=hval, interpolation='nearest', cmap=plt.cm.gray)
    plt.colorbar()

    return array


def display_binoc_color_patches(binoc_patches, psz, fig_num=None, normalize=True):
    NN = binoc_patches.shape[1]
    patches = np.zeros((psz, psz, 3, 2 * NN))
    count = 0
    for nn in range(NN):
        patches[..., count] = binoc_patches[..., nn].reshape(6, psz, psz)[:3, ...].T
        patches[..., count + 1] = binoc_patches[..., nn].reshape(6, psz, psz)[3:, ...].T
        count += 2
    patches = patches.reshape((psz * psz * 3, 2 * NN))

    return display_color_patches(patches=patches, psz=psz, fig_num=fig_num, normalize=normalize)


def display_color_patches(patches, psz, fig_num=None, normalize=True):
    # plot the vectors in A
    NN = patches.shape[1]
    buf = 1
    sz = int(np.sqrt(NN))
    if sz % 2:
        sz += 1
    array = np.zeros(((psz + buf) * sz + buf, (psz + buf) * sz + buf, 3))
    pind = 0
    for r in range(sz):
        for c in range(sz):
            if pind >= NN:
                continue
            if normalize:
                hval = np.max(np.abs(patches[:, pind]))
                if hval == 0.: hval = 1.
                patchesvalues = 127.5 * patches[:, pind].reshape(psz, psz, 3) / hval + 127.5
            else:
                patchesvalues = patches[:, pind].reshape(psz, psz, 3)
            array[buf + (psz + buf) * c:buf + (psz + buf) * c + psz, buf + (psz + buf) * r:buf + (psz + buf) * r + psz,
            :] = patchesvalues
            pind += 1
    array = np.clip(array, 0., 255.).astype(np.uint8)

    if fig_num is None:
        plt.figure()
    else:
        plt.figure(fig_num)
    plt.clf()
    plt.imshow(array, interpolation='nearest')

    return array


def display_final(m, save_string='final'):
    savepath = os.path.join(state_dir, m.model_name + '_' + m.tstring)
    if not os.path.exists(savepath): os.makedirs(savepath)

    repr_string = m.__repr__()
    model_details_fname = os.path.join(savepath, 'model_details_final.txt')
    with open(model_details_fname, 'w') as fh:
        fh.write(repr_string)

    max_factors = m.D

    d = m.display_whitening(save_string=save_string, max_factors=max_factors, zerophasewhiten=False)
    if not d is None:
        fname = os.path.join(savepath, 'whitenmatrix_hires_' + save_string + '.png')
        if d['whitenmatrix'].ndim == 2:
            toimage(np.floor(.5 * (d['whitenmatrix'] + 1) * 255)).save(fname)
        else:
            toimage(d['whitenmatrix']).save(fname)

    d = m.display_whitening(save_string=save_string, max_factors=max_factors)
    if not d is None:
        fname = os.path.join(savepath, 'whitenmatrix_hires_zerophase_' + save_string + '.png')
        if d['whitenmatrix'].ndim == 2:
            toimage(np.floor(.5 * (d['whitenmatrix'] + 1) * 255)).save(fname)
        else:
            toimage(d['whitenmatrix']).save(fname)

    if hasattr(m, 'NN'):
        max_factors = m.NN

        d = m.display(save_string=save_string, max_factors=max_factors, zerophasewhiten=False)
        fname = os.path.join(savepath, 'A_hires_' + save_string + '.png')
        if d['A'].ndim == 2:
            toimage(np.floor(.5 * (d['A'] + 1) * 255)).save(fname)
        else:
            toimage(d['A']).save(fname)

        d = m.display(save_string=save_string, max_factors=max_factors)
        fname = os.path.join(savepath, 'A_hires_zerophase_' + save_string + '.png')
        if d['A'].ndim == 2:
            toimage(np.floor(.5 * (d['A'] + 1) * 255)).save(fname)
        else:
            toimage(d['A']).save(fname)


def display_multilayer(learner, max_N=8, fix_norm=2., label=''):

    if not label == '':
        label = '_' + label

    from scipy.optimize import fmin_l_bfgs_b

    model_sequence = learner.model_sequence
    output_functions = [param['output_function'] for param in learner.layer_params]

    num_layers = len(learner.model_sequence)

    D = model_sequence[0].D
    # for each layer of the network:
    for layer_num in range(num_layers):
        if layer_num == 0:
            continue

        # create function and derivatives
        func, grad, original_args = get_theano_multilayer_func_grad(model_sequence,
                                                                    output_functions,
                                                                    fix_norm,
                                                                    layer_num,
                                                                    debug=False)

        # optimize for each layer beyond the first.
        A = model_sequence[layer_num].A.get_value()
        N = min(max_N, A.shape[1])
        original_args_len = len(original_args)
        optimal_patches = []
        for n in range(N):

            layer_proj = A[:, n].ravel()
            layer_proj = layer_proj.reshape(layer_proj.size, 1)
            original_args.append(layer_proj)
            print [item.shape for item in original_args]

            #x0 = model_sequence[0].inputmean.copy().ravel() + 10. * np.random.randn(D)
            x0 = np.random.randn(model_sequence[0].A.get_value().shape[0])
            if fix_norm:
                norm = fix_norm
                x0 = norm * x0 / (.01 + np.sqrt(np.sum(x0 ** 2)))
            print x0.shape
            f0 = func(x0, *original_args)
            print 'f0', f0

            fmin_output = fmin_l_bfgs_b(func=func, x0=x0, fprime=grad, args=original_args, disp=1, maxfun=100)
            x = fmin_output[0]
            print 'optimal x:', x.min(), x.max(), x.mean(), np.median(x)
            x = np.dot(model_sequence[0].dewhitenmatrix, x)# + model_sequence[0].inputmean
            print 'optimal x:', x.min(), x.max(), x.mean(), np.median(x)
            optimal_patches.append(x)

            original_args = original_args[:original_args_len]
        optimal_patches = np.array(optimal_patches).T

        array = display_patches(optimal_patches, psz=int(np.sqrt(model_sequence[0].D)))
        save_path = os.path.join(state_dir, model_sequence[layer_num].model_name + '_' + model_sequence[layer_num].tstring)
        fname = os.path.join(save_path, 'Optimal_Stimuli_layer_%d%s.png' % (layer_num, label))
        toimage(np.floor(.5 * (array + 1) * 255)).save(fname)


def get_theano_multilayer_func_grad(model_sequence, output_functions, fix_norm, layer_num, debug=False):
    small_value = .01

    if debug:
        import theano
        theano.config.compute_test_value = 'warn'

    from theano import tensor as T
    from theano import function, grad

    x = T.vector('input_x')
    if debug:
        x.tag.test_value = np.random.randn(48 ** 2)

    y = x.reshape((x.size, 1), ndim=2)

    output = None
    arg_expression = [x]
    arg_values = []
    # construct a theano expression for the feedforward network
    for i in range(layer_num + 1):
        print i, layer_num

        m = model_sequence[i]

        if i == layer_num:
            print 'outputlayer'
            print arg_expression

            # whitening transform
            inputmean = T.matrix(name='layer_%d_inputmean' % i)
            wh = T.matrix(name='layer_%d_wh' % i)
            A = T.matrix(name='layer_%d_A' % i)

            if debug:
                inputmean.tag.test_value = np.random.randn(2048, 1)
                wh.tag.test_value = np.random.randn(1402, 2048)
                A.tag.test_value = np.random.randn(1402, 1)

            y = T.sum(A * T.dot(wh, y - inputmean))
            arg_expression.extend([inputmean, wh, A])

            # we will append the appropriate vector for each minimzation:
            arg_values.extend([np.double(m.inputmean), np.double(m.whitenmatrix)])

        else:
            print 'firstlayer'
            print arg_expression
            # whitening transform
            inputmean = T.matrix(name='layer_%d_inputmean' % i)
            wh = T.matrix(name='layer_%d_wh' % i)
            A = T.matrix(name='layer_%d_A' % i)

            if debug:
                inputmean.tag.test_value = np.random.randn(48 ** 2, 1)
                wh.tag.test_value = np.random.randn(590, 2304)
                A.tag.test_value = np.random.randn(590, 1024)

            if output_functions[i] == 'proj_rect_sat':

                #nomean = y - inputmean
                nomean = y
                if fix_norm:
                    print 'fix_norm'
                    norm = fix_norm
                    nomean = norm * nomean / (small_value + T.sqrt(T.sum(nomean ** 2)))

                #projection = T.dot(A.T, T.dot(wh, nomean))
                projection = T.dot(A.T, nomean)
                #arg_expression.extend([inputmean, wh, A])
                arg_expression.extend([A])

                #arg_values.extend([np.double(m.inputmean), np.double(m.whitenmatrix), np.double(m.A.get_value())])
                arg_values.extend([np.double(m.A.get_value())])

                # half-wave
                N = projection.shape[0]
                u = T.zeros((N * 2, 1), dtype=projection.dtype)
                u_pos = T.switch(T.gt(projection, 0.), projection, 0.)
                u_neg = T.switch(T.lt(projection, 0.), T.abs_(projection), 0.)
                u = T.set_subtensor(u[::2, :], u_pos)
                u = T.set_subtensor(u[1::2, :], u_neg)

                # saturation
                rect_value = 8.0
                y = T.switch(T.gt(u, rect_value), rect_value, u)
            else:
                assert NotImplemented, output_functions[i]

    print arg_expression
    output = -y

    dx = grad(output, x)
    func = function(arg_expression, outputs=output)
    grad = function(arg_expression, outputs=dx)

    return func, grad, arg_values


