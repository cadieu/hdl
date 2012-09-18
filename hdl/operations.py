import numpy as np


def convolve4d_view(image, kernel, mode='valid', stride=(1, 1)):
    from skimage.util.shape import view_as_windows

    imshp = image.shape
    kshp = kernel.shape

    offset = None
    if mode == 'valid':
        featshp = (imshp[0], kshp[0], (imshp[2] - kshp[2]) / stride[0] + 1,
                   (imshp[3] - kshp[3]) / stride[1] + 1) # num images, features, szy, szx
    elif mode == 'same':
        assert stride == (1, 1)
        featshp = (imshp[0], kshp[0], imshp[2], imshp[3]) # num images, features, szy, szx
        offset = (kshp[2] / 2, kshp[3] / 2)
    #elif mode == 'full':
    #    featshp = (imshp[0],kshp[0],imshp[2] + kshp[2] - 1,imshp[3] + kshp[3] - 1) # num images, features, szy, szx
    else:
        raise NotImplemented, 'Unkonwn mode %s' % mode

    kernel_flipped = kernel[:, :, ::-1, ::-1]

    output = np.zeros(featshp, dtype=image.dtype)
    this_image = None
    for im_i in range(imshp[0]):
        if mode == 'valid':
            this_image = image[im_i, ...]
        elif mode == 'same':
            if this_image is None:
                this_image_shp = (imshp[1], imshp[2] + kshp[2] - 1, imshp[3] + kshp[3] - 1)
                this_image = np.zeros(this_image_shp, dtype=image.dtype)
            this_image[:, offset[0]:(offset[0] + imshp[2]), offset[1]:(offset[1] + imshp[3])] = image[im_i, ...]
        else:
            raise NotImplemented
        imager = view_as_windows(this_image, (kshp[1], kshp[2], kshp[3]))[0, ::stride[0], ::stride[1], ...]
        # imager.shape = (featszr, featszc, channels, ksz[2], ksz[3])
        feat = np.tensordot(kernel_flipped, imager, axes=((1, 2, 3), (2, 3, 4)))

        output[im_i, ...] = feat

    return output
