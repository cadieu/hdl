from time import time

import pylab as pl
import numpy as np

from scipy.misc import lena

from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d

from hdl.interface import SparseCoding

###############################################################################
# Load Lena image and extract patches

lena = lena() / 256.0

# downsample for higher speed
lena = lena[::2, ::2] + lena[1::2, ::2] + lena[::2, 1::2] + lena[1::2, 1::2]
lena /= 4.0
height, width = lena.shape

# Distort the right half of the image
print 'Distorting image...'
distorted = lena.copy()
distorted[:, height / 2:] += 0.075 * np.random.randn(width, height / 2)

# Extract all clean patches from the left half of the image
print 'Extracting clean patches...'
t0 = time()
patch_size = (7, 7)
data = extract_patches_2d(distorted[:, :height / 2], patch_size)
data = data.reshape(data.shape[0], -1)
data -= np.mean(data, axis=0)
data /= np.std(data, axis=0)
print 'done in %.2fs.' % (time() - t0)

###############################################################################
# Learn the dictionary from clean patches (new way)

print 'Learning the dictionary SparseCoding... '
t0 = time()
dico = SparseCoding(n_atoms=100, alpha=1., max_iter=10000)
V = dico.fit(data).components_
dt = time() - t0
print 'done in %.2fs.' % dt

pl.figure(figsize=(4.2, 4))
for i, comp in enumerate(V[:100]):
    pl.subplot(10, 10, i + 1)
    pl.imshow(comp.reshape(patch_size), cmap=pl.cm.gray_r,
        interpolation='nearest')
    pl.xticks(())
    pl.yticks(())
pl.suptitle('Dictionary learned from Lena patches\n' +
            'Train time %.1fs on %d patches' % (dt, len(data)),
    fontsize=16)
pl.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

pl.savefig('sparsecoding_dictionary_learning.png')

###############################################################################
# Learn the dictionary from clean patches (old way)

print 'Learning the dictionary MiniBatchDictionaryLearning... '
t0 = time()
dico = MiniBatchDictionaryLearning(n_atoms=100, alpha=1, n_iter=500)
V = dico.fit(data).components_
dt = time() - t0
print 'done in %.2fs.' % dt

pl.figure(figsize=(4.2, 4))
for i, comp in enumerate(V[:100]):
    pl.subplot(10, 10, i + 1)
    pl.imshow(comp.reshape(patch_size), cmap=pl.cm.gray_r,
        interpolation='nearest')
    pl.xticks(())
    pl.yticks(())
pl.suptitle('Dictionary learned from Lena patches\n' +
            'Train time %.1fs on %d patches' % (dt, len(data)),
    fontsize=16)
pl.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

pl.savefig('sklearn_dictionary_learning.png')
