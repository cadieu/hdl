from time import time

import numpy as np

from scipy.misc import lena

from sklearn.feature_extraction.image import extract_patches_2d

from hdl.interface import SparseCoding

###############################################################################
# Load Lena image and extract patches

lena = lena() / 256.0
height, width = lena.shape

# Extract all clean patches from the left half of the image
print 'Extracting clean patches...'
t0 = time()
patch_size = (6, 6)
data = extract_patches_2d(lena, patch_size)
data = data.reshape(data.shape[0], -1)
data -= np.mean(data, axis=0)
data /= np.std(data, axis=0)
print 'done in %.2fs.' % (time() - t0)

###############################################################################
# Learn the dictionary from clean patches

print 'Learning the dictionary SparseCoding... '
t0 = time()
np.random.seed(42)
dico = SparseCoding(n_atoms=36, alpha=.1, max_iter=200)
V = dico.fit(data).components_
print 'done in %.2fs.' % (time() - t0)

print 'Transform the data...'
t0 = time()
coef = dico.transform(data[:10,:])
print 'done in %.2fs.' % (time() - t0)

np.testing.assert_almost_equal(coef[:10,:].sum(),3.23775956895)
np.testing.assert_almost_equal(V[:,:10].sum(),39.9447109218)
