import random
import numpy as np

from skimage.filters import gaussian

def distortion(args, severity=5, method='speckle_noise'):
    # severity 0-10
    #

    if method is 'speckle_noise':
        return lambda img: speckle_noise(img)
    # elif method is 'random':
    #     return
    #
    #
    #
    # return lambda img: custom_transform(img)


def speckle_noise(x):
    severity=5
    c = [.15, .2, 0.35, 0.45, 0.6][severity - 1]

    x = np.array(x) / 255.
    x = np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255
    return x.astype(np.float32)


def gaussian_blur(x):
    severity=2
    c = [1, 2, 3, 4, 6][severity - 1]

    x = gaussian(np.array(x) / 255., sigma=c, multichannel=True)
    x = np.clip(x, 0, 1) * 255
    return x.astype(np.float32)


def spatter(x):
    severity=4
    c = [(0.65, 0.3, 4, 0.69, 0.6, 0),
         (0.65, 0.3, 3, 0.68, 0.6, 0),
         (0.65, 0.3, 2, 0.68, 0.5, 0),
         (0.65, 0.3, 1, 0.65, 1.5, 1),
         (0.67, 0.4, 1, 0.65, 1.5, 1)][severity - 1]

    x = np.array(x, dtype=np.float32) / 255.

    liquid_layer = np.random.normal(size=x.shape, loc=c[0], scale=c[1])

    liquid_layer = gaussian(liquid_layer, sigma=c[2])
    liquid_layer[liquid_layer < c[3]] = 0

    m = np.where(liquid_layer > c[3], 1, 0)
    m = gaussian(m.astype(np.float32), sigma=c[4])
    m[m < 0.8] = 0

    # mud spatter
    color = 63 / 255. * np.ones_like(x) * m
    x *= (1 - m)

    return np.clip(x + color, 0, 1) * 255


def inverse(x):
    x = np.array(x).astype(np.float32)
    return 255. - x


def stripe(x):
    x = np.array(x).astype(np.float32)
    x[:,:7] = 255. - x[:,:7]
    x[:,21:] = 255. - x[:,21:]
    return x
