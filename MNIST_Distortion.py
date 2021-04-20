import numpy as np
import skimage as sk
from skimage import io
from skimage.filters import gaussian
from io import BytesIO
import torch
import torchvision
from torchvision import transforms, models, datasets
from torch.utils.data import Dataset
import random
from importlib import reload
import ctypes
from skimage.color import rgb2gray


# wand, radius, sigma, angle
def gaussian_blur(x):
    strength=2
    c = [1, 2, 3, 4, 6][strength - 1]

    x = gaussian(np.array(x) / 255., sigma=c, multichannel=True)
    x = np.clip(x, 0, 1) * 255
    return x.astype(np.float32)


def quantize(x):
    q = 4
    c = [5, 4, 3, 2, 1][q-1]

    x = np.array(x).astype(np.float32)
    x *= (2 ** c - 1) / 255.
    x = x.round()
    x *= 255. / (2 ** c - 1)

    return x


def speckle_noise_2(x):
    s = 4
    c = [5, 4, 3, 2, 1][s - 1]

    x = np.array(x).astype(np.float32)
    x *= (3 ** c - 1) / 255.
    x = x.round()
    x *= 255. / (2 ** c - 1)

    return x


def gaussian_noise_2(x):
    g_b = 3
    c = [1, 2, 3, 4, 6][g_b - 1]

    x = gaussian(np.array(x) / 255., sigma=c, multichannel=True)
    x = np.clip(x, 0, 1) * 255
#     return x.astype(np.float32)

    return x


def shot(x):
    c = [5, 4, 3, 2, 1][2]

    x = np.array(x).astype(np.float32)
    x *= (2 ** c - 1) / 255.
    x = x.round()
    x *= 255. / (2 ** c - 1)

    return x


def shot_noise_2(x):
    c = [4, 3, 2, 1, 1][2]

    x = np.array(x).astype(np.float32)
    x *= (2 ** c - 1) / 255.
    x = x.round()
    x *= 255. / (2 ** c - 1)

    return x


def brightness(x):
    b = 5
    c = [.1, .2, .3, .4, .5][b - 1]

    x = np.array(x) / 255.
    x = sk.color.gray2rgb(x)
    x = sk.color.rgb2hsv(x)
    x[2] = np.clip(x[2] + c, 0, 1)
#     x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    x = sk.color.hsv2rgb(x)
    x = sk.color.rgb2gray(x)
    x = np.clip(x, 0, 1) * 255
    
    return x.astype(np.float32)


def inverse(x):
    x = np.array(x).astype(np.float32)
    a = 255. - x
    x = a
    
    return x

###############################################################################################################

def gaussian_noise(x):
    severity=5 
    c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]

    x = np.array(x) / 255.
    x = np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255
#     x = np.clip(x + np.torch.tensor(random_noise(img, mode='gaussian', mean=0, var=0.05, clip=True)) * 255
    return x.astype(np.float32)


def shot_noise(x):
    severity=5
    c = [60, 25, 12, 5, 3][severity - 1]

    x = np.array(x) / 255.
    x = np.clip(np.random.poisson(x * c) / float(c), 0, 1) * 255
    return x.astype(np.float32)


def speckle_noise(x):
    severity=5
    c = [.15, .2, 0.35, 0.45, 0.6][severity - 1]
    
    x = np.array(x) / 255.
    x = np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255
    
    return x.astype(np.float32)


def spatter(x):
    increase = 2
    c = [(0.65, 0.3, 4, 0.69, 0.6, 0),
         (0.65, 0.3, 3, 0.68, 0.6, 0),
         (0.65, 0.3, 2, 0.68, 0.5, 0),
         (0.65, 0.3, 1, 0.65, 1.5, 1),
         (0.67, 0.4, 1, 0.65, 1.5, 1)][increase - 1]

    x = np.array(x, dtype=np.float32) / 255.

    spots = np.random.normal(size=x.shape, loc=c[0], scale=c[1])

    spots = gaussian(spots, sigma=c[2])
    spots[spots < c[3]] = 0
    
    m = np.where(spots > c[3], 1, 0)
    m = gaussian(m.astype(np.float32), sigma=c[4])
    m[m < 0.8] = 0

    # mud spatter
    color = 63 / 255. * np.ones_like(x) * m
    x *= (1 - m)

    return np.clip(x + color, 0, 1) * 255


def contrast(x):
    increase = 4 
    c = [0.4, .3, .2, .1, .05][increase - 1]

    x = np.array(x) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    x = np.clip((x - means) * c + means, 0, 1) * 255
    return x.astype(np.float32)


def brightness(x):
    increase = 5
    c = [.1, .2, .3, .4, .5][increase - 1]

    x = np.array(x) / 255.
    x = sk.color.gray2rgb(x)
    x = sk.color.rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    x = sk.color.hsv2rgb(x)
    x = sk.color.rgb2gray(x)

    x = np.clip(x, 0, 1) * 255
    return x.astype(np.float32)


def saturate(x):
    increase = 5
    c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][increase - 1]

    x = np.array(x) / 255.
    x = sk.color.gray2rgb(x)
    x = sk.color.rgb2hsv(x)
    x = np.clip(x * c[0] + c[1], 0, 1)
    x = sk.color.hsv2rgb(x)
    x = sk.color.rgb2gray(x)

    x = np.clip(x, 0, 1) * 255
    return x.astype(np.float32)


def inverse(x):
    x = np.array(x).astype(np.float32)
    x = 255. - x
    return x


def stripe(x):
    x = np.array(x).astype(np.float32)
    x[:,:7] = 255. - x[:,:7]
    x[:,21:] = 255. - x[:,21:]
    return x

###############################################################################################################

def distortions(x):

    g = [1, 2, 3, 4, 6][1]
    x_g = gaussian(np.array(x) / 255., sigma=g, multichannel=True)
    x_g = np.clip(x_g, 0, 1) * 255

    q = [5, 4, 3, 2, 1][3]
    x_q = np.array(x_g).astype(np.float32)
    x_q *= (2 ** q - 1) / 255.
    x_q = x_q.round()
    x_q *= 255. / (2 ** q - 1)

    i = [5, 4, 3, 2, 1][3]
    x_i = np.array(x_q).astype(np.float32)
    x_i *= (3 ** i - 1) / 255.
    x_i = x_i.round()
    x_i *= 255. / (2 ** i - 1)

    g_c = [1, 2, 3, 4, 6][2]
    x_g_c = gaussian(np.array(x_i) / 255., sigma=g_c, multichannel=True)
    x_g_c = np.clip(x_g_c, 0, 1) * 255

    i_2 = [5, 4, 3, 2, 1][2]
    x_i_2 = np.array(x_g_c).astype(np.float32)
    x_i_2 *= (2 ** i_2 - 1) / 255.
    x_i_2 = x_i_2.round()
    x_i_2 *= 255. / (2 ** i_2 - 1)

    return x_i_2

###############################################################################################################

def hyper_distortions(x):

    x = gaussian_blur(x)
    x = quantize(x)
    x = speckle_noise_2(x)
    x = gaussian_noise_2(x)
    x = shot(x)
             
    return x


def fine_distortions(x):

    x = shot_noise_2(x)
    x = gaussian_blur(x)     
    
    return x


