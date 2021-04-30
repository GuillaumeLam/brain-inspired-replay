import random
import numpy as np

from skimage.filters import gaussian


def distortion(args):

    if args.distortion=='gaussian_blur':
        return lambda img: gaussian_blur(img, args.severity, args.experiment=='CIFAR100')
    elif args.distortion=='shot_noise':
        return lambda img: shot_noise(img, args.severity, args.experiment=='CIFAR100')
    elif args.distortion=='hyper':
        return lambda img: hyper_distortions(img, args.experiment=='CIFAR100')
    elif args.distortion=='med':
        return lambda img: medium_distortions(img, args.experiment=='CIFAR100')
    elif args.distortion=='fine':
        return lambda img: fine_distortions(img, args.experiment=='CIFAR100')


def gaussian_blur(img, severity=2, cifar=False):

    c = [1, 2, 3, 4, 6][severity - 1]
    x = np.array(img).astype(np.float32)
    if cifar:
        minim = np.min(x)
        maxi = np.max(x)
        if maxi == minim:
            return x - x
        x = (x - minim) / (maxi-minim)
    x = gaussian(x, sigma=c, multichannel=False)
    x = np.clip(x, 0, 1)
    x = ((x*(maxi-minim)) + minim) if cifar else x

    return x


def shot_noise(img, severity=3, cifar=False):
    # c = [4, 3, 2, 1, 1][2]
    c = [4, 3, 2, 1, 1][severity-1]
    x = np.array(img).astype(np.float32)
    if cifar:
        minim = np.min(x)
        maxi = np.max(x)
        if maxi == minim:
            return x - x
        x = (x - minim) / (maxi-minim)
    x *= (2 ** c - 1)
    x = x.round()
    x /= (2 ** c - 1)
    x = ((x*(maxi-minim)) + minim) if cifar else x

    return x


def hyper_distortions(img, cifar=False):
    x = np.array(img).astype(np.float32)

    x = shot_noise(x, 3, cifar)
    x = gaussian_blur(x, 3, cifar)

    return x


def medium_distortions(img, cifar=False):
    x = np.array(img).astype(np.float32)

    x = shot_noise(x, 2, cifar)
    x = gaussian_blur(x, 2, cifar)

    return x


def fine_distortions(img, cifar=False):
    x = np.array(img).astype(np.float32)

    x = shot_noise(x, 1, cifar)
    x = gaussian_blur(x, 1, cifar)

    return x
