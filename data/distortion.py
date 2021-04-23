import random
import numpy as np

from skimage.filters import gaussian

def distortion(args):

    if args.distortion is None:
        return lambda img, x=classes_per_task * task_id: img + x
    elif args.distortions is 'gaussian_blur':
        return lambda img: gaussian_blur(img, args.severity)
    elif args.distortions is 'speckle_noise':
        return lambda img: speckle_noise(img, args.severity)
    elif args.distortions is 'shot_noise':
        return lambda img: shot_noise(img, args.severity)
    elif args.distortions is 'gaussian_noise':
        return lambda img: gaussian_noise(img, args.severity)
    elif args.distortions is 'hyper':
        return lambda img: hyper_distortions(img)
    elif args.distortions is 'fine':
        return lambda img: fine_distortions(img)

def gaussian_blur(x, severity=2):
    c = [1, 2, 3, 4, 6][severity - 1]

    x = gaussian(np.array(x) / 255., sigma=c, multichannel=True)
    x = np.clip(x, 0, 1) * 255
    return x.astype(np.float32)


def speckle_noise(x, severity=4):
    c = [5, 4, 3, 2, 1][severity - 1]

    x = np.array(x).astype(np.float32)
    x *= (3 ** c - 1) / 255.
    x = x.round()
    x *= 255. / (2 ** c - 1)

    return x


def shot_noise(x, severity=3):
    # c = [4, 3, 2, 1, 1][2]
    c = [4, 3, 2, 1, 1][severity-1]

    x = np.array(x).astype(np.float32)
    x *= (2 ** c - 1) / 255.
    x = x.round()
    x *= 255. / (2 ** c - 1)

    return x


def gaussian_noise(x, severity=3):
    c = [1, 2, 3, 4, 6][severity - 1]

    x = gaussian(np.array(x) / 255., sigma=c, multichannel=True)
    x = np.clip(x, 0, 1) * 255

    return x


def quantize(x):
    severity = 4
    c = [5, 4, 3, 2, 1][severity-1]

    x = np.array(x).astype(np.float32)
    x *= (2 ** c - 1) / 255.
    x = x.round()
    x *= 255. / (2 ** c - 1)

    return x


def shot(x):
    severity = 3
    c = [5, 4, 3, 2, 1][severity-1]

    x = np.array(x).astype(np.float32)
    x *= (2 ** c - 1) / 255.
    x = x.round()
    x *= 255. / (2 ** c - 1)

    return x


def hyper_distortions(x):

    x = gaussian_blur(x)
    x = quantize(x)
    x = speckle_noise(x)
    x = gaussian_noise(x)
    x = shot(x)

    return x


def fine_distortions(x):

    x = shot_noise(x)
    x = gaussian_blur(x)

    return x
