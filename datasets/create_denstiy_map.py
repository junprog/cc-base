# Creating Density Map Code using only numpy
# reference to https://github.com/TencentYoutuResearch/CrowdCounting-SASNet/blob/main/prepare_dataset.py

import numpy as np

def gaussian_filter(kernnel_size=(3, 3), sigma=0.5):
    m, n = [(ss - 1.) / 2. for ss in kernnel_size]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()

    if sumh != 0:
        h /= sumh

    return h

def create_density_map(im_shape: tuple, gt: np.ndarray, kernel_size=(15, 15), sigma=5) -> np.ndarray:
    """create density map
    Args:
        im_shape:       tuple
                        image shape
        gt:             numpy.array
                        head point location
        kernel_size:    default=(15, 15)
                        gausian kernel size
        sigma:          default=5
                        gaussian parameter (sigma)
    
    Return:
        density:        numpy.array
                        density map
    """
    if gt.ndim != 2:
        raise TypeError('gt dim is wrong.')

    density = np.zeros(im_shape)
    h, w = im_shape

    if len(gt) == 0:
        return density

    for point in gt:

        