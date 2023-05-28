import numpy as np
from scipy import ndimage
import imageio as io
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import os

CONV_GAUSS_CREATOR = [1, 1]
MINIMUM_RESOLUTION = 16
GRAYSCALE = 1
RGB = 2
BIN_SIZE = 256



def reduce(im, blur_filter):
    """
    Reduces an image by a factor of 2 using the blur filter
    :param im: Original image
    :param blur_filter: Blur filter
    :return: the downsampled image
    """
    blurred_image = ndimage.filters.convolve(
        ndimage.filters.convolve(im, blur_filter, mode='constant', cval=0.0),
        blur_filter.T, mode='constant', cval=0.0)
    return blurred_image[::2, ::2]

def expand(im, blur_filter):
    """
    Expand an image by a factor of 2 using the blur filter
    :param im: Original image
    :param blur_filter: Blur filter
    :return: the expanded image
    """
    new_im = np.zeros((im.shape[0] * 2, im.shape[1] * 2))
    new_im[::2, ::2] = im
    blurred_image = ndimage.filters.convolve(
        ndimage.filters.convolve(new_im, blur_filter, mode='constant', cval=0.0),
        blur_filter.T, mode='constant', cval=0.0)
    return blurred_image


def build_gaussian_filter(filter_size):
    """
    Helper function that get filter_size and returns a gaussisan filter of that size
    :param filter_size: the wanted size for the gaussian filter
    :return: the gaussian filter as a np array with shape of (1,filter_size)
    """
    new_filter = CONV_GAUSS_CREATOR
    for i in range(filter_size - 2):
        new_filter = np.convolve(new_filter, CONV_GAUSS_CREATOR)
    return (new_filter / np.sum(new_filter)).reshape(1, filter_size)


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    Builds a gaussian pyramid for a given image
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter
            (an odd scalar that represents a squared filter)
            to be used in constructing the pyramid filter
    :return: pyr, filter_vec. Where pyr is the resulting pyramid as a
            standard python array with maximum length of max_levels,
            where each element of the array is a grayscale image.
            and filter_vec is a row vector of shape (1, filter_size)
            used for the pyramid construction.
    """
    pyr = list()
    pyr.append(im)
    blur_filter = build_gaussian_filter(filter_size)
    tmp_image = im
    for i in range(max_levels - 1):
        tmp_image = reduce(tmp_image, blur_filter)
        pyr.append(tmp_image)
        if tmp_image.shape[0] <= MINIMUM_RESOLUTION or tmp_image.shape[1] <= MINIMUM_RESOLUTION:
            break
    return pyr, blur_filter


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    Builds a laplacian pyramid for a given image
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter
            (an odd scalar that represents a squared filter)
            to be used in constructing the pyramid filter
    :return: pyr, filter_vec. Where pyr is the resulting pyramid as a
            standard python array with maximum length of max_levels,
            where each element of the array is a grayscale image.
            and filter_vec is a row vector of shape (1, filter_size)
            used for the pyramid construction.
    """
    gauss_pyr, blur_filter = build_gaussian_pyramid(im, max_levels, filter_size)
    lap_pyr = list()
    for i in range(len(gauss_pyr) - 1):
        lap_pyr.append(gauss_pyr[i] - expand(gauss_pyr[i + 1], 2 * blur_filter))
    lap_pyr.append(gauss_pyr[len(gauss_pyr) - 1])
    return lap_pyr, blur_filter


def laplacian_to_image(lpyr, filter_vec, coeff):
    """

    :param lpyr: Laplacian pyramid
    :param filter_vec: Filter vector
    :param coeff: A python list in the same length as the number of levels in
            the pyramid lpyr.
    :return: Reconstructed image
    """
    lpyr_size = len(lpyr)
    recon_img = lpyr[lpyr_size - 1] * coeff[lpyr_size - 1]
    for i in range(lpyr_size - 2, -1, -1):
        recon_img = lpyr[i] * coeff[i] + expand(recon_img, 2 * filter_vec)
    return recon_img


def im_normlizer(im, pad_num):
    """
    Helper function for the render the function normlize the image and pad it with zeroes
    :param im:The Image we normlize as a 2D matrix
    :param pad_num: The new length of the columns (Assumes Pas_num greater then the column size)
    :return: The normlized image sized according to pad
    """
    im_min = np.min(im)
    im_max = np.max(im)
    if im_min == im_max:
        return np.zeros(im.shape)
    norm_im = (im - im_min) / (im_max - im_min)
    return np.pad(norm_im, ((0, pad_num - im.shape[0]), (0, 0)), mode='constant', constant_values=0)


def render_pyramid(pyr, levels):
    """
    Render the pyramids as one large image with 'levels' smaller images
        from the pyramid
    :param pyr: The pyramid, either Gaussian or Laplacian
    :param levels: the number of levels to present
    :return: res a single black image in which the pyramid levels of the
            given pyramid pyr are stacked horizontally.
    """
    pyr_norm = list()
    pad_num = pyr[0].shape[0]
    for i in range(min(levels, len(pyr))):
        pyr_norm.append(im_normlizer(pyr[i], pad_num))
    return np.hstack(pyr_norm)


def display_pyramid(pyr, levels):
    """
    display the rendered pyramid
    """
    res = render_pyramid(pyr, levels)
    plt.figure()
    plt.imshow(res, cmap='gray')
    plt.show()


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im,
                     filter_size_mask):
    """
     Pyramid blending implementation
    :param im1: input grayscale image
    :param im2: input grayscale image
    :param mask: a boolean mask
    :param max_levels: max_levels for the pyramids
    :param filter_size_im: is the size of the Gaussian filter (an odd
            scalar that represents a squared filter)
    :param filter_size_mask: size of the Gaussian filter(an odd scalar
            that represents a squared filter) which defining the filter used
            in the construction of the Gaussian pyramid of mask
    :return: the blended image
    """
    mask_float = mask.astype(np.float64)
    mask_gauss = build_gaussian_pyramid(mask_float, max_levels, filter_size_mask)[0]
    im1_lap, im_filter = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    im2_lap = build_laplacian_pyramid(im2, max_levels, filter_size_im)[0]
    blend_lap = list()
    for k in range(min(len(im1_lap), len(im2_lap))):
        tmp = mask_gauss[k] * im1_lap[k] + (1 - mask_gauss[k]) * im2_lap[k]
        blend_lap.append(tmp)
    return np.clip(laplacian_to_image(blend_lap, im_filter, [1] * max_levels),0,1)


def blend_rgb(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    The function blend 2 rgb images by blending each color
    :param im1: an image represented by a numpy array shape(x,y,3)
    :param im2: an image represented by a numpy array shape(x,y,3)
    :param mask:  a boolean array of shape (x,y)
    :param max_levels:
    :param filter_size_im:
    :param filter_size_mask:
    :return:
    """
    blend_img = np.empty(im1.shape)
    for i in range(3):
        blend_img[:, :, i] = pyramid_blending(im1[:, :, i], im2[:, :, i], mask,
                                              max_levels, filter_size_im, filter_size_mask)
    return blend_img



def blending_example1():
    """
    Perform pyramid blending on two images RGB and a mask
    :return: image_1, image_2 the input images, mask the mask
        and out the blended image
    """
    im1 = read_image(relpath("BlendingPics/Blend5Image2.jpg"), RGB)
    im2 = read_image(relpath("BlendingPics/Blend5Image1.jpg"), RGB)
    mask = np.round(read_image(relpath("BlendingPics/Mask5Try2.jpg"), GRAYSCALE)).astype(bool)
    blended_img = blend_rgb(im1, im2, mask, 10, 5, 3)
    figure = plt.figure()
    top_left = figure.add_subplot(2, 2, 1)
    top_right = figure.add_subplot(2, 2, 2)
    bottom_left = figure.add_subplot(2, 2, 3)
    bottom_right = figure.add_subplot(2, 2, 4)
    top_left.imshow(im1)
    top_right.imshow(im2)
    bottom_left.imshow(mask, cmap='gray')
    bottom_right.imshow(blended_img)
    plt.show()
    return im1, im2, mask, blended_img


def blending_example2():
    """
    Perform pyramid blending on two images RGB and a mask
    :return: image_1, image_2 the input images, mask the mask
        and out the blended image
    """
    im1 = read_image(relpath("BlendingPics/EggEye2_1.jpg"), RGB)
    im2 = read_image(relpath("BlendingPics/EggEye1.jpg"), RGB)
    mask = np.round(read_image(relpath("BlendingPics/MaskEggEye.jpg"), GRAYSCALE)).astype(bool)
    blended_img = blend_rgb(im1, im2, mask, 10, 5, 3)
    figure = plt.figure()
    top_left = figure.add_subplot(2, 2, 1)
    top_right = figure.add_subplot(2, 2, 2)
    bottom_left = figure.add_subplot(2, 2, 3)
    bottom_right = figure.add_subplot(2, 2, 4)
    top_left.imshow(im1)
    top_right.imshow(im2)
    bottom_left.imshow(mask, cmap='gray')
    bottom_right.imshow(blended_img)
    plt.show()
    return im1, im2, mask, blended_img

if __name__ == '__main__':
    blending_example1()
    blending_example2()