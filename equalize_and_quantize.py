import imageio as io
import numpy as np
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

GRAYSCALE = 1
RGB = 2
RGB_YIQ_TRANSFORMATION_MATRIX = np.array([[0.299, 0.587, 0.114],
                                          [0.596, -0.275, -0.321],
                                          [0.212, -0.523, 0.311]])
BIN_SIZE = 256






def read_image(filename, representation):
    """
    Reads an image and converts it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    :return: Returns the image as a np.float64 matrix normalized to [0,1]
    """
    img = 0
    if representation != GRAYSCALE and representation != RGB:
        raise "representation value is Unknown"
    try:
        img = io.imread(filename)
    except FileNotFoundError:
        raise 'File does not exist'
    if representation == GRAYSCALE:
        return rgb2gray(img)
    else:
        return img / (BIN_SIZE - 1)

def rgb2yiq(imRGB):
    """
    Transform an RGB image into the YIQ color space
    :param imRGB: height X width X 3 np.float64 matrix in the [0,1] range
    :return: the image in the YIQ space
    """
    pre_shape = imRGB.shape
    tmp_matrix = imRGB.reshape(-1, 3)
    yiq_preshape = np.dot(tmp_matrix, RGB_YIQ_TRANSFORMATION_MATRIX.transpose())
    return yiq_preshape.reshape(pre_shape)


def yiq2rgb(imYIQ):
    """
    Transform a YIQ image into the RGB color space
    :param imYIQ: height X width X 3 np.float64 matrix in the [0,1] range for
        the Y channel and in the range of [-1,1] for the I,Q channels
    :return: the image in the RGB space
    """
    pre_shape = imYIQ.shape
    tmp_matrix = imYIQ.reshape(-1, 3)
    yiq_rgb_trans = np.linalg.inv(RGB_YIQ_TRANSFORMATION_MATRIX)
    rgb_preshape = np.dot(tmp_matrix, yiq_rgb_trans.transpose())
    return rgb_preshape.reshape(pre_shape)


def get_t_from_histogram(hist_orig):
    """"
     Function make the T  function (as an array_ from a histogram
     :param hist_orig: the histogram we want to make T function for
     :return an array where each array[k]=T(k)
    """
    cum_hist_orig = np.cumsum(hist_orig)  # cumulative histogram
    num_of_pixels = cum_hist_orig[-1]  # get pixel count
    norm_hist_orig = cum_hist_orig / num_of_pixels  # normalize
    max_gray_level = np.max(np.nonzero(norm_hist_orig))
    min_gray_level = np.min(np.nonzero(norm_hist_orig))
    # create the histogram after parts 4-6
    temp1 = (norm_hist_orig - norm_hist_orig[min_gray_level])
    temp1[temp1 < 0] = 0
    temp2 = (norm_hist_orig[max_gray_level] - norm_hist_orig[min_gray_level])
    if temp2 == 0:
        t_array = np.zeros(norm_hist_orig.shape)
        t_array[min_gray_level] = norm_hist_orig[min_gray_level]
    else:
        t_array = np.floor((temp1 / temp2) * (BIN_SIZE - 1)).astype(np.uint8)
    return t_array


def img_equalize_rgb(t_arr, im_orig):
    """
    The function recive an array as T function and a RGB(already in YIQ ) image and equalize it
    :param t_arr: T function to equalize the image with
    :param im_orig:  the YIQ for the RGB image we want to equalize
    :return: the equalized image(still in YIQ)
    """
    tmp_img_y = im_orig[:, :, 0]
    tmp_img_y = np.round(tmp_img_y * 255).astype(np.uint8)
    tmp_img_y_shape = tmp_img_y.shape
    tmp_img_i = im_orig[:, :, 1]
    tmp_img_q = im_orig[:, :, 2]
    tmp_img_y = tmp_img_y.flatten()
    tmp_img_y = (lambda k: t_arr[k])(tmp_img_y)
    tmp_img_y = tmp_img_y.reshape(tmp_img_y_shape)
    tmp_img_y = (tmp_img_y.astype(np.float64)) / 255
    tmp = np.dstack((tmp_img_y, tmp_img_i, tmp_img_q))
    return tmp


def img_equlize_gray(t_arr, im_orig):
    """
    The function recive an array as T function and a grayscale image and equalize it
    :param t_arr: T function to equalize the image with
    :param im_orig: the grayscale presentation  for the image we want to equalize
    :return: an equalized image (as a np array)
    """

    tmp_im = np.round(im_orig * 255).astype(np.uint8)
    im_orig_shape = im_orig.shape
    tmp_im = tmp_im.reshape(-1)
    t_func = lambda k: t_arr[k]
    tmp_im = t_func(tmp_im)
    tmp_im = tmp_im.reshape(im_orig_shape)
    return (tmp_im.astype(np.float64)) / 255


def histogram_equalize(im_orig):
    """
    Perform histogram equalization on the given image
    :param im_orig: Input float64 [0,1] image
    :return: [im_eq, hist_orig, hist_eq]
    """
    tmp_im = im_orig
    if len(im_orig.shape) == 3 and im_orig.shape[2] == 3:
        is_rgb = True
        tmp_im = rgb2yiq(im_orig)
        hist_orig = np.histogram(tmp_im[:, :, 0], BIN_SIZE)
    else:
        is_rgb = False
        hist_orig = np.histogram(tmp_im, BIN_SIZE)

    T = get_t_from_histogram(hist_orig[0])
    if is_rgb:
        im_eq = img_equalize_rgb(T, tmp_im)
    else:
        im_eq = img_equlize_gray(T, tmp_im)
    if is_rgb:
        im_eq = yiq2rgb(im_eq)

    return [im_eq, hist_orig[0], np.histogram(im_eq, BIN_SIZE)[0]]


def quantize_get_q_from_z(histo, z_arr, n_quant):
    """
    Function get a histogram and a partition  and return quantize values
    :param histo: 1d Array represents the histogram
    :param z_arr: 1d array with size of n_quant+1 represents the indexes of the partition
    :param n_quant: the size of the return array of quantize values
    :return:an array of size n_quant with each element represents the value to quantize the image scales to
    """
    q_arr = np.zeros(n_quant)
    g_arr = np.arange(BIN_SIZE)
    floord_z_arr = (np.floor(z_arr)).astype(np.int)
    for i in range(n_quant):
        counter = np.sum((histo[floord_z_arr[i] + 1: floord_z_arr[i + 1] + 1])
                         * (g_arr[floord_z_arr[i] + 1: floord_z_arr[i + 1] + 1]))
        denominator = np.sum((histo[floord_z_arr[i] + 1: floord_z_arr[i + 1] + 1]))
        if denominator == 0:
            raise "Error Division by Zero , n_quant too big"
        q_arr[i] = counter / denominator
    return q_arr


def quantize_get_z_from_q(q_arr, n_quant):
    """
    The function receive a quantize values array and its size and return an array represents a partiotion for them
    :param q_arr: a quantize values arrays size of n_quant
    :param n_quant: the size of the q_arr
    :return: return an array represents the partition with each elemnts hold a value of index
            array size is n_quant+1
    """
    z_arr = np.zeros(n_quant + 1)
    z_arr[0] = -1
    z_arr[n_quant] = BIN_SIZE - 1
    for i in range(1, n_quant):
        z_arr[i] = (q_arr[i - 1] + q_arr[i]) / 2
    return z_arr


def quantize_error_calculation(histo, q_arr, z_arr, n_quant):
    """
    The function calculate the error of the quantize iteration
    :param histo: The histogram we work on
    :param q_arr: the quantize values
    :param z_arr: the quantize indexes
    :param n_quant: the size of q_array
    :return: a number represents the error
    """
    helper_arr = np.arange(BIN_SIZE)
    helper_arr2 = np.zeros(BIN_SIZE)
    floord_z_arr = (np.floor(z_arr)).astype(np.int)
    for i in range(n_quant):
        helper_arr2[floord_z_arr[i] + 1: floord_z_arr[i + 1] + 1] = q_arr[i]
    return np.sum((np.square(helper_arr - helper_arr2)) * histo)


def find_first_z_arr(histo, n_quant):
    """
    The function get the first partition for the quantize algorithm by dividing it
      to partiton with similar number of pixels
    :param histo: The histogram we base the partition on
    :param n_quant: the number of colors we want to quantize to
    :return: an array of a partition with similar number of pixels in each part of the partition
    """
    my_cumsum = np.cumsum(histo)
    z_arr = np.empty(n_quant + 1)
    z_arr[0] = -1
    z_arr[n_quant] = BIN_SIZE - 1
    pixels_per_z = np.floor(my_cumsum[-1] / n_quant)
    for i in range(1, n_quant):
        my_cumsum[my_cumsum < pixels_per_z] = 0
        z_arr[i] = np.min(np.nonzero(my_cumsum))
        my_cumsum = my_cumsum - pixels_per_z
    return z_arr


def quant_iterations(histo, n_iter, n_quant):
    """
    The function to the iterations of the quantize to decide the partition indexes and the values of each
        for the quantization function
    :param histo: The histogram of the image we quantize
    :param n_iter: the maximal number of iteration before stopping
    :param n_quant: the number of colors we want to quantize the image to
    :return: an array of the error in [0] the values to quantize in [1] and their partition in [2]
    """
    z_arr = find_first_z_arr(histo, n_quant)
    errors = np.empty(n_iter)
    q_arr = np.empty(n_quant)
    for i in range(n_iter):
        q_arr = quantize_get_q_from_z(histo, z_arr, n_quant)
        errors[i] = quantize_error_calculation(histo, q_arr, z_arr, n_quant)
        z_arr = quantize_get_z_from_q(q_arr, n_quant)
        if i != 0 and errors[i - 1] < errors[i]:
            errors = errors[:i]
            break

    return errors, q_arr, z_arr


def creating_t_arr_for_quant(z_arr, q_arr, n_quant):
    """
    The function get the array of Z and array of q and the n_quant and create an array to change the values of the
    image
    :param z_arr: the indexes  for each partition for the new values
    :param q_arr: the new values we put in each partition
    :param n_quant: the number of colors we quantize to
    :return: an array where each index hold the value that he passes the values of each index to
    """
    t_arr = np.zeros(BIN_SIZE)
    floord_z_arr = (np.floor(z_arr)).astype(np.int)
    for i in range(n_quant):
        t_arr[floord_z_arr[i] + 1: floord_z_arr[i + 1] + 1] = q_arr[i]
    return np.floor(t_arr)


def quantize(im_orig, n_quant, n_iter):
    """
    Performs optimal quantization of a given greyscale or RGB image
    :param im_orig: Input float64 [0,1] image
    :param n_quant: Number of intensities im_quant image will have
    :param n_iter: Maximum number of iterations of the optimization
    :return:  im_quant - is the quantized output image
              error - is an array with shape (n_iter,) (or less) of
                the total intensities error for each iteration of the
                quantization procedure
    """
    tmp_im = im_orig
    if len(im_orig.shape) == 3:
        is_rgb = True
        tmp_im = rgb2yiq(im_orig)
        histo = np.histogram(tmp_im[:, :, 0], BIN_SIZE)[0]
    else:
        is_rgb = False
        histo = np.histogram(tmp_im, BIN_SIZE)[0]
    errors, q_arr, z_arr = quant_iterations(histo, n_iter, n_quant)
    t_arr = creating_t_arr_for_quant(z_arr, q_arr, n_quant)
    if is_rgb:
        im_quant = img_equalize_rgb(t_arr, tmp_im)
        im_quant = rgb2yiq(im_quant)
    else:
        im_quant = img_equlize_gray(t_arr, tmp_im)
    return [im_quant, errors]
