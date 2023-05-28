import cv2
import numpy as np
from skimage.exposure import rescale_intensity

ZERO_PADDING = 1
REFLECT_PADDING = 2
REPLICATE_PADDING = 3
WARP_PADDING = 4

SHAPEN_KERNEL = np.array(([0,-1,0],[-1,5,-1],[0,-1,0]),dtype='int')
LAPLACIAN_KERNEL = np.array(([0,1,0],[1,-4,1],[0,1,0]),dtype='int')


CONV_GAUSS_CREATOR =[1,1]
def convolve(image, kernel, padMethod=REPLICATE_PADDING):
    """
    A simple convolution method with few padding options
    OpenCV have its own option: cv2.filter2D
    :param image: the image we convolve on
    :param kernel: kernel
    :param padMethod: ZERO_PADDING,REFLECT_PADDING_REPLICATE_PADDING,WARP_PADDING name self explained
    :return:
    """
    (iH, iW)= image.shape[:2]
    (kH, kW)= kernel.shape[:2]
    pad = (kW - 1)//2
    if padMethod == WARP_PADDING:
        image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_WRAP)
    elif padMethod == REFLECT_PADDING:
        image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    elif padMethod == REPLICATE_PADDING:
        image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    else:
        image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT)

    ret = np.zeros((iH, iW), dtype="float32")
    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            roi = image[y - pad:y + pad + 1, x - pad:x+pad+1]
            k = (roi * kernel).sum()
            ret[y-pad,x-pad]=k
    ret = rescale_intensity(ret, in_range=(0,255))
    ret = (ret * 255).astype("uint8")
    return ret


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




def shapenFilter(image):
    """
    sharpen an RGB image
    :param image: RGB image
    :return:sharpend RGB image
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    light, a, b = cv2.split(lab)

    #sharp by convolution
    light = convolve(light, SHAPEN_KERNEL)

    #increase contrast by adaptive histogram equilization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    light_eq = clahe.apply(light)
    lab_eq = cv2.merge((light_eq, a, b))


    #apply sharpening filter and return
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)




if __name__ == '__main__':
    image = cv2.imread("Images/friends_faces.jpg")
    sharp = shapenFilter(image)
    cv2.imshow("name1",image)
    cv2.imshow("name2", sharp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()