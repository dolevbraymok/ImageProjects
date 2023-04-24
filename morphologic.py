import numpy as np
from scipy.signal import convolve2d

def erosion(img, kernel):
    pad_size = kernel.shape[0] // 2
    pad_img = np.pad(img, pad_size, mode='constant', constant_values=0)
    eroded_img = np.zeros(img.shape, dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if np.all(convolve2d(pad_img[i:i+2*pad_size+1, j:j+2*pad_size+1], kernel, mode='valid')):
                eroded_img[i, j] = 255
    return eroded_img


def dilation(image, kernel):
    pass

def opening(image, kernel):
    pass

def closing(image, kernel):
    pass

def gradient_morph(image, kernel):
    pass

def top_hat(image, kernel):
    pass

def bottom_hat(image, kernel):
    pass