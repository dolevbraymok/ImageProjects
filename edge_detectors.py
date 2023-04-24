import cv2
import numpy as np
import filters

SOBELX_KERNEL = np.array(([-1,0,1],[-2,0,2],[-1,0,1]),dtype='int')
SOBELY_KERNEL = numpy.transpose(SOBELX_KERNEL)


def canny_edge_detector(gray):
    """
    straightforward  implementation, inefficient
    assume image enterd already grayscale
    :param image:
    :return:
    """

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), sigma)

    # Compute gradients using Sobel filter
    gx = convolve(image,SOBELX_KERNEL)
    gy = convolve(image,SOBELY_KERNEL)

    # Compute magnitude and orientation of gradients
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    # Non-maximum suppression
    suppressed = np.zeros_like(magnitude)
    for i in range(1, magnitude.shape[0] - 1):
        for j in range(1, magnitude.shape[1] - 1):
            if (angle[i, j] >= 0 and angle[i, j] < 22.5) or (angle[i, j] >= 157.5 and angle[i, j] < 202.5) or (
                    angle[i, j] >= 337.5 and angle[i, j] <= 360):
                if (magnitude[i, j] > magnitude[i - 1, j]) and (magnitude[i, j] > magnitude[i + 1, j]):
                    suppressed[i, j] = magnitude[i, j]
            elif (angle[i, j] >= 22.5 and angle[i, j] < 67.5) or (angle[i, j] >= 202.5 and angle[i, j] < 247.5):
                if (magnitude[i, j] > magnitude[i - 1, j - 1]) and (magnitude[i, j] > magnitude[i + 1, j + 1]):
                    suppressed[i, j] = magnitude[i, j]
            elif (angle[i, j] >= 67.5 and angle[i, j] < 112.5) or (angle[i, j] >= 247.5 and angle[i, j] < 292.5):
                if (magnitude[i, j] > magnitude[i, j - 1]) and (magnitude[i, j] > magnitude[i, j + 1]):
                    suppressed[i, j] = magnitude[i, j]
            elif (angle[i, j] >= 112.5 and angle[i, j] < 157.5) or (angle[i, j] >= 292.5 and angle[i, j] < 337.5):
                if (magnitude[i, j] > magnitude[i - 1, j + 1]) and (magnitude[i, j] > magnitude[i + 1, j - 1]):
                    suppressed[i, j] = magnitude[i, j]

    # Double thresholding
    thresholded = np.zeros_like(suppressed)
    thresholded[suppressed > high_threshold] = 255
    thresholded[(suppressed >= low_threshold) & (suppressed <= high_threshold)] = 128

    # Edge tracking by hysteresis
    for i in range(1, thresholded.shape[0] - 1):
        for j in range(1, thresholded.shape[1] - 1):
            if thresholded[i, j] == 128:
                if (thresholded[i - 1, j - 1] == 255) or (thresholded[i - 1, j] == 255) or (
                        thresholded[i - 1, j + 1] == 255) or \
                        (thresholded[i, j - 1] == 255) or (thresholded[i, j + 1] == 255) or (
                        thresholded[i + 1, j - 1] == 255) or \
                        (thresholded[i + 1,j] == 255):
                    thresholded[i,j] = 255
                else:
                    thresholded[i,j] = 0

    return thresholded.astype(np.uint8)




def sobel_edge_detector(gray, threshold):
    """
    sobel edge detector implementation
    assume image enterd already grayscale
    :param gray:
    :param threshold:
    :return:
    """

    # Apply Sobel filters
    sobelx = filters.convolve(gray,SOBELX_KERNEL)
    sobely = filters.convolve(gray,SOBELY_KERNEL)

    # Compute gradient magnitude and direction
    gradient_magnitude = np.sqrt(np.square(sobelx) + np.square(sobely))
    gradient_direction = np.arctan2(sobely, sobelx)

    # Apply threshold to detect edges
    binary_output = np.zeros_like(gray)
    binary_output[gradient_magnitude > threshold] = 255

    return binary_output