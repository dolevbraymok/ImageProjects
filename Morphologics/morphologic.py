import numpy as np
from scipy.signal import convolve2d
import cv2


# Note:cv2 have its own implementation for the function implemented here


def erosion(img, kernel):
    """
    Shrinks the boundaries of foreground objects in an image
    :param img:binary or grayscale image we want to erode
    :param kernel: kernel we convolve with
    :return: eroded image
    """
    pad_size = kernel.shape[0] // 2
    pad_img = np.pad(img, pad_size, mode='constant', constant_values=0)
    eroded_img = np.zeros(img.shape, dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if np.all(convolve2d(pad_img[i:i + 2 * pad_size + 1, j:j + 2 * pad_size + 1], kernel, mode='valid')):
                eroded_img[i, j] = 255
    return eroded_img


def dilation(image, kernel):
    """
    Expands the boundaries of foreground objects in an image
    :param image: binary or grayscale image we want to dilate
    :param kernel: kernel we convolve with
    :return: dilated image
    """
    padded_img = np.pad(image, ((kernel.shape[0] // 2,
                                 kernel.shape[0] // 2), (kernel.shape[1] // 2,
                                                         kernel.shape[1] // 2)), mode='constant')
    output = np.zeros(image.shape, dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            output[i, j] = np.max(padded_img[i:i + kernel.shape[0], j:j + kernel.shape[1]] * kernel)
    return output


def opening(image, kernel_size):
    """
    Erosion followed by dilation, used to remove small objects from an image.
    :param image: binary or grayscale image to open
    :param kernel_size:self explained
    :return: opend image
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return dilation(erosion(image, kernel), kernel)


def closing(image, kernel_size):
    """
    Dilation followed by erosion, used to fill in small holes in an image
    :param image:binary or grayscale image to close
    :param kernel_size: self explained
    :return: closed image
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return erosion(dilation(image, kernel), kernel)


def gradient_morph(image, kernel):
    """
    dilation - erosion :  can be used to highlight edges and boundries of objects
    :param image:binary or grayscale image
    :param kernel:self explained
    :return: dilation - erosion
    """
    return dilation(image, kernel) - erosion(image, kernel)


def top_hat(image, kernel_size):
    """
    the difference between the input image and its opening.
     It is used to enhance bright features on a dark background.
    :param image:binary or grayscale image
    :param kernel_size:
    :return:
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return image - opening(image, kernel)


def bottom_hat(image, kernel_size):
    """
    the difference between the closing of the input image and the input image itself.
     It is used to enhance dark features on a bright backgroun
    :param image:binary or grayscale image
    :param kernel_size:
    :return:
    """
    return closing(image, kernel_size) - image


# Small function that count fingers in an image of hand as an example for morphological usage


def finger_counter(filename):
    kernel_size = 5

    # Read the input image
    image = cv2.imread(filename)
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to remove noise
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    # Threshold the image to create a binary image
    _, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    closed = closing(thresh, kernel_size)
    # Find contours in the eroded image
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Count the number of contours
    num_fingers = len(contours)
    # Draw the contours on the original image
    cv2.drawContours(image, contours, -1, (0, 0, 255), 2)
    # Display the image and the number of fingers
    cv2.imshow('Fingers', image)
    print("Number of fingers:", num_fingers)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# small function that detect lanes use morphology to fill the edges and reduce noise
def lane_detection(filename):
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # getting edges
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150)

    # define a ROI
    height, width = image.shape[:2]
    roi_vertices = [(0, height), (width / 2, height / 2), (width, height)]
    roi_mask = np.zeros_like(edges)
    cv2.fillPoly(roi_mask, np.array([roi_vertices], np.int32), 255)
    roi = cv2.bitwise_and(edges, roi_mask)

    # dilation for gaps filling in the edges and erosion for noise reduction
    closed = closing(roi, 3)

    # using hough transform to detect lines in the image and draw them on original image.
    lines = cv2.HoughLinesP(closed, rho=1, theta=np.pi / 180,
                            threshold=20, minLineLength=50, maxLineGap=10)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow("Lane Detector", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
