
import time
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.ndimage
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage import label, center_of_mass
from scipy.ndimage import convolve
import shutil
from imageio import imwrite




DERIV_FILTER = np.array([[1, 0, -1]]).astype(np.float64)
K_HARRIS_CONST = 0.04

DESC_RADIUS = 3
SPREAD_CORNERS_PARAM = 7


def harris_corner_detector(im):
    """
  Detects harris corners.
  Make sure the returned coordinates are x major!!!
  :param im: A 2D array representing an image.
  :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
  """
    conv = DERIV_FILTER.astype(im.dtype).reshape(3, 1)
    I_x = convolve(im, conv)
    I_y = convolve(im, conv.T)
    I_x_pow = sol4_utils.blur_spatial(np.power(I_x, 2), 3)
    I_y_pow = sol4_utils.blur_spatial(np.power(I_y, 2), 3)
    I_x_y = sol4_utils.blur_spatial(I_x * I_y, 3)

    tmp_matrix = np.dstack((I_x_pow, I_x_y, I_x_y, I_y_pow)).reshape((-1, 2, 2))
    det = np.linalg.det(tmp_matrix)
    tra = K_HARRIS_CONST * (np.trace(tmp_matrix.T) ** 2)
    coord_t = np.where(non_maximum_suppression((det - tra).reshape(im.shape)) == True)
    return np.dstack((coord_t[1], coord_t[0])).reshape(-1, 2)


def get_patch(im, p, desc_rad):
    """
    The function return a patch around p  with desc_rad radius
    :param g_pyr: gaussian pyramid for values
    :param p: the index we patch around
    :param desc_rad: the radius of the patch
    :return: np.array with shape (1+2*desc_rad, 1+2*desc_rad)
    """
    small_x = p[1] - desc_rad
    small_y = p[0] - desc_rad
    big_x = p[1] + desc_rad + 1
    big_y = p[0] + desc_rad + 1
    coords = np.mgrid[small_x:big_x, small_y: big_y]
    return scipy.ndimage.map_coordinates(im, coords, order=1,
                                                  prefilter=False).reshape(1 + 2 * desc_rad, 1 + 2 * desc_rad)


def sample_descriptor(im, pos, desc_rad):
    """
  Samples descriptors at the given corners.
  :param im: A 2D array representing an image.
  :param pos: An array with shape (N,2), where pos[i,:] are the [x,y] coordinates of the ith corner point.
  :param desc_rad: "Radius" of descriptors to compute.
  :return: A 3D array with shape (N,K,K) containing the ith descriptor at desc[i,:,:].
  """
    desc_container = list()
    np.apply_along_axis(create_desc_container, 1, pos,
                        desc_container=desc_container, desc_rad=desc_rad, im=im)
    ret = np.array(desc_container).astype(np.float64)
    return ret


def create_desc_container(p, desc_container, desc_rad, im):
    """
    The function appends a descriptor of a point around an image to a container
    :param p: the point of the descriptor
    :param desc_container: the container we append the descriptor to
    :param desc_rad: the radius of the descriptor
    :param im: the image
    :return: None
    """
    p_patch = get_patch(im, p, desc_rad)
    p_mean = np.mean(p_patch)
    p_norm = np.linalg.norm(p_patch - p_mean)
    if p_norm == 0:
        desc_container.append(np.zeros((1 + 2 * desc_rad, 1 + 2 * desc_rad)))
    else:
        desc_container.append((p_patch - p_mean) / p_norm)



def find_features(pyr):
    """
  Detects and extracts feature points from a pyramid.
  :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
  :return: A list containing:
              1) An array with shape (N,2) of [x,y] feature location per row found in the image.
                 These coordinates are provided at the pyramid level pyr[0].
              2) A feature descriptor array with shape (N,K,K)
  """
    pos = spread_out_corners(pyr[0], SPREAD_CORNERS_PARAM, SPREAD_CORNERS_PARAM, 1 + (4 * DESC_RADIUS))
    return [pos, sample_descriptor(pyr[2], pos / 4, DESC_RADIUS)]


def match_features(desc1, desc2, min_score):
    """
  Return indices of matching descriptors.
  :param desc1: A feature descriptor array with shape (N1,K,K).
  :param desc2: A feature descriptor array with shape (N2,K,K).
  :param min_score: Minimal match score.
  :return: A list containing:
              1) An array with shape (M,) and dtype int of matching indices in desc1.
              2) An array with shape (M,) and dtype int of matching indices in desc2.
  """
    check = create_scores(desc1, desc2, desc1.shape[1], min_score)
    indexes = np.where(check == True)
    return [indexes[0], indexes[1]]


def create_scores(desc1, desc2, k, min_score):
    """
    The function get to np.arrays of descriptor of the same size and a minimum score and return
    a boolean matrix such that (i,j)==true iff desc1[i] and desc2[j] are a match
    :param desc1: an array of descriptors size (N,k,k)
    :param desc2: an array of descriptors size (M,k,k)
    :param k: an int represents the dimensions of a descriptor
    :param min_score: the minimum score we want between 2 feature points
    :return:
    """
    scores = np.tensordot(desc1.reshape(desc1.shape[0], k * k), desc2.reshape(desc2.shape[0], k * k), [1, 1])
    row_max = scores.argsort(axis=1)[:, -2:]
    row_max = np.array(np.dstack((np.repeat(np.arange(scores.shape[0]), 2),
                                  row_max[:, -2:].reshape(-1)))).reshape(-1, 2)
    row_matrix = np.full((scores.shape[0] * scores.shape[1]), False).reshape((scores.shape[0], scores.shape[1]))
    row_matrix[row_max[:, 0], row_max[:, 1]] = True
    column_max = scores.argsort(axis=0).T[:, -2:]
    column_max = np.array(np.dstack((np.repeat(np.arange(scores.shape[1]), 2)
                                     , column_max[:, -2:].reshape(-1)))).reshape(-1, 2)
    column_matrix = np.full((scores.shape[0] * scores.shape[1]), False).reshape((scores.shape[0], scores.shape[1]))
    column_matrix[column_max[:, 1], column_max[:, 0]] = True

    min_score_matrix = scores > min_score
    return column_matrix & row_matrix & min_score_matrix


def apply_homography(pos1, H12):
    """
  Apply homography to inhomogenous points.
  :param pos1: An array with shape (N,2) of [x,y] point coordinates.
  :param H12: A 3x3 homography matrix.
  :return: An array with the same shape as pos1 with [x,y] point coordinates obtained from transforming pos1 using H12.
  """
    pos1_3d = np.hstack((pos1, np.ones((pos1.shape[0], 1))))
    post_mult = np.dot(H12, pos1_3d.T).T
    return post_mult[:, : -1] / np.array([post_mult[:, 2]]).T


def ransac_homography(points1, points2, num_iter, inlier_tol, translation_only=False):
    """
  Computes homography between two sets of points using RANSAC.
  :param points1: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 1.
  :param points2: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 2.
  :param num_iter: Number of RANSAC iterations to perform.
  :param inlier_tol: inlier tolerance threshold.
  :param translation_only: see estimate rigid transform
  :return: A list containing:
              1) A 3x3 normalized homography matrix.
              2) An Array with shape (S,) where S is the number of inliers,
                  containing the indices in pos1/pos2 of the maximal set of inlier matches found.
  """
    if translation_only:
        num_of_p = 1
    else:
        num_of_p = 2
    num_of_match = points1.shape[0]
    max_index = np.array([])
    for i in range(num_iter):
        indexes = np.random.choice(num_of_match, num_of_p, False)
        cur_p1 = points1[indexes]
        cur_p2 = points2[indexes]
        homography = estimate_rigid_transform(cur_p1, cur_p2, translation_only)
        p2_prime = apply_homography(points1, homography)
        dist = np.linalg.norm(p2_prime - points2, axis=1) ** 2
        good_index = np.where(dist < inlier_tol)[0]
        if len(good_index) > len(max_index):
            max_index = good_index
    chosen_p1 = points1[max_index, :]
    chosen_p2 = points2[max_index, :]
    best_homography = estimate_rigid_transform(chosen_p1, chosen_p2, translation_only)
    return [best_homography / best_homography[2, 2], max_index]


def display_matches(im1, im2, points1, points2, inliers):
    """
  Dispalay matching points.
  :param im1: A grayscale image.
  :param im2: A grayscale image.
  :parma points1: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im1.
  :param points2: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im2.
  :param inliers: An array with shape (S,) of inlier matches.
  """

    im_stack = pad_images(im1, im2)
    plt.imshow(im_stack, cmap='gray')
    outliners = np.setdiff1d(np.arange(points1.shape[0]), inliers)
    inliners_x = [points1[inliers][:, 0], im1.shape[1] + points2[inliers][:, 0]]
    outliners_x = [points1[outliners][:, 0], im1.shape[1] + points2[outliners][:, 0]]
    inliners_y = [points1[inliers][:, 1], points2[inliers][:, 1]]
    outliners_y = [points1[outliners][:, 1], points2[outliners][:, 1]]

    plt.plot(outliners_x, outliners_y,color='blue', marker='o', markerfacecolor='red',
             markersize=3, lw=.5)
    plt.plot(inliners_x, inliners_y, color='yellow', marker='o', markerfacecolor='red',
             markersize=3, lw=.5)

    plt.show()


def pad_images(im1, im2):
    """
    The function gets to image and pad one of them to stack them together
    :param im1: image to appear on the left side
    :param im2: image to appear on the right side
    :return: new image such that im1 is on the left and im2 on the right
    """
    max_len = max(im1.shape[1], im2.shape[1])
    pad_im1 = np.pad(im1, ((0, max_len - im1.shape[1]), (0, 0)), mode='constant', constant_values=0)
    pad_im2 = np.pad(im2, ((0, max_len - im2.shape[1]), (0, 0)), mode='constant', constant_values=0)
    return np.hstack((pad_im1, pad_im2))


def accumulate_homographies(H_succesive, m):
    """
  Convert a list of succesive homographies to a
  list of homographies to a common reference frame.
  :param H_succesive: A list of M-1 3x3 homography
    matrices where H_successive[i] is a homography which transforms points
    from coordinate system i to coordinate system i+1.
  :param m: Index of the coordinate system towards which we would like to
    accumulate the given homographies.
  :return: A list of M 3x3 homography matrices,
    where H2m[i] transforms points from coordinate system i to coordinate system m
  """
    h_succesive_len = len(H_succesive)
    h_m_list = [np.eye(3)]
    for cur_homograph in H_succesive[m:]:
        temp_homo = np.dot(h_m_list[-1] , np.linalg.inv(cur_homograph))
        h_m_list.append(temp_homo/temp_homo[2,2])
    for cur_homograph in H_succesive[::-1][h_succesive_len - m:]:
        temp_homo = np.dot(h_m_list[0], cur_homograph)
        h_m_list.insert(0, temp_homo/temp_homo[2,2])
    return h_m_list


def compute_bounding_box(homography, w, h):
    """
  computes bounding box of warped image under homography, without actually warping the image
  :param homography: homography
  :param w: width of the image
  :param h: height of the image
  :return: 2x2 array, where the first row is [x,y] of the top left corner,
   and the second row is the [x,y] of the bottom right corner
  """
    tl = np.dot(homography, np.array([0, 0, 1]))
    bl = np.dot(homography, np.array([0, h, 1]))
    tr = np.dot(homography, np.array([w, 0, 1]))
    br = np.dot(homography, np.array([w, h, 1]))
    new_coord = np.array([tl, bl, tr, br])
    return np.array([[np.min(new_coord[:, 0]), np.min(new_coord[:, 1])],
                     [np.max(new_coord[:, 0]), np.max(new_coord[:, 1])]]).astype(np.int)


def warp_channel(image, homography):
    """
  Warps a 2D image with a given homography.
  :param image: a 2D image.
  :param homography: homograhpy.
  :return: A 2d warped image.
  """
    top_left, bot_right = compute_bounding_box(homography, image.shape[1], image.shape[0])
    height_change = (bot_right[0] - top_left[0]) + 1
    width_change = (bot_right[1] - top_left[1]) + 1
    coords = np.dstack(np.meshgrid((np.arange(top_left[0], bot_right[0] + 1)),
                                   np.arange(top_left[1], bot_right[1] + 1))).reshape(-1, 2)
    warped_coords = apply_homography(coords, np.linalg.inv(homography))
    return scipy.ndimage.map_coordinates(image, np.array([warped_coords[:,1], warped_coords[:,0]]), order=1,
                                         prefilter=False).reshape(width_change, height_change)



def warp_image(image, homography):
    """
  Warps an RGB image with a given homography.
  :param image: an RGB image.
  :param homography: homograhpy.
  :return: A warped image.
  """
    return np.dstack([warp_channel(image[..., channel], homography) for channel in range(3)])


def filter_homographies_with_translation(homographies, minimum_right_translation):
    """
  Filters rigid transformations encoded as homographies by the amount of translation from left to right.
  :param homographies: homograhpies to filter.
  :param minimum_right_translation: amount of translation below which the transformation is discarded.
  :return: filtered homographies..
  """
    translation_over_thresh = [0]
    last = homographies[0][0, -1]
    for i in range(1, len(homographies)):
        if homographies[i][0, -1] - last > minimum_right_translation:
            translation_over_thresh.append(i)
            last = homographies[i][0, -1]
    return np.array(translation_over_thresh).astype(np.int)


def estimate_rigid_transform(points1, points2, translation_only=False):
    """
  Computes rigid transforming points1 towards points2, using least squares method.
  points1[i,:] corresponds to poins2[i,:]. In every point, the first coordinate is *x*.
  :param points1: array with shape (N,2). Holds coordinates of corresponding points from image 1.
  :param points2: array with shape (N,2). Holds coordinates of corresponding points from image 2.
  :param translation_only: whether to compute translation only. False (default) to compute rotation as well.
  :return: A 3x3 array with the computed homography.
  """
    centroid1 = points1.mean(axis=0)
    centroid2 = points2.mean(axis=0)

    if translation_only:
        rotation = np.eye(2)
        translation = centroid2 - centroid1

    else:
        centered_points1 = points1 - centroid1
        centered_points2 = points2 - centroid2

        sigma = centered_points2.T @ centered_points1
        U, _, Vt = np.linalg.svd(sigma)

        rotation = U @ Vt
        translation = -rotation @ centroid1 + centroid2

    H = np.eye(3)
    H[:2, :2] = rotation
    H[:2, 2] = translation
    return H


def non_maximum_suppression(image):
    """
  Finds local maximas of an image.
  :param image: A 2D array representing an image.
  :return: A boolean array with the same shape as the input image, where True indicates local maximum.
  """
    # Find local maximas.
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(image, footprint=neighborhood) == image
    local_max[image < (image.max() * 0.1)] = False

    # Erode areas to single points.
    lbs, num = label(local_max)
    centers = center_of_mass(local_max, lbs, np.arange(num) + 1)
    centers = np.stack(centers).round().astype(np.int)
    ret = np.zeros_like(image, dtype=np.bool)
    ret[centers[:, 0], centers[:, 1]] = True

    return ret


def spread_out_corners(im, m, n, radius):
    """
  Splits the image im to m by n rectangles and uses harris_corner_detector on each.
  :param im: A 2D array representing an image.
  :param m: Vertical number of rectangles.
  :param n: Horizontal number of rectangles.
  :param radius: Minimal distance of corner points from the boundary of the image.
  :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
  """
    corners = [np.empty((0, 2), dtype=np.int)]
    x_bound = np.linspace(0, im.shape[1], n + 1, dtype=np.int)
    y_bound = np.linspace(0, im.shape[0], m + 1, dtype=np.int)
    for i in range(n):
        for j in range(m):
            # Use Harris detector on every sub image.
            sub_im = im[y_bound[j]:y_bound[j + 1], x_bound[i]:x_bound[i + 1]]
            sub_corners = harris_corner_detector(sub_im)
            sub_corners += np.array([x_bound[i], y_bound[j]])[np.newaxis, :]
            corners.append(sub_corners)
    corners = np.vstack(corners)
    legit = ((corners[:, 0] > radius) & (corners[:, 0] < im.shape[1] - radius) &
             (corners[:, 1] > radius) & (corners[:, 1] < im.shape[0] - radius))
    ret = corners[legit, :]
    return ret


class PanoramicVideoGenerator:
    """
  Generates panorama from a set of images.
  """

    def __init__(self, data_dir, file_prefix, num_images, bonus=False):
        """
    The naming convention for a sequence of images is file_prefixN.jpg,
    where N is a running number 001, 002, 003...
    :param data_dir: path to input images.
    :param file_prefix: see above.
    :param num_images: number of images to produce the panoramas with.
    """
        self.bonus = bonus
        self.file_prefix = file_prefix
        self.files = [os.path.join(data_dir, '%s%03d.jpg' % (file_prefix, i + 1)) for i in range(num_images)]
        self.files = list(filter(os.path.exists, self.files))
        self.panoramas = None
        self.homographies = None
        print('found %d images' % len(self.files))

    def align_images(self, translation_only=False):
        """
    compute homographies between all images to a common coordinate system
    :param translation_only: see estimte_rigid_transform
    """
        # Extract feature point locations and descriptors.
        points_and_descriptors = []
        for file in self.files:
            image = sol4_utils.read_image(file, 1)
            self.h, self.w = image.shape
            pyramid, _ = sol4_utils.build_gaussian_pyramid(image, 3, 7)
            points_and_descriptors.append(find_features(pyramid))

        # Compute homographies between successive pairs of images.
        Hs = []
        for i in range(len(points_and_descriptors) - 1):
            points1, points2 = points_and_descriptors[i][0], points_and_descriptors[i + 1][0]
            desc1, desc2 = points_and_descriptors[i][1], points_and_descriptors[i + 1][1]

            # Find matching feature points.
            ind1, ind2 = match_features(desc1, desc2, .7)
            points1, points2 = points1[ind1, :], points2[ind2, :]

            # Compute homography using RANSAC.
            H12, inliers = ransac_homography(points1, points2, 100, 6, translation_only)

            # Uncomment for debugging: display inliers and outliers among matching points.
            # In the submitted code this function should be commented out!
            # display_matches(self.images[i], self.images[i+1], points1 , points2, inliers)

            Hs.append(H12)

        # Compute composite homographies from the central coordinate system.
        accumulated_homographies = accumulate_homographies(Hs, (len(Hs) - 1) // 2)
        self.homographies = np.stack(accumulated_homographies)
        self.frames_for_panoramas = filter_homographies_with_translation(self.homographies, minimum_right_translation=5)
        self.homographies = self.homographies[self.frames_for_panoramas]

    def generate_panoramic_images(self, number_of_panoramas):
        """
        combine slices from input images to panoramas.
        :param number_of_panoramas: how many different slices to take from each input image
        """
        if self.bonus:
            self.generate_panoramic_images_bonus(number_of_panoramas)
        else:
            self.generate_panoramic_images_normal(number_of_panoramas)

    def generate_panoramic_images_normal(self, number_of_panoramas):
        """
    combine slices from input images to panoramas.
    :param number_of_panoramas: how many different slices to take from each input image
    """
        assert self.homographies is not None

        # compute bounding boxes of all warped input images in the coordinate system of the middle image (as given by the homographies)
        self.bounding_boxes = np.zeros((self.frames_for_panoramas.size, 2, 2))
        for i in range(self.frames_for_panoramas.size):
            self.bounding_boxes[i] = compute_bounding_box(self.homographies[i], self.w, self.h)

        # change our reference coordinate system to the panoramas
        # all panoramas share the same coordinate system
        global_offset = np.min(self.bounding_boxes, axis=(0, 1))
        self.bounding_boxes -= global_offset

        slice_centers = np.linspace(0, self.w, number_of_panoramas + 2, endpoint=True, dtype=np.int)[1:-1]
        warped_slice_centers = np.zeros((number_of_panoramas, self.frames_for_panoramas.size))
        # every slice is a different panorama, it indicates the slices of the input images from which the panorama
        # will be concatenated
        for i in range(slice_centers.size):
            slice_center_2d = np.array([slice_centers[i], self.h // 2])[None, :]
            # homography warps the slice center to the coordinate system of the middle image
            warped_centers = [apply_homography(slice_center_2d, h) for h in self.homographies]
            # we are actually only interested in the x coordinate of each slice center in the panoramas' coordinate system
            warped_slice_centers[i] = np.array(warped_centers)[:, :, 0].squeeze() - global_offset[0]

        panorama_size = np.max(self.bounding_boxes, axis=(0, 1)).astype(np.int) + 1

        # boundary between input images in the panorama
        x_strip_boundary = ((warped_slice_centers[:, :-1] + warped_slice_centers[:, 1:]) / 2)
        x_strip_boundary = np.hstack([np.zeros((number_of_panoramas, 1)),
                                      x_strip_boundary,
                                      np.ones((number_of_panoramas, 1)) * panorama_size[0]])
        x_strip_boundary = x_strip_boundary.round().astype(np.int)

        self.panoramas = np.zeros((number_of_panoramas, panorama_size[1], panorama_size[0], 3), dtype=np.float64)
        for i, frame_index in enumerate(self.frames_for_panoramas):
            # warp every input image once, and populate all panoramas
            image = sol4_utils.read_image(self.files[frame_index], 2)
            warped_image = warp_image(image, self.homographies[i])
            x_offset, y_offset = self.bounding_boxes[i][0].astype(np.int)
            y_bottom = y_offset + warped_image.shape[0]

            for panorama_index in range(number_of_panoramas):
                # take strip of warped image and paste to current panorama
                boundaries = x_strip_boundary[panorama_index, i:i + 2]
                image_strip = warped_image[:, boundaries[0] - x_offset: boundaries[1] - x_offset]
                x_end = boundaries[0] + image_strip.shape[1]
                self.panoramas[panorama_index, y_offset:y_bottom, boundaries[0]:x_end] = image_strip

        # crop out areas not recorded from enough angles
        # assert will fail if there is overlap in field of view between the left most image and the right most image
        crop_left = int(self.bounding_boxes[0][1, 0])
        crop_right = int(self.bounding_boxes[-1][0, 0])
        assert crop_left < crop_right, 'for testing your code with a few images do not crop.'
        self.panoramas = self.panoramas[:, :, crop_left:crop_right, :]

    def generate_panoramic_images_bonus(self, number_of_panoramas):
        """
    The bonus
    :param number_of_panoramas: how many different slices to take from each input image
    """
        pass

    def save_panoramas_to_video(self):
        assert self.panoramas is not None
        out_folder = 'tmp_folder_for_panoramic_frames/%s' % self.file_prefix
        try:
            shutil.rmtree(out_folder)
        except:
            print('could not remove folder')
            pass
        os.makedirs(out_folder)
        # save individual panorama images to 'tmp_folder_for_panoramic_frames'
        for i, panorama in enumerate(self.panoramas):
            imwrite('%s/panorama%02d.png' % (out_folder, i + 1), panorama)
        if os.path.exists('%s.mp4' % self.file_prefix):
            os.remove('%s.mp4' % self.file_prefix)
        # write output video to current folder
        os.system('ffmpeg -framerate 3 -i %s/panorama%%02d.png %s.mp4' %
                  (out_folder, self.file_prefix))





"""


















"""

def main():
    is_bonus = False
    experiment = 'ENTER FILE ADDRESS HERE'
    exp_no_ext = experiment.split('.')[0]
    os.system('mkdir dump')
    os.system(('mkdir ' + str(os.path.join('dump', '%s'))) % exp_no_ext)
    os.system(('ffmpeg -i ' + str(os.path.join('videos', '%s ')) + str(os.path.join('dump', '%s', '%s%%03d.jpg'))) % (experiment, exp_no_ext, exp_no_ext))


    s = time.time()
    panorama_generator = PanoramicVideoGenerator(os.path.join('dump', '%s') % exp_no_ext, exp_no_ext, 2100,bonus=is_bonus)
    panorama_generator.align_images(translation_only=True)
    panorama_generator.generate_panoramic_images(9)
    print(' time for %s: %.1f' % (exp_no_ext, time.time() - s))
    panorama_generator.save_panoramas_to_video()


if __name__ == '__main__':
  main()