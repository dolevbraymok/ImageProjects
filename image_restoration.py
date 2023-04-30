import numpy as np
from scipy.spatial import cKDTree


def create_mask(image_shape, mask_shape, mask_center):
    """
    Creates a binary mask of shape `mask_shape` centered at `mask_center` within
    an image of shape `image_shape`.
    """
    mask = np.zeros(image_shape[:2], dtype=np.bool)
    y, x = mask_center
    h, w = mask_shape
    mask[y - h // 2:y - h // 2 + h, x - w // 2:x - w // 2 + w] = True
    return mask


def get_patch(image, center, patch_size):
    """
    Returns a square patch of size `patch_size` centered at `center` within an image.
    """
    y, x = center
    r = patch_size // 2
    return image[y - r:y - r + patch_size, x - r:x - r + patch_size]


def find_best_match(patch, patches, kdtree):
    """
    Finds the patch in `patches` that is closest to `patch` using a KD-tree for efficient
    nearest neighbor search.
    """
    distances, indices = kdtree.query(patch.ravel(), k=1)
    return patches[indices]


def exemplar_inpainting(image, mask, patch_size, alpha=1.0):
    """
    Performs Exemplar-Based Inpainting on an `image` using a binary `mask` and square patches
    of size `patch_size`. The `alpha` parameter controls the weight given to the source image
    and the patch from the exemplar image.
    """
    # Create a copy of the image and mask to hold the inpainted result
    result = np.copy(image)
    result_mask = np.copy(mask)

    # Compute the KD-tree for the patches in the image
    patches = np.array([get_patch(image, (y, x), patch_size) for y in range(patch_size, image.shape[0] - patch_size)
                        for x in range(patch_size, image.shape[1] - patch_size)])
    kdtree = cKDTree(patches.reshape(-1, patch_size * patch_size))

    # Loop over the pixels in the mask and fill them in
    mask_y, mask_x = np.where(mask)
    for y, x in zip(mask_y, mask_x):
        # Get the patch surrounding the masked pixel
        patch = get_patch(result, (y, x), patch_size)

        # Find the best matching patch in the image
        best_match = find_best_match(patch, patches, kdtree)

        # Blend the patch from the image with the source patch from the exemplar image
        result[y, x] = alpha * best_match[y - patch_size // 2, x - patch_size // 2] + (1 - alpha) * patch[
            patch_size // 2, patch_size // 2]

        # Mark the pixel as filled in the result mask
        result_mask[y, x] = False

    return result, result_mask
