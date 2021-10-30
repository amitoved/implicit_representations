import os

import imageio
import numpy as np

from demo import constants


def get_project_images(resolution, exclude_alpha=True):
    available_resolutions = constants.AVAILABLE_RESOLUTIONS
    assert resolution in available_resolutions, f'training_resolution must be one of the following: {available_resolutions}'
    training_images_folder = os.path.join(constants.DATA_DIR, str(resolution))
    training_images_filenames = os.listdir(training_images_folder)
    full_training_images_filenames = [os.path.join(training_images_folder, fn) for fn in training_images_filenames if
                                      os.path.isfile(os.path.join(training_images_folder, fn))]
    images = []
    for training_image_filename in full_training_images_filenames:
        image = imageio.imread(training_image_filename)
        # The following is a result of varying background values in the data
        foreground = np.stack([image[..., -1]] * 3, axis=2)
        background = 1 - foreground
        image = foreground * image[..., :3] + background * 255
        if not exclude_alpha:
            image = np.concatenate([image, foreground[..., 0:1]], axis=-1)
        images.append(image / 255)
    images = np.stack(images, axis=0)
    return images
