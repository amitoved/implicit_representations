import argparse
import os

import imageio
import numpy as np

from implicit_representations import constants


def train_implicit_representation_model(args):
    # Import images for training
    available_resolutions = [24, 48, 96, 144, 192, 256, 512]
    assert args.training_resolution in available_resolutions, f'training_resolution must be one of the following: {available_resolutions}'
    training_images_folder = os.path.join(constants.DATA_DIR, str(args.training_resolution))
    training_images_filenames = os.listdir(training_images_folder)
    training_images_filenames = [os.path.join(training_images_folder, fn) for fn in training_images_filenames if
                                 os.path.isfile(os.path.join(training_images_folder, fn))]

    images = []
    for training_image_filename in training_images_filenames:
        image = imageio.imread(training_image_filename)
        images.append(image[..., :3] / 255)
    images = np.stack(images, axis=0)
    identifiers = np.arange(len(images))

    data_generator = datagen(images, identifiers, args.batch_size)

    from


def datagen(images, identifiers, batch_size):
    n_images, n_rows, n_cols, channels = images.shape
    n_pixels = n_rows * n_cols
    flattened_images = images.reshape([n_images, n_pixels, 3])
    x, y = np.meshgrid(np.linspace(-1, 1, n_cols), np.linspace(-1, 1, n_rows))
    pixel_positions = np.stack([x.flatten(), y.flatten()], axis=1)

    while True:
        sampled_image_indices = np.random.randint(0, n_images, size=batch_size)
        sampled_pixels = np.random.randint(n_pixels, size=batch_size)
        identifiers_ = identifiers[sampled_image_indices]
        pixel_positions_ = pixel_positions[sampled_pixels]
        rgb_values = flattened_images[sampled_image_indices, sampled_pixels]
        yield (pixel_positions_, identifiers_), rgb_values


if __name__ == "__main__":
    config = {}
    config["n_layers"] = 6
    config["neurons_per_layer"] = 128
    config["activation"] = "sine"
    config["batch_size"] = 2 ** 16
    config["training_resolution"] = 48

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-layers", default=config["n_layers"])
    parser.add_argument("--neurons-per-layer", default=config["neurons_per_layer"])
    parser.add_argument("--activation", default=config["activation"])
    parser.add_argument("--training-resolution", default=config["training_resolution"], type=int)
    parser.add_argument("--batch-size", default=config["batch_size"], type=int)

    args = parser.parse_args()
    success = train_implicit_representation_model(args)
