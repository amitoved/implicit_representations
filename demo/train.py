import argparse
import os
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras import optimizers, callbacks, models

from demo import constants
from utils.image_utils import get_project_images
from utils.model_utils import build_combined_model


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


def train_implicit_representation_model(args):
    training_images = get_project_images(args.training_resolution)
    validation_images = get_project_images(args.validation_resolution)

    n_images = len(training_images)
    identifiers = np.arange(n_images)

    training_data_generator = datagen(training_images, identifiers, args.batch_size)
    validation_data_generator = datagen(validation_images, identifiers, args.batch_size)

    experiment_folder = os.path.join(constants.EXPERIMENTS_DIR,
                                     f"res_{args.training_resolution}_act_{args.activation}_loss_{args.loss}_embed_dim_{args.embedding_dim}")
    os.makedirs(experiment_folder, exist_ok=True)
    model_path = os.path.join(experiment_folder, 'model')

    if Path(model_path).exists():
        model = models.load_model(model_path)
    else:
        model = build_combined_model(args.n_layers, args.neurons_per_layer, args.embedding_dim, n_images,
                                     args.activation)
    model.compile(loss=args.loss, optimizer=optimizers.Adam(lr=1e-3))
    checkpoint_callback = callbacks.ModelCheckpoint(filepath=model_path, save_weights_only=False, monitor='val_loss',
                                                    verbose=1, save_best_only=True)
    log = model.fit_generator(training_data_generator, steps_per_epoch=args.steps_per_epoch, epochs=args.epochs,
                              workers=1,
                              use_multiprocessing=False, callbacks=[checkpoint_callback],
                              validation_data=validation_data_generator, validation_steps=100)

    plt.figure(figsize=(10, 5))
    plt.plot(log.history['loss'], color='blue', lw=2)
    plt.plot(log.history['val_loss'], color='red', lw=2)
    plt.yscale('log')
    plt.title(f'Training loss ({args.loss})\n', fontsize=14)
    plt.xlabel('epoch', fontsize=10)
    plt.ylabel(args.loss, fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend({'loss', 'val_loss'}, fontsize=10)
    plt.savefig(os.path.join(experiment_folder, 'training_loss.png'))
    if args.plot:
        plt.show()
    plt.close()


if __name__ == "__main__":
    config = dict()
    config["n_layers"] = 8
    config["neurons_per_layer"] = 128
    config["activation"] = "relu"
    config["loss"] = "mse"
    config["embedding_dim"] = 4
    config["batch_size"] = 2 ** 16
    config["epochs"] = 30
    config["steps_per_epoch"] = 1000
    config["training_resolution"] = 144
    config["validation_resolution"] = 192
    config["plot"] = True

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-layers", default=config["n_layers"])
    parser.add_argument("--neurons-per-layer", default=config["neurons_per_layer"])
    parser.add_argument("--activation", default=config["activation"], choices=constants.SUPPORTED_ACTIVATIONS)
    parser.add_argument("--embedding-dim", default=config["embedding_dim"])
    parser.add_argument("--epochs", default=config["epochs"])
    parser.add_argument("-spe", "--steps-per-epoch", default=config["steps_per_epoch"])
    parser.add_argument("--training-resolution", default=config["training_resolution"],
                        choices=constants.AVAILABLE_RESOLUTIONS)
    parser.add_argument("--validation-resolution", default=config["validation_resolution"],
                        choices=constants.AVAILABLE_RESOLUTIONS)
    parser.add_argument("--batch-size", default=config["batch_size"], type=int)
    parser.add_argument("--loss", default=config["loss"], type=str)
    parser.add_argument("--plot", default=config["plot"], type=bool)

    args = parser.parse_args()
    train_implicit_representation_model(args)
