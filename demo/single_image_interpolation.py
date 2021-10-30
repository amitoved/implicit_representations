import argparse
import os

import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras import models

from demo import constants
from utils.image_utils import get_project_images
from utils.model_utils import generate_image_by_index


def image_interpolation(args):
    experiment_folder = os.path.join(constants.EXPERIMENTS_DIR, args.experiment_name)
    model_path = os.path.join(experiment_folder, 'model')
    model = models.load_model(model_path)

    images = get_project_images(np.max(constants.AVAILABLE_RESOLUTIONS))
    image_idx = args.image_idx
    if not image_idx:
        image_idx = np.random.randint(len(images))

    n_subplots = len(constants.AVAILABLE_RESOLUTIONS) + 1
    plt.figure(figsize=(2 * (n_subplots), 3))
    plt.suptitle('Single image interpolation\n', fontsize=20)
    for i, res in enumerate(constants.AVAILABLE_RESOLUTIONS):
        pred_img = generate_image_by_index(model, image_idx, res, res)
        plt.subplot(1, n_subplots, i + 1)
        plt.title(f'{res}x{res}', fontsize=14)
        plt.imshow(pred_img)
        plt.axis('off')
    plt.subplot(1, n_subplots, i + 2)
    plt.title(f'Ground truth', fontsize=14)
    plt.imshow(images[image_idx])
    plt.axis('off')

    plt.savefig(os.path.join(experiment_folder, f'image_interpolation_idx_{image_idx}'))
    if args.plot:
        plt.show()
    plt.close()


if __name__ == "__main__":
    config = dict()
    config["experiment_name"] = 'res_144_act_relu_loss_mse_embed_dim_4'
    config["image_idx"] = None
    config["plot"] = True

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", default=config["experiment_name"])
    parser.add_argument("--image-idx", default=config["image_idx"], type=int)
    parser.add_argument("--plot", default=config["plot"], type=bool)

    args = parser.parse_args()
    success = image_interpolation(args)
