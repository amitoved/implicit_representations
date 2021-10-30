import argparse
import os

import imageio
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE
from tensorflow.keras import models
from tqdm import tqdm

from demo import constants
from utils.image_utils import get_project_images
from utils.model_utils import generate_image_by_vector


def interpolation_of_pairs(args):
    experiment_folder = os.path.join(constants.EXPERIMENTS_DIR, args.experiment_name)
    model_path = os.path.join(experiment_folder, 'model')
    model = models.load_model(model_path)

    images = get_project_images(args.evaluation_resolution, exclude_alpha=False)
    n_images = len(images)

    implicit_representation_model = model.get_layer('implicit_representation_model')
    embedding_model = model.get_layer('embedding_model')
    embedding_matrix = embedding_model.get_layer('embedding_matrix').get_weights()[0]

    tsne = TSNE().fit_transform(embedding_matrix)
    tx, ty = tsne[:, 0], tsne[:, 1]
    tx = (tx - tx.min()) / (tx.max() - tx.min())
    ty = (ty - ty.min()) / (ty.max() - ty.min())

    width = 800
    height = 600
    max_dim = 48
    tsne_image = Image.new('RGBA', (width, height))
    for idx, x in enumerate(images):
        tile = Image.fromarray(np.uint8(x * 255))
        rs = max(1, tile.width / max_dim, tile.height / max_dim)
        tile = tile.resize((int(tile.width / rs), int(tile.height / rs)), Image.ANTIALIAS)
        tsne_image.paste(tile, (int((width - max_dim) * tx[idx]),
                                int((height - max_dim) * ty[idx])))
    tsne_image.save(os.path.join(experiment_folder, 'tsne.png'))
    if args.plot:
        plt.imshow(tsne_image)
        plt.title('T-SNE of the embeddings', fontsize=20)
        plt.axis('off')
        plt.show()

    dist_matrix = cdist(embedding_matrix, embedding_matrix, metric='euclidean')
    dist_matrix = np.triu(dist_matrix, k=0)
    max_val = dist_matrix.max()
    dist_matrix[dist_matrix == 0] = max_val
    plt.figure(figsize=(5, 5))
    plt.imshow(dist_matrix, cmap='magma')
    plt.title('Euclidean distance matrix of the embedded vectors\n', fontsize=12)
    plt.savefig(os.path.join(experiment_folder, 'distance_matrix.png'))
    if args.plot:
        plt.show()
    plt.close()

    for count in range(3):
        i, j = np.unravel_index(np.argmin(dist_matrix), [n_images, n_images])
        print(i, j)
        dist_matrix[i, j] = max_val
        imgs = []
        for phase in tqdm(np.linspace(0, 2 * np.pi, args.interpolation_steps)):
            beta = 0.5 * np.sin(phase) + 0.5
            embedding_vec = beta * embedding_matrix[i] + (1 - beta) * embedding_matrix[j]
            pred_img = generate_image_by_vector(implicit_representation_model, embedding_vec,
                                                args.evaluation_resolution,
                                                args.evaluation_resolution)
            imgs.append(pred_img)

        gif_duration = 3  # [sec]
        # frame_duration = 1000 * (gif_duration / args.interpolation_steps)  # [msec]
        gif_path = os.path.join(experiment_folder, f'blending_{i}_{j}.gif')
        imageio.mimsave(gif_path, imgs, fps=args.interpolation_steps / gif_duration)


if __name__ == "__main__":
    config = dict()
    config["experiment_name"] = 'res_144_act_relu_loss_mse_embed_dim_4'
    config["evaluation_resolution"] = 48
    config["interpolation_steps"] = 40
    config["plot"] = True

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", default=config["experiment_name"])
    parser.add_argument("--evaluation-resolution", default=config["evaluation_resolution"], type=int)
    parser.add_argument("--interpolation-steps", default=config["interpolation_steps"], type=int)
    parser.add_argument("--plot", default=config["plot"], type=bool)

    args = parser.parse_args()
    success = interpolation_of_pairs(args)
