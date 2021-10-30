import numpy as np
from tensorflow.keras import layers, models, initializers, regularizers, backend as K

from demo import constants


def sine(x):
    return K.sin(x)


def build_embedding_model(n_images, embedding_dim):
    initializer = initializers.RandomUniform(minval=0., maxval=1.)
    instance_idx = layers.Input(shape=(1,), name='embedding_input')
    embedded_vec = layers.Embedding(input_dim=n_images, output_dim=embedding_dim, input_length=n_images,
                                    embeddings_initializer=initializer, name='embedding_matrix',
                                    embeddings_regularizer=regularizers.L2(0.))(instance_idx)
    embedded_vec = layers.Reshape((embedding_dim,))(embedded_vec)
    model = models.Model(inputs=instance_idx, outputs=embedded_vec, name='embedding_model')
    return model


def build_implicit_representation_model(n_layers, neurons_per_layer, embedding_dim, activation):
    assert activation in constants.SUPPORTED_ACTIVATIONS, f'only {constants.SUPPORTED_ACTIVATIONS} are supported'
    if activation == 'sine':
        activation = sine
    position_query = layers.Input(shape=(2,))
    instance_embedding = layers.Input(shape=(embedding_dim,))

    x = layers.concatenate([position_query, instance_embedding])
    x = layers.Dense(neurons_per_layer, activation=activation)(x)
    for i in range(n_layers - 1):
        x = layers.Dense(neurons_per_layer, activation=activation)(x)
    pred_rgb = layers.Dense(3)(x)
    model = models.Model(inputs=[position_query, instance_embedding], outputs=pred_rgb,
                         name='implicit_representation_model')

    for layer in model.layers:
        if 'dense' in layer.name:
            w, b = layer.get_weights()
            w = (2 * np.random.rand(*w.shape) - 1.0) * np.sqrt(6 / neurons_per_layer)
            layer.set_weights([w, b])
    return model


def build_combined_model(n_layers, neurons_per_layer, embedding_dim, n_images, activation):
    position_query = layers.Input(shape=(2,))
    instance_idx = layers.Input(shape=(1,))
    embedding_model = build_embedding_model(n_images, embedding_dim)
    implicit_representation_model = build_implicit_representation_model(n_layers, neurons_per_layer, embedding_dim,
                                                                        activation)

    embedded_vec = embedding_model(instance_idx)
    rgb = implicit_representation_model([position_query, embedded_vec])
    model = models.Model(inputs=[position_query, instance_idx], outputs=rgb)
    return model


def generate_image_by_index(model, idx, rows, cols, postprocess=True):
    x, y = np.meshgrid(np.linspace(-1, 1, cols), np.linspace(-1, 1, rows))
    pixel_positions = np.stack([x.flatten(), y.flatten()], axis=1)
    n_pixels = rows * cols
    pred_img = model.predict([pixel_positions, int(idx) * np.ones((n_pixels), dtype=int)]).reshape([rows, cols, 3])
    if postprocess:
        pred_img = pred_img - pred_img.min()
        pred_img = pred_img / (pred_img.max() / 255 + 1e-9)
        pred_img = pred_img.astype(np.uint8())
    return pred_img


def generate_image_by_vector(implicit_representation_model, embedding_vec, rows, cols, postprocess=True):
    x, y = np.meshgrid(np.linspace(-1, 1, cols), np.linspace(-1, 1, rows))
    pixel_positions = np.stack([x.flatten(), y.flatten()], axis=1)
    n_pixels = rows * cols
    pred_img = implicit_representation_model.predict(
        x=[pixel_positions, embedding_vec * np.ones((n_pixels, 1))]).reshape([rows, cols, 3])

    if postprocess:
        pred_img = pred_img - pred_img.min()
        pred_img = pred_img / (pred_img.max() / 255 + 1e-9)
        pred_img = pred_img.astype(np.uint8())
    return pred_img
