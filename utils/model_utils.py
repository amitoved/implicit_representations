from tensorflow.keras import layers, models, initializers, regularizers, backend as K


def sine_activation(x):
    return K.sin(x)


def build_embedding_model(n_images, embedding_dim):
    initializer = initializers.RandomUniform(minval=0., maxval=1.)
    instance_idx = layers.Input(shape=(1,), name='embedding_input')
    embedded_vec = layers.Embedding(input_dim=n_images, output_dim=embedding_dim, input_length=n_images,
                                    embeddings_initializer=initializer, name='embedding_matrix',
                                    embeddings_regularizer=regularizers.L2(0.01))(instance_idx)
    embedded_vec = layers.Reshape((embedding_dim,))(embedded_vec)
    model = models.Model(inputs=instance_idx, outputs=embedded_vec, name='embedding_model')
    return model


def build_implicit_representation_model(n_layers, neurons_per_layer, embedding_dim):
    position_query = layers.Input(shape=(2,))
    instance_embedding = layers.Input(shape=(embedding_dim,))

    activation = sine_activation
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


def build_combined_model(n_layers, neurons_per_layer, embedding_dim, n_images):
    position_query = layers.Input(shape=(2,))
    instance_idx = layers.Input(shape=(1,))
    embedding_model = build_embedding_model(n_images, embedding_dim)
    implicit_representation_model = build_implicit_representation_model(n_layers, neurons_per_layer, embedding_dim)

    embedded_vec = embedding_model(instance_idx)
    rgb = implicit_representation_model([position_query, embedded_vec])
    model = models.Model(inputs=[position_query, instance_idx], outputs=rgb)
    return model


n_layers = 8
neurons_per_layer = 128
embedding_dim = 4
model = build_combined_model(n_layers, neurons_per_layer, embedding_dim, n_images)
