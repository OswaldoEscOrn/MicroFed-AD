# Edge-Machine-Learning-Models/federated/models.py

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    UpSampling1D,
    Dense,
    Flatten,
    Reshape,
    BatchNormalization,
    Dropout,
    Layer,
    LSTM,
    RepeatVector,
    TimeDistributed,
)
from tensorflow.keras.optimizers import Adam


# ============================================================
# CONV-AE
# ============================================================

def build_conv_ae(input_shape=(24, 4), learning_rate=1e-3):
    inputs = Input(shape=input_shape)

    x = Conv1D(32, 3, activation="relu", padding="same")(inputs)
    x = MaxPooling1D(2, padding="same")(x)

    x = Conv1D(16, 3, activation="relu", padding="same")(x)
    encoded = MaxPooling1D(2, padding="same")(x)

    x = Conv1D(16, 3, activation="relu", padding="same")(encoded)
    x = UpSampling1D(2)(x)

    x = Conv1D(32, 3, activation="relu", padding="same")(x)
    x = UpSampling1D(2)(x)

    outputs = Conv1D(input_shape[1], 3, activation="linear", padding="same")(x)

    model = Model(inputs, outputs, name="conv_ae")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mae",
        metrics=["mae"],
    )
    return model


# ============================================================
# CONV-VAE
# ============================================================

class Sampling(Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class VAELossLayer(Layer):
    def __init__(self, kl_weight=0.001, **kwargs):
        super(VAELossLayer, self).__init__(**kwargs)
        self.kl_weight = kl_weight

    def call(self, inputs):
        x_true, x_pred, z_mean, z_log_var = inputs

        reconstruction_loss = K.mean(K.abs(x_true - x_pred))

        kl_loss = -0.5 * K.sum(
            1 + z_log_var - K.square(z_mean) - K.exp(z_log_var),
            axis=1,
        )
        kl_loss = K.mean(kl_loss) * self.kl_weight

        total_loss = reconstruction_loss + kl_loss
        self.add_loss(total_loss)

        return x_pred

    def compute_output_shape(self, input_shape):
        return input_shape[1]


def build_conv_vae(
    input_shape=(24, 4),
    learning_rate=1e-3,
    latent_dim=16,
    kl_weight=0.001,
    output_activation="sigmoid",
):
    timesteps, n_features = input_shape

    encoder_inputs = Input(shape=input_shape, name="encoder_input")

    x = Conv1D(32, kernel_size=5, activation="relu", padding="same")(encoder_inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2, padding="same")(x)
    x = Dropout(0.2)(x)

    x = Conv1D(64, kernel_size=3, activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2, padding="same")(x)
    x = Dropout(0.2)(x)

    x = Conv1D(128, kernel_size=3, activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2, padding="same")(x)

    x = Flatten()(x)
    x = Dense(64, activation="relu")(x)

    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])

    conv_shape = (timesteps // 8, 128)

    x = Dense(int(np.prod(conv_shape)), activation="relu")(z)
    x = Reshape(conv_shape)(x)

    x = Conv1D(128, kernel_size=3, activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = UpSampling1D(size=2)(x)

    x = Conv1D(64, kernel_size=3, activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = UpSampling1D(size=2)(x)

    x = Conv1D(32, kernel_size=3, activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = UpSampling1D(size=2)(x)

    decoder_outputs = Conv1D(
        n_features,
        kernel_size=5,
        activation=output_activation,
        padding="same",
    )(x)

    final_outputs = VAELossLayer(kl_weight=kl_weight)(
        [encoder_inputs, decoder_outputs, z_mean, z_log_var]
    )

    vae = Model(encoder_inputs, final_outputs, name="conv_vae")
    vae.compile(optimizer=Adam(learning_rate=learning_rate))
    return vae


# ============================================================
# DEEP-AE
# ============================================================

def build_deep_ae(
    input_shape=(96,),
    learning_rate=1e-3,
    encoding_dim=32,
):
    if isinstance(input_shape, tuple):
        input_dim = int(np.prod(input_shape))
    else:
        input_dim = int(input_shape)

    encoder = Sequential(
        [
            Input(shape=(input_dim,), name="input"),
            Dense(256, activation="relu", name="encoder_dense1"),
            BatchNormalization(name="encoder_bn1"),
            Dropout(0.3, name="encoder_dropout1"),

            Dense(128, activation="relu", name="encoder_dense2"),
            BatchNormalization(name="encoder_bn2"),
            Dropout(0.3, name="encoder_dropout2"),

            Dense(64, activation="relu", name="encoder_dense3"),
            BatchNormalization(name="encoder_bn3"),
            Dropout(0.2, name="encoder_dropout3"),

            Dense(encoding_dim, activation="relu", name="bottleneck"),
        ],
        name="encoder",
    )

    decoder = Sequential(
        [
            Input(shape=(encoding_dim,), name="decoder_input"),
            Dense(64, activation="relu", name="decoder_dense1"),
            BatchNormalization(name="decoder_bn1"),
            Dropout(0.2, name="decoder_dropout1"),

            Dense(128, activation="relu", name="decoder_dense2"),
            BatchNormalization(name="decoder_bn2"),
            Dropout(0.3, name="decoder_dropout2"),

            Dense(256, activation="relu", name="decoder_dense3"),
            BatchNormalization(name="decoder_bn3"),
            Dropout(0.3, name="decoder_dropout3"),

            Dense(input_dim, activation="linear", name="output"),
        ],
        name="decoder",
    )

    input_layer = Input(shape=(input_dim,), name="autoencoder_input")
    encoded = encoder(input_layer)
    decoded = decoder(encoded)

    autoencoder = Model(input_layer, decoded, name="deep_ae")

    autoencoder.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )

    return autoencoder


# ============================================================
# LSTM-AE
# ============================================================

def build_lstm_ae(
    input_shape=(24, 4),
    learning_rate=1e-3,
    lstm_units=128,
    encoding_dim=32,
):
    """
    LSTM AutoEncoder adapted from your LSTM-AE real / real2 scripts.

    Original:
    Encoder:
      LSTM(128, return_sequences=True)
      BN
      Dropout(0.2)
      LSTM(64, return_sequences=True)
      BN
      Dropout(0.2)
      LSTM(32, return_sequences=False)
      Dense(32, tanh)

    Decoder:
      RepeatVector(24)
      LSTM(32, return_sequences=True)
      BN
      Dropout(0.2)
      LSTM(64, return_sequences=True)
      BN
      Dropout(0.2)
      LSTM(128, return_sequences=True)
      BN
      TimeDistributed(Dense(4, linear))

    Loss: MAE
    """

    sequence_length, n_features = input_shape

    inputs = Input(shape=(sequence_length, n_features), name="lstm_input")

    encoded = LSTM(
        lstm_units,
        activation="tanh",
        return_sequences=True,
        name="encoder_lstm1",
    )(inputs)
    encoded = BatchNormalization(name="encoder_bn1")(encoded)
    encoded = Dropout(0.2, name="encoder_dropout1")(encoded)

    encoded = LSTM(
        lstm_units // 2,
        activation="tanh",
        return_sequences=True,
        name="encoder_lstm2",
    )(encoded)
    encoded = BatchNormalization(name="encoder_bn2")(encoded)
    encoded = Dropout(0.2, name="encoder_dropout2")(encoded)

    encoded = LSTM(
        lstm_units // 4,
        activation="tanh",
        return_sequences=False,
        name="encoder_lstm3",
    )(encoded)

    encoded = Dense(
        encoding_dim,
        activation="tanh",
        name="bottleneck",
    )(encoded)

    decoded = RepeatVector(sequence_length, name="repeat_vector")(encoded)

    decoded = LSTM(
        lstm_units // 4,
        activation="tanh",
        return_sequences=True,
        name="decoder_lstm1",
    )(decoded)
    decoded = BatchNormalization(name="decoder_bn1")(decoded)
    decoded = Dropout(0.2, name="decoder_dropout1")(decoded)

    decoded = LSTM(
        lstm_units // 2,
        activation="tanh",
        return_sequences=True,
        name="decoder_lstm2",
    )(decoded)
    decoded = BatchNormalization(name="decoder_bn2")(decoded)
    decoded = Dropout(0.2, name="decoder_dropout2")(decoded)

    decoded = LSTM(
        lstm_units,
        activation="tanh",
        return_sequences=True,
        name="decoder_lstm3",
    )(decoded)
    decoded = BatchNormalization(name="decoder_bn3")(decoded)

    outputs = TimeDistributed(
        Dense(n_features, activation="linear"),
        name="output",
    )(decoded)

    model = Model(inputs, outputs, name="lstm_ae")

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mae",
        metrics=["mse"],
    )

    return model


# ============================================================
# Unified model builder
# ============================================================

def build_model(
    model_name,
    input_shape=(24, 4),
    learning_rate=1e-3,
    latent_dim=16,
    kl_weight=0.001,
    vae_output_activation="sigmoid",
    deep_encoding_dim=32,
    lstm_units=128,
    lstm_encoding_dim=32,
):
    model_name = model_name.lower()

    if model_name in ["conv_ae", "convae", "conv-ae"]:
        return build_conv_ae(
            input_shape=input_shape,
            learning_rate=learning_rate,
        )

    if model_name in ["conv_vae", "convvae", "conv-vae"]:
        return build_conv_vae(
            input_shape=input_shape,
            learning_rate=learning_rate,
            latent_dim=latent_dim,
            kl_weight=kl_weight,
            output_activation=vae_output_activation,
        )

    if model_name in ["deep_ae", "deepae", "deep-ae"]:
        return build_deep_ae(
            input_shape=input_shape,
            learning_rate=learning_rate,
            encoding_dim=deep_encoding_dim,
        )

    if model_name in ["lstm_ae", "lstmae", "lstm-ae"]:
        return build_lstm_ae(
            input_shape=input_shape,
            learning_rate=learning_rate,
            lstm_units=lstm_units,
            encoding_dim=lstm_encoding_dim,
        )

    raise ValueError(f"Unsupported model_name: {model_name}")