# Edge-Machine-Learning-Models/federated/client.py

import copy
import tensorflow as tf
from models import build_model


class FederatedClient:
    def __init__(
        self,
        client_id,
        X_train,
        y_train,
        model_name="conv_ae",
        input_shape=(24, 4),
        lr=1e-3,
        latent_dim=16,
        kl_weight=0.001,
        vae_output_activation="sigmoid",
        deep_encoding_dim=32,
        lstm_units=128,
        lstm_encoding_dim=32,
    ):
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train
        self.model_name = model_name
        self.input_shape = input_shape
        self.lr = lr
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.vae_output_activation = vae_output_activation
        self.deep_encoding_dim = deep_encoding_dim
        self.lstm_units = lstm_units
        self.lstm_encoding_dim = lstm_encoding_dim

    def _build(self):
        return build_model(
            model_name=self.model_name,
            input_shape=self.input_shape,
            learning_rate=self.lr,
            latent_dim=self.latent_dim,
            kl_weight=self.kl_weight,
            vae_output_activation=self.vae_output_activation,
            deep_encoding_dim=self.deep_encoding_dim,
            lstm_units=self.lstm_units,
            lstm_encoding_dim=self.lstm_encoding_dim,
        )

    def fit_fedavg(self, global_weights, local_epochs=1, batch_size=128):
        model = self._build()
        model.set_weights(copy.deepcopy(global_weights))

        model.fit(
            self.X_train,
            self.X_train,
            epochs=local_epochs,
            batch_size=batch_size,
            verbose=0,
            shuffle=True,
        )

        return model.get_weights(), len(self.X_train)

    def fit_fedprox(self, global_weights, local_epochs=1, batch_size=128, mu=0.01):
        model = self._build()
        model.set_weights(copy.deepcopy(global_weights))

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        global_weights_tf = [tf.convert_to_tensor(w) for w in global_weights]

        dataset = (
            tf.data.Dataset.from_tensor_slices((self.X_train, self.X_train))
            .shuffle(len(self.X_train))
            .batch(batch_size)
        )

        for _ in range(local_epochs):
            for xb, yb in dataset:
                with tf.GradientTape() as tape:
                    preds = model(xb, training=True)

                    model_name = self.model_name.lower()

                    if model_name in ["deep_ae", "deepae", "deep-ae"]:
                        base_loss = tf.reduce_mean(tf.square(yb - preds))

                    elif model_name in ["conv_ae", "convae", "conv-ae"]:
                        base_loss = tf.reduce_mean(tf.abs(yb - preds))

                    elif model_name in ["lstm_ae", "lstmae", "lstm-ae"]:
                        base_loss = tf.reduce_mean(tf.abs(yb - preds))

                    elif model_name in ["conv_vae", "convvae", "conv-vae"]:
                        if len(model.losses) > 0:
                            base_loss = tf.add_n(model.losses)
                        else:
                            base_loss = tf.reduce_mean(tf.abs(yb - preds))

                    else:
                        base_loss = tf.reduce_mean(tf.abs(yb - preds))

                    prox_term = tf.add_n(
                        [
                            tf.reduce_sum(tf.square(w - gw))
                            for w, gw in zip(model.trainable_weights, global_weights_tf)
                        ]
                    )

                    loss = base_loss + (mu / 2.0) * prox_term

                grads = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

        return model.get_weights(), len(self.X_train)
