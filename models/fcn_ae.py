import tensorflow as tf
from tensorflow.keras import layers


class FullyConnectedAutoEncoder(tf.keras.Model):
    def __init__(self, latent_dim, input_shape):
        super(FullyConnectedAutoEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.input_length = input_shape
        self.inference_net = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(self.input_length, 1)),
            tf.keras.layers.Dense(256),
            tf.keras.layers.Dense(128),
            tf.keras.layers.Flatten(),
            # No activation
            tf.keras.layers.Dense(latent_dim),
        ]
        )

        self.generative_net = tf.keras.Sequential(
            [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(128),
            tf.keras.layers.Dense(256),
            tf.keras.layers.Dense(self.input_length)]
        )        

    def call(self, inputs, training=None):
        h = self.inference_net(inputs)
        x_hat = self.generative_net(h)
        return x_hat

    def encode(self, inputs):
        return self.inference_net(inputs)

# @tf.function
# def compute_loss(model, x):
#     x_hat = model.decode(x)
#     return tf.losses.mean_squared_error(x, x_bar)

# @tf.function
# def compute_apply_gradients(model, x, optimizer):
#     with tf.GradientTape() as tape:
#         loss = compute_loss(model, x)
#     gradients = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# def grad(model, inputs):
#     with tf.GradientTape() as tape:
#         reconstruction, inputs_reshaped = model(inputs)
#         loss_value = loss(inputs_reshaped, reconstruction)
#     return loss_value, tape.gradient(loss_value, model.trainable_variables), inputs_reshaped, reconstruction