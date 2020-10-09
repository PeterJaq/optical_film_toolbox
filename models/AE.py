import tensorflow as tf
# from tensorflow.keras.layers import Conv1D, Conv1DTranspose, layers
from tensorflow.keras.models import Model
from models.networks.autoencoder import FCN

import numpy as np 


class AE(tf.keras.Model):
    """Convolutional variational autoencoder."""

    def __init__(self, encoding_dim, input_shapes, input_length, netwok_type='FCN'):
        super(AE, self).__init__()
        self.input_shapes = input_shapes
        self.encoding_dim = encoding_dim
        self.input_length = input_length

        self.encoder, self.decoder = self.build_network(encoding_dim, netwok_type)
    
    def build_network(self, encoding_dim, network_type):
        if network_type == 'FCN':
            inputs, encoder, decoder = FCN(input_length=self.input_length, encoding_dim=self.encoding_dim, hidden_layers=20)

        encoder = Model(inputs=inputs, outputs=decoder)
        decoder = Model(inputs=inputs, outputs=decoder)
        return encoder, decoder 


if __name__ == "__main__":
  """
  test 
  """
  ae = AE(latent_dim=10, input_shapes=(200, 2), input_length=200)

  sample_n= np.random.rand(1, 200)
  sample_k = np.random.rand(1, 200)
  ae.compile(loss=['mae', 'mae'], optimizer=tf.keras.optimizers.Adam())

  hist = ae.encoder.fit(x=[sample_n, sample_k], y=[sample_n, sample_k], epochs=1000)
  print(hist.history)

  # encoder.fit()
