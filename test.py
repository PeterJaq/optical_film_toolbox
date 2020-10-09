from models.AE import AE
import tensorflow as tf 
import numpy as np

import matplotlib.pyplot as plt

if __name__ == "__main__":
  """
  test 
  """
  ae = AE(encoding_dim=10, input_shapes=(200, 2), input_length=200)

  x = [x for x in range(0, 200)]
  sample_n= np.random.rand(1, 200)
  sample_k = np.random.rand(1, 200)
  print(ae.encoder.summary())
  ae.encoder.compile(loss=['mae', 'mae'], optimizer=tf.keras.optimizers.Adam())

  hist = ae.encoder.fit(x=[sample_n, sample_k], y=[sample_n, sample_k], epochs=100)
  print(hist.history)

  result = ae.encoder.predict(x=[sample_n, sample_k])

  plt.plot(x, list(sample_n[0][:]))
  plt.plot(x, list(result[0][0][:]))
  plt.savefig('figure/sample_n.png')
