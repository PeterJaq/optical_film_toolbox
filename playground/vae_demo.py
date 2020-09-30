from operator import mod
import os 
import sys

sys.path.append('D:\Project\optical_film_toolbox_master\optical_film_toolbox')

import pandas as pd 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 
import plotly
from openTSNE import TSNE
from openTSNE.callbacks import ErrorLogger
from common.utils.tsne_utils import plot, MACOSKO_COLORS
# from models.AE import AE
# from models.CVAE import CVAE

from models.fcn_ae import FullyConnectedAutoEncoder
from common.refractive import RefractiveIndex

save_model = True
show_result = False 
show_tSNE = True 

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

path = 'D:\Project\optical_film_toolbox_master\optical_film_toolbox\config\material.txt'
material_li = []
with open(path, 'r') as f:
  for line in f.readlines():
      material_li.append(line.strip().replace(' ', '').split(','))

print(material_li)

database = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                 os.path.normpath("../db/"))
catalog = RefractiveIndex(database)

wavelength_start = 260
wavelength_end = 1000
wavelength_re = 10

input_data = {}

for material in material_li:
  material_set = material[0]
  material_name = material[1]
  material_author = material[2]

  material_n = []
  material_k = []
  print(material_name)
  mat = catalog.getMaterial(material_set, material_name, material_author)
  for i in range(wavelength_start, wavelength_end, wavelength_re):
    material_n.append(mat.getRefractiveIndex(i))
    try:
      material_k.append(mat.getExtinctionCoefficient(i))
    except:
      material_k.append(0)

  input_data[material_name] = np.concatenate((np.array(material_n), np.array(material_k)))

train_dataset = tf.data.Dataset.from_tensor_slices(list(input_data.values())).shuffle(1).batch(1)
test_dataset = tf.data.Dataset.from_tensor_slices(list(input_data.values())).shuffle(1).batch(1)

model = FullyConnectedAutoEncoder(latent_dim=10, input_shape=int(((1000-260)/10)*2))

optimizer = tf.keras.optimizers.Adam(1e-4)
global_step = tf.Variable(0)

num_epochs = 100
batch_size = 64
batch_losses = []

for epoch in range(num_epochs):
    for step,x in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            x_rec_logits = model(x)
            rec_loss =tf.losses.mean_squared_error(x,x_rec_logits)
            batch_losses.append(rec_loss)
            rec_loss =tf.reduce_mean(rec_loss)
        
        grads=tape.gradient(rec_loss,model.trainable_variables)
        optimizer.apply_gradients(zip(grads,model.trainable_variables))
        if step%100==0:
            print(epoch,step,float(rec_loss))


if show_result is True:
  for test in test_dataset:
    wl = [x for x in range(wavelength_start, wavelength_end, wavelength_re)]
    result = list(model(test))[0]
    midpoint = int(len(result)/2)

    print(len(result), midpoint)

    result_n = result[:midpoint]
    result_k = result[midpoint:]

    plt.plot(wl, result_n)
    plt.plot(wl, result_k)
    # plt.plot(wl, true_n)
    # plt.plot(wl, true_k)
    plt.savefig('D:/Project/optical_film_toolbox_master/optical_film_toolbox/figure/test.png')

if show_tSNE:
  tSNE_result = []

  tsne = TSNE(
    perplexity=30,
    metric="euclidean",
    n_jobs=4,
    random_state=42,
  )

  for test in test_dataset:
    result = list(model.encode(test))[0]
    tSNE_result.append(list(result))

  embedding_train = tsne.fit(np.array(tSNE_result))

  plot(embedding_train, list(input_data.keys()))
  plt.figure(figsize=(80, 80))
  plt.savefig('D:/Project/optical_film_toolbox_master/optical_film_toolbox/figure/tSNE.png')