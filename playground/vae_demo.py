from operator import mod
import os 
import sys


sys.path.append('/home/jizhidemifan/Project/optical_film_toolbox')

import pandas as pd 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 
import plotly
import plotly.graph_objects as go


from openTSNE import TSNE
from openTSNE.callbacks import ErrorLogger
from common.utils.tsne_utils import plot, MACOSKO_COLORS

from models.fcn_ae import FullyConnectedAutoEncoder
from refractivesqlite import dboperations as DB


save_model = True
show_result = True
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

rangeMin = 0.3
rangeMax = 1.0
re = 10

dbpath = "./db_sqlite/refractive.db"
db = DB.Database(dbpath)
matList = db.search_custom('select * from pages where shelf="main"')
matPageId = list()
for mat in matList:
  if mat[-3] < rangeMin and mat[-2] > rangeMax:
    print(mat)
    matPageId.append(mat[0])
    
print(matPageId)

input_data = {}

for matId in matPageId:
  mat = db.get_material(matId)
  material_name = mat.pageinfo['book'] + '_' + mat.pageinfo['page']
  material_n = []
  material_k = []

  for i in range(int(rangeMin*1000), int(rangeMax*1000), re):
    material_n.append(mat.get_refractiveindex(i))
    try:
      material_k.append(mat.get_extinctioncoefficient(i))
    except:
      material_k.append(0)

  input_data[material_name] = np.concatenate((np.array(material_n), np.array(material_k)))


num_epochs = 500
batch_size = 256
batch_losses = []


train_dataset = tf.data.Dataset.from_tensor_slices(list(input_data.values())).shuffle(1).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices({'name': list(input_data.keys()),
                                                   'value':list(input_data.values())}).batch(1)

model = FullyConnectedAutoEncoder(latent_dim=10, input_shape=int(((1000-300)/10)*2))

optimizer = tf.keras.optimizers.Adam(1e-4)
global_step = tf.Variable(0)

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
            print(epoch, step, float(rec_loss))


if show_result:
  for test in test_dataset:
    try:
      wl = [x for x in range(int(rangeMin*1000), int(rangeMax*1000), re)]
      name = str(test['name'].numpy().astype(str)[0])

      result_hat = list(model(test['value']))[0]
      result = list(test['value'])[0]
      midpoint = int(len(result)/2)
      result_n = result[:midpoint]
      result_k = result[midpoint:]

      result_hat_n = result_hat[:midpoint]
      result_hat_k = result_hat[midpoint:]

      plt.plot(wl, result_n, label=name+'_n')
      plt.plot(wl, result_k, label=name+'_k')

      plt.plot(wl, result_hat_n, label=name+'_n_hat')
      plt.plot(wl, result_hat_k, label=name+'_k_hat')
      plt.legend(loc='upper right')
      plt.savefig(f"/home/jizhidemifan/Project/optical_film_toolbox/figure/sample_ae/{name}_.png")
      plt.clf()

    except Exception as e:
      plt.clf()
      print(e)
      print(f"save {name} figure fail!")

if show_tSNE:
  tSNE_result = []

  tsne = TSNE(
    perplexity=8,
    metric="euclidean",
    n_jobs=4,
    random_state=42,
  )


  for test in test_dataset:
    # print(test)
    name = test['name']
    result = list(model.encode(test['value']))[0]
    tSNE_result.append(result)

  embedding_train = tsne.fit(np.array(tSNE_result))
  # print(embedding_train)
  # plot(embedding_train, list(input_data.keys()))
  # plt.savefig('/home/jizhidemifan/Project/optical_film_toolbox/figure/tSNE.png')

fig = go.Figure(data=go.Scatter(
                                x=embedding_train[:][0],
                                y=embedding_train[:][1],
                                text=list(input_data.keys())
))
fig.write_image("figure/plotly.png")
