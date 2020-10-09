import tensorflow as tf
import numpy as np 
import matplotlib.pylab as plt
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, EarlyStopping, TerminateOnNaN, LambdaCallback
import os 

from common import MaterialLoader
from models.AE import AE
from sklearn import preprocessing


materials = MaterialLoader()

train_dataset_n = []
train_dataset_k = []

wavelength_start = 300
wavelength_end = 1200
WLstep = 2
log_dir = 'logs/'

for sample in materials.load_total_material_generator():

    wl = sample[1][0]
    n = sample[1][1]
    k = sample[1][2]

    if wl[0] <= wavelength_start and wl[-1] >= wavelength_end:
        print(sample[0],len(n), len(k))
        n_tmp = []
        k_tmp = []
        for idx, wl in enumerate(wl):
            if wl >= wavelength_start and wl < wavelength_end:
                if idx % WLstep == 0:
                    n_tmp.append(n[idx])
                    k_tmp.append(k[idx])
        train_dataset_n.append(n_tmp)
        train_dataset_k.append(k_tmp)

min_max_scaler = preprocessing.MinMaxScaler((0, 1))
train_dataset_n = min_max_scaler.fit_transform(train_dataset_n)
train_dataset_k = min_max_scaler.fit_transform(train_dataset_k)

nan_pos = np.isnan(train_dataset_n)
train_dataset_n = train_dataset_n[~nan_pos.any(axis=1)]
train_dataset_k = train_dataset_k[~nan_pos.any(axis=1)]

train_dataset_input = tf.data.Dataset.from_tensor_slices({'n_in': train_dataset_n,'k_in': train_dataset_k})
train_dataset_output = tf.data.Dataset.from_tensor_slices({'n_out': train_dataset_n,'k_out': train_dataset_k})

train_dataset = tf.data.Dataset.zip((train_dataset_input, train_dataset_output))
# train_dataset = tf.data.Dataset.from_tensor_slices((train_dataset, train_dataset))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(32)
# print(list(train_dataset.as_numpy_iterator()))

logging = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False, write_grads=False, write_images=False, update_freq='batch')
checkpoint = ModelCheckpoint(os.path.join(log_dir, 'ep{epoch:03d}-loss{loss:.3f}.h5'),
    monitor='loss',
    verbose=1,
    save_weights_only=False,
    save_best_only=True,
    period=1)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, verbose=1, cooldown=0, min_lr=1e-10)
early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=1)
terminate_on_nan = TerminateOnNaN()

callbacks=[logging, checkpoint, reduce_lr, early_stopping, terminate_on_nan]

ae = AE(encoding_dim=10, 
        input_shapes=((wavelength_end-wavelength_start)/WLstep, 2), 
        input_length=int((wavelength_end-wavelength_start)/WLstep))

ae.encoder.compile(loss=['mae', 'mae'], optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

hist = ae.encoder.fit(train_dataset, epochs=100, callbacks=callbacks)

# print(hist.history().values())