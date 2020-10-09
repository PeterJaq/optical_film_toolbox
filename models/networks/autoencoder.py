import tensorflow as tf 
from tensorflow.keras.layers import Input, Dense, Concatenate

def FCN(input_length, encoding_dim, hidden_layers=4):

    input_n = Input(shape=(1, input_length), name='n_in')
    input_k = Input(shape=(1, input_length), name='k_in')

    dense_n = Dense(256)(input_n)
    dense_k = Dense(256)(input_k)

    dense_concat = Concatenate()([dense_n, dense_k])

    dense_array = Dense(128)(dense_concat)
    for num in range(hidden_layers - 1):
        dense_array = Dense(128)(dense_array)

    encoder = Dense(encoding_dim)(dense_array)

    dense_array = Dense(128)(encoder)

    for num in range(hidden_layers - 1):
        dense_array = Dense(128)(dense_array)

    decoder_dense_n = Dense(256)(dense_array)
    decoder_dense_k =Dense(256)(dense_array)

    decoder_out_n = Dense(input_length, name='n_out')(decoder_dense_n)
    decoder_out_k = Dense(input_length, name='k_out')(decoder_dense_k)


    return [input_n, input_k], encoder,[decoder_out_n, decoder_out_k]    

def CNN(input_length, encoding_dim, hidden_layers=4):
    input_n = tf.keras.layers.Input(shape=(input_length, 1))
    input_k = tf.keras.layers.Input(shape=(input_length, 1))

    encoder_conv1d_n_1 = tf.keras.layers.Conv1D(
                    filters=64, kernel_size=3, strides=1, activation='relu')(input_n)
    encoder_conv1d_k_1 = tf.keras.layers.Conv1D(
                    filters=64, kernel_size=3, strides=1, activation='relu')(input_k)

    encoder_conv1d_n_2 = tf.keras.layers.Conv1D(
                    filters=32, kernel_size=3, strides=1, activation='relu')(encoder_conv1d_n_1)
    encoder_conv1d_k_2 = tf.keras.layers.Conv1D(
                    filters=32, kernel_size=3, strides=1, activation='relu')(encoder_conv1d_k_1)

    concat_nk = tf.keras.layers.concatenate([encoder_conv1d_n_2, encoder_conv1d_k_2])

    encoder = tf.keras.layers.Dense(encoding_dim)(concat_nk)

    decoder_conv1dtranspose_n_1 = tf.keras.layers.Conv1DTranspose(filters=64, 
                                    kernel_size=3, strides=1, padding='same', activation='relu')(encoder)
    decoder_conv1dtranspose_k_1 =  tf.keras.layers.Conv1DTranspose(filters=64, 
                                    kernel_size=3, strides=1, padding='same', activation='relu')(encoder)

    decoder_conv1dtranspose_n_2 = tf.keras.layers.Conv1DTranspose(filters=64, 
                                    kernel_size=3, strides=1, padding='same', activation='relu')(decoder_conv1dtranspose_n_1)
    decoder_conv1dtranspose_k_2 =  tf.keras.layers.Conv1DTranspose(filters=64, 
                                    kernel_size=3, strides=1, padding='same', activation='relu')(decoder_conv1dtranspose_k_1)

    decoder_n = tf.keras.layers.Dense(input_length)(decoder_conv1dtranspose_n_2)
    decoder_k = tf.keras.layers.Dense(input_length)(decoder_conv1dtranspose_k_2)
