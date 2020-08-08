import tensorflow as tf 
from tensorflow.keras.layers import Input, Dense, Conv1D, Concatenate

class autoEncoder:

    def __init__(self,
                input_shape,
                encoder_dims,
                activation = 'relu') -> None:

        self.input_shape = input_shape
        self.encoder_dims = encoder_dims 
        self.activation = activation

    def build_encoder(self):

        input_feature_1 = Input(shape=self.input_shape)
        input_feature_2 = Input(shape=self.input_shape)
        input_feature_3 = Input(sahpe=self.input_shape)

        encoded_1 = Dense(128, activation='relu')(input_feature_1)
        encoded_2 = Dense(128, activation='relu')(input_feature_2)
        encoded_3 = Dense(128, activation='relu')(input_feature_3)

        

        encoded = Dense(64, activation='relu')(encoded)
        encoded = Dense(32, activation='relu')(encoded)

        decoded = Dense(64, activation='relu')(encoded)
        decoded = Dense(128, activation='relu')(decoded)
        decoded = Dense(784, activation='sigmoid')(decoded)

        encoder_model = Model(input_feature, decoded)

        return encoder_model 

    def build_decoder(self):


        return decoder_model   