import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Conv2D, Conv3D, Input, AveragePooling2D, \
    multiply, Dense, Dropout, Flatten, AveragePooling3D, LSTM, Concatenate, TimeDistributed, Lambda
from tensorflow.python.keras.models import Model

class Attention_mask(tf.keras.layers.Layer):
    def call(self, x):
        xsum = K.sum(x, axis=1, keepdims=True)
        xsum = K.sum(xsum, axis=2, keepdims=True)
        xshape = K.int_shape(x)
        return x / xsum * xshape[1] * xshape[2] * 0.5

    def get_config(self):
        config = super(Attention_mask, self).get_config()
        return config


class TSM(tf.keras.layers.Layer):
    def call(self, x, n_frame, fold_div=3):
        nt, h, w, c = x.shape
        x = K.reshape(x, (-1, n_frame, h, w, c))
        fold = c // fold_div
        last_fold = c - (fold_div-1)*fold
        out1, out2, out3 = tf.split(x, [fold, fold, last_fold], axis=-1)

        # Shift left
        padding_1 = tf.zeros_like(out1)
        padding_1 = padding_1[:,-1,:,:,:]
        padding_1 = tf.expand_dims(padding_1, 1)
        _, out1 = tf.split(out1, [1, n_frame-1], axis=1)
        out1 = tf.concat([out1, padding_1], axis=1)

        # Shift right
        padding_2 = tf.zeros_like(out2)
        padding_2 = padding_2[:,0,:,:, :]
        padding_2 = tf.expand_dims(padding_2, 1)
        out2, _ = tf.split(out2, [n_frame-1,1], axis=1)
        out2 = tf.concat([padding_2, out2], axis=1)

        out = tf.concat([out1, out2, out3], axis=-1)
        out = K.reshape(out, (-1, h, w, c))

        return out

    def get_config(self):
        config = super(TSM, self).get_config()
        return config


def TSM_Cov2D(x, n_frame, nb_filters=128, kernel_size=(3, 3), activation='tanh', padding='same'):
    x = TSM()(x, n_frame)
    x = Conv2D(nb_filters, kernel_size, padding=padding, activation=activation)(x)
    return x


def TS_CAN_single(n_frame, nb_filters1, nb_filters2, input_shape, kernel_size=(3, 3), dropout_rate1=0.25, dropout_rate2=0.5,
           pool_size=(2, 2), nb_dense=128):

    diff_input = Input(shape=input_shape, batch_size=20)
    d1 = TSM_Cov2D(diff_input, n_frame, nb_filters1, kernel_size, padding='same', activation='tanh')
    d2 = TSM_Cov2D(d1, n_frame, nb_filters1, kernel_size, padding='valid', activation='tanh')

    g1 = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(d2)
    g1 = Attention_mask()(g1)
    gated1 = multiply([d2, g1])

    d3 = AveragePooling2D(pool_size)(gated1)
    d4 = Dropout(dropout_rate1)(d3)

    d5 = TSM_Cov2D(d4, n_frame, nb_filters2, kernel_size, padding='same', activation='tanh')
    d6 = TSM_Cov2D(d5, n_frame, nb_filters2, kernel_size, padding='valid', activation='tanh')

    g2 = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(d6)
    g2 = Attention_mask()(g2)
    gated2 = multiply([d6, g2])

    d7 = AveragePooling2D(pool_size)(gated2)
    d8 = Dropout(dropout_rate1)(d7)

    d9 = Flatten()(d8)
    d10 = Dense(nb_dense, activation='tanh')(d9)
    d11 = Dropout(dropout_rate2)(d10)
    out = Dense(1)(d11)
    model = Model(inputs=diff_input, outputs=out)

    return model


input_shape = (36, 36, 3)
model = TS_CAN_single(10, 32, 64, input_shape, dropout_rate1=0.25, dropout_rate2=0.5, nb_dense=128)
model.load_weights('./ts_can_single_att.hdf5')


