from model import TS_CAN_BATCH, TSM, TSM_Cov2D, Attention_mask
import tensorflow as tf
from tensorflow.python.keras import backend as K
import os
import sys

def convert_to_tflite():
    img_rows = 36
    img_cols = 36
    frame_depth = 10 #NOTE: this was 10
    model_checkpoint = 'personalized_model.hdf5'

    model = TS_CAN_BATCH(frame_depth, 32, 64, (img_rows, img_cols, 3))
    model.load_weights(model_checkpoint)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # save the model
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)
    return

if __name__ == "__main__":
    convert_to_tflite()

