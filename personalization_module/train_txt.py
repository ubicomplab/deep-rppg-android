from __future__ import print_function

import argparse
import itertools
import json
import os
import datetime

import numpy as np
import scipy.io
import tensorflow as tf

from data_generator import DataGenerator
from model import TS_CAN
import glob
from utils import preprocess_raw_video

np.random.seed(100)  # for reproducibility
print(tf.__version__)
tf.test.is_gpu_available()

tf.keras.backend.clear_session()
# %%
parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('--video_path', type=str,
                    help='Location for video path')
parser.add_argument('--label_path', type=str,
                    help='Location for PPG label')
parser.add_argument('-img', '--img_size', type=int, default=36, help='img_size')
parser.add_argument('-crp_img', '--cropped_size', type=int, default=36, help='img_size')
parser.add_argument('-tr_data', '--tr_dataset', type=str, default='AFRL', help='training dataset name')
parser.add_argument('-ts_data', '--ts_dataset', type=str, default='AFRL', help='test dataset name')
parser.add_argument('-o', '--save_dir', type=str, default='./rPPG-checkpoints',
                    help='Location for parameter checkpoints and samples')
parser.add_argument('-a', '--nb_filters1', type=int, default=32,
                    help='number of convolutional filters to use')
parser.add_argument('-b', '--nb_filters2', type=int, default=64,
                    help='number of convolutional filters to use')
parser.add_argument('-c', '--dropout_rate1', type=float, default=0.25,
                    help='dropout rates')
parser.add_argument('-d', '--dropout_rate2', type=float, default=0.5,
                    help='dropout rates')
parser.add_argument('-l', '--lr', type=float, default=1.0,
                            help='learning rate')
parser.add_argument('-e', '--nb_dense', type=int, default=128,
                    help='number of dense units')
parser.add_argument('-f', '--cv_split', type=int, default=0,
                    help='cv_split')
parser.add_argument('-g', '--nb_epoch', type=int, default=8,
                    help='nb_epoch')
parser.add_argument('-t', '--nb_task', type=int, default=12,
                    help='nb_task')
parser.add_argument('-x', '--batch_size', type=int, default=8,
                    help='batch')
parser.add_argument('-fd', '--frame_depth', type=int, default=10,
                    help='frame_depth for 3DCNN')
parser.add_argument('-temp', '--temporal', type=str, default='TS_CAN_PEAKDETECTION',
                    help='3DCNN, 2DCNN or mix')
parser.add_argument('-save', '--save_all', type=int, default=1,
                    help='save all or not')
parser.add_argument('-resp', '--respiration', type=int, default=0,
                    help='train with resp or not')
parser.add_argument('-shuf', '--shuffle', type=str, default=True,
                    help='shuffle samples')
parser.add_argument('-da', '--data_aug', type=int, default=0,
                    help='data augmentation')
parser.add_argument('-ds', '--data_fs', type=int, default=30,
                    help='data frequency')
parser.add_argument('-dw', '--eval_window', type=int, default=360,
                    help='data frequency')


args = parser.parse_args()
print('input args:\n', json.dumps(vars(args), indent=4, separators=(',', ':')))  # pretty print arg

# %% Training


def train(args, img_rows, img_cols):
    print('================================')
    print('Train...')

    input_shape = (img_rows, img_cols, 3)

    # Reading Data
    if not os.path.exists('./chunked_data'):
        os.makedirs('./chunked_data')
    preprocess_raw_video(args.video_path, args.label_path, dim=36)

    path_of_video_tr = sorted(glob.glob('./chunked_data/*.mat'))

    print('sample path: ', path_of_video_tr[0])
    print('Trian Length: ', len(path_of_video_tr))

    print('Using TS-CAN')
    input_shape = (img_rows, img_cols, 3)
    model = TS_CAN(args.frame_depth, args.nb_filters1, args.nb_filters2, input_shape,
                         dropout_rate1=args.dropout_rate1, dropout_rate2=args.dropout_rate2, nb_dense=args.nb_dense)
    model.load_weights('./cv_0_epoch48_model.hdf5')

    optimizer = tf.keras.optimizers.Adam()
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    #%% Create data genener
    training_generator = DataGenerator(path_of_video_tr, 180, (args.img_size, args.img_size),
                                       batch_size=args.batch_size, frame_depth=args.frame_depth,
                                       temporal='TS_CAN', respiration=args.respiration, shuffle=args.shuffle)

    save_best_callback = tf.keras.callbacks.ModelCheckpoint(filepath="./personalized_model.hdf5",
                                                            save_best_only=False, verbose=1)
    #%% Model Training and Saving Results
    history = model.fit(x=training_generator, epochs=args.nb_epoch, verbose=1,
                        callbacks=[save_best_callback])

if __name__ == "__main__":
    train(args,img_rows=args.img_size, img_cols=args.img_size)



