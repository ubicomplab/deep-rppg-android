'''
Data Generator for Multi-Task Temporal Shift Attention Networks for On-Device Contactless Vitals Measurement
Author: Xin Liu
'''

import math

import h5py
import numpy as np
from tensorflow import keras
import scipy.io

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, paths_of_videos, nframe_per_video, dim, batch_size=32, frame_depth=10,
                 shuffle=True, temporal=True, respiration=0):
        self.dim = dim
        self.batch_size = batch_size
        self.paths_of_videos = paths_of_videos
        self.nframe_per_video = nframe_per_video
        self.shuffle = shuffle
        self.temporal = temporal
        self.frame_depth = frame_depth
        self.respiration = respiration
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return math.ceil(len(self.paths_of_videos) / self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.paths_of_videos[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        # 'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.paths_of_videos))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_video_temp):
        'Generates data containing batch_size samples'

        if self.respiration == 1:
            label_key = "drsub"
        else:
            label_key = 'dysub'

        if self.temporal == 'TS_CAN':
            data = np.zeros((self.nframe_per_video * len(list_video_temp), self.dim[0], self.dim[1], 6), dtype=np.float32)
            label = np.zeros((self.nframe_per_video * len(list_video_temp), 1), dtype=np.float32)
            num_window = int(self.nframe_per_video / self.frame_depth) * len(list_video_temp)
            for index, temp_path in enumerate(list_video_temp):
                f1 = scipy.io.loadmat(temp_path)
                dXsub = np.transpose(np.array(f1["dXsub"]), [0, 1, 2, 3])
                dysub = np.array(f1[label_key])
                data[index*self.nframe_per_video:(index+1)*self.nframe_per_video, :, :, :] = dXsub
                label[index*self.nframe_per_video:(index+1)*self.nframe_per_video, :] = dysub
            motion_data = data[:, :, :, :3]
            apperance_data = data[:, :, :, -3:]
            apperance_data = np.reshape(apperance_data, (num_window, self.frame_depth, self.dim[0], self.dim[1], 3))
            apperance_data = np.average(apperance_data, axis=1)
            apperance_data = np.repeat(apperance_data[:, np.newaxis, :, :, :], self.frame_depth, axis=1)
            apperance_data = np.reshape(apperance_data, (apperance_data.shape[0] * apperance_data.shape[1],
                                                         apperance_data.shape[2], apperance_data.shape[3],
                                                         apperance_data.shape[4]))
            output = (motion_data, apperance_data)
        else:
            raise ValueError('Unsupported Model!')

        return output, label
