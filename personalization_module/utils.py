from collections import OrderedDict

import h5py
import numpy as np
import torch
from scipy.sparse import spdiags
import cv2
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import scipy

def preprocess_raw_video(videoFilePath, labelFilePath, dim=36, CHUNK_SIZE=180):

    #########################################################################
    # set up
    i = 0
    video_numpy = np.load(videoFilePath)
    label_numpy = np.load(labelFilePath)
    video_numpy = np.random.rand(2000, 36, 36, 3)
    label_numpy = np.random.rand(2000, 1)
    totalFrames = video_numpy.shape[0]
    Xsub = np.zeros((totalFrames, dim, dim, 3), dtype = np.float32)
    height = video_numpy.shape[1]
    width = video_numpy.shape[2]
    print("Orignal Height", height)
    print("Original width", width)
    #########################################################################
    # Crop each frame size into dim x dim
    for k in range(video_numpy.shape[0]):
        img = video_numpy[k]
        vidLxL = img #cv2.resize(img_as_float(img[:, int(width/2)-int(height/2 + 1):int(height/2)+int(width/2), :]), (dim, dim), interpolation = cv2.INTER_AREA)
        # vidLxL = cv2.rotate(vidLxL, cv2.ROTATE_90_CLOCKWISE) # rotate 90 degree
        # vidLxL = cv2.cvtColor(vidLxL.astype('float32'), cv2.COLOR_BGR2RGB)
        vidLxL[vidLxL > 1] = 1
        vidLxL[vidLxL < (1/255)] = 1/255
        Xsub[i, :, :, :] = vidLxL
        i = i + 1
    # plt.imshow(Xsub[0])
    # plt.title('Sample Preprocessed Frame')
    # plt.show()
    #########################################################################
    # Normalized Frames in the motion branch
    normalized_len = video_numpy.shape[0] - 1
    dXsub = np.zeros((normalized_len, dim, dim, 3), dtype = np.float32)
    for j in range(normalized_len - 1):
        dXsub[j, :, :, :] = (Xsub[j+1, :, :, :] - Xsub[j, :, :, :]) / (Xsub[j+1, :, :, :] + Xsub[j, :, :, :])
    dXsub = dXsub / np.std(dXsub)
    #########################################################################
    # Normalize raw frames in the apperance branch
    Xsub = Xsub - np.mean(Xsub)
    Xsub = Xsub  / np.std(Xsub)
    Xsub = Xsub[:totalFrames-1, :, :, :]
    #########################################################################
    # Plot an example of data after preprocess
    dXsub = np.concatenate((dXsub, Xsub), axis = 3);

    # Process Labels
    label_numpy = np.diff(label_numpy, axis=0)
    label_numpy = label_numpy / np.std(label_numpy)
    print(label_numpy.shape)
    print(dXsub.shape)
    count = 0
    for idx in range(0, label_numpy.shape[0], CHUNK_SIZE):
        end_idx = idx + CHUNK_SIZE
        if end_idx <= label_numpy.shape[0]:
            dXsub_chunk = dXsub[idx:end_idx]
            gt_chunk = label_numpy[idx:end_idx]
            save_path = './chunked_data/' + 'SelfCali_C' + str(count).zfill(2)
            scipy.io.savemat(save_path + '.mat', mdict={'dXsub': dXsub_chunk, 'dysub': gt_chunk})
            count += 1