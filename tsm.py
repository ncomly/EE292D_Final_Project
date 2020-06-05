# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# import keras
# from keras.models import Model

import tensorflow as tf
from tensorflow.keras.models import Model

import numpy as np

class TemporalShift(Model):
    def __init__(self, net, n_segment=3, n_div=8, inplace=False):
        super(TemporalShift, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace

        if inplace:
            print('=> Using in-place shift...')
        print('=> Using fold div: {}'.format(self.fold_div))


    def call(self, x):

        size = x.shape
        x = tf.reshape(x, [-1, 29, size[1], size[2], size[3]])

        # split along channel dimension (1/fold_div shifted right,
        # 1/fold_div shifted left, rest not shifted)
        folds = tf.split(x, num_or_size_splits=self.fold_div, axis=4)
        # shift right
        shift_right = tf.concat([tf.zeros([x.shape[0], 1, x.shape[2], x.shape[3], x.shape[4]//self.fold_div], dtype=tf.float32), 
            x[:, 1:, :, :, :x.shape[4]//self.fold_div]], axis=1)
    
        shift_left = tf.concat([x[:, :-1, :, :, :x.shape[4]//self.fold_div],
                tf.zeros([x.shape[0], 1, x.shape[2], x.shape[3], x.shape[4]//self.fold_div], dtype=tf.float32)], axis=1)
        shift_right = tf.concat([shift_right, shift_left, x[:, :, :, :, 2 * x.shape[4] // self.fold_div:]], axis=4)
        return tf.reshape(shift_right, size)


if __name__ == '__main__':
    # test inplace shift v.s. vanilla shift
    print("test start")
    tsm1 = TemporalShift(tf.keras.Sequential(), n_segment=8, n_div=8, inplace=False)
    # tsm2 = TemporalShift(tf.keras.Sequential(), n_segment=8, n_div=8, inplace=True)

    x = tf.random.uniform([2 * 8, 3, 224, 224])
    # tf.stop_gradient(x)
    y1 = tsm1(x)


    print('Test passed.')

