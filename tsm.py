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

    def forward(self, x):
        x = self.shift(x, self.n_segment, fold_div=self.fold_div, inplace=self.inplace)
        return self.net(x)

    @staticmethod
    def shift(x, n_segment, fold_div=3, inplace=False):
        nt, c, h, w = x.shape
        n_batch = nt // n_segment
        x = tf.reshape(x, [n_batch, n_segment, c, h, w])
        print(f'tsm reshape{x.shape}')
        
        fold = c // fold_div
        if inplace:
            # Due to some out of order error when performing parallel computing. 
            # May need to write a CUDA kernel.
            raise NotImplementedError  
            # out = InplaceShift.apply(x, fold)
        else:
            #session = tf.compat.v1.Session()
            #ith session.as_default():
            #x = np.array(x.eval(session=session))
            out = np.zeros(x.shape)
            print(x[:, 1:, :fold])
            print(x)
            print(out.shape)
            print(out[:, :-1, :fold].shape)
            out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
            out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

            out = tf.convert_to_tensor(out)
            out = tf.reshape(out, [nt, c, h, w])
        return out

    def call(self, x):
        size = x.shape
        x = tf.reshape(x, [-1, 29, size[1], size[2], size[3]])
        #return tf.reshape(tf.roll(x, 1, 1), size)
        #return tf.concat((tf.roll(x[:,:,:,:,:size[3]//4], 1, 1), 
        #                    x[:,:,:,:,size[3]//4::]),4)
        return tf.reshape(tf.concat((tf.roll(x[:,:,:,:,:size[3]//4], 1, 1), 
                                     x[:,:,:,:,size[3]//4::]),4), size)

if __name__ == '__main__':
    # test inplace shift v.s. vanilla shift
    print("test start")
    tsm1 = TemporalShift(tf.keras.Sequential(), n_segment=8, n_div=8, inplace=False)
    # tsm2 = TemporalShift(tf.keras.Sequential(), n_segment=8, n_div=8, inplace=True)

    x = tf.random.uniform([2 * 8, 3, 224, 224])
    # tf.stop_gradient(x)
    y1 = tsm1(x)


    print('Test passed.')

