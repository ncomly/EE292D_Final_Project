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

        # split along channel dimension (1/fold_div shifted right,
        # 1/fold_div shifted left, rest not shifted)
        folds = tf.split(x, num_or_size_splits=self.fold_div, axis=4)
        # shift right
        '''shift_right = tf.roll(folds[0], shift=1, axis=1)
        s_shape = shift_right.shape
        # zero padding
        shift_right = tf.concat([tf.zeros([s_shape[0], 1, s_shape[2], s_shape[3], s_shape[4]], dtype=tf.float32), shift_right[:, 1:, :, :, :]], axis=1)

        # shift left
        shift_left = tf.roll(folds[1], shift=-1, axis=1)
        s_shape = shift_left.shape
        # zero padding
        shift_left = tf.concat([shift_left[:, :-1, :, :, :], tf.zeros([s_shape[0], 1, s_shape[2], s_shape[3], s_shape[4]], dtype=tf.float32)], axis=1)
        # print("shift right ", shift_right.shape)
        # print("shift left ", shift_left.shape)
        # concatenate shifted arrays in both directions
        shift_right = tf.concat([shift_right, shift_left], axis=4)
        # concatenate shifted arrays with rest of data (not shifted)
        for i in range(self.fold_div - 2):
            shift_right = tf.concat([shift_right, folds[i + 2]], axis=4)
        # reshape back to original x size
        return tf.reshape(shift_right, size)'''

        shift_right = tf.roll(x[:, :, :, :, :x.shape[4] // self.fold_div], shift=1, axis=1)
        s_shape = shift_right.shape
        # shift right
        shift_right = tf.concat([tf.zeros([s_shape[0], 1, s_shape[2], s_shape[3], s_shape[4]], dtype=tf.float32), shift_right[:, 1:, :, :, :]], axis=1)
    
        shift_left = tf.roll(x[:, :, :, :, x.shape[4] // self.fold_div:2 * x.shape[4] // self.fold_div], shift=-1, axis=1)
        s_shape = shift_left.shape
        shift_left = tf.concat([shift_left[:, :-1, :, :, :], tf.zeros([s_shape[0], 1, s_shape[2], s_shape[3], s_shape[4]], dtype=tf.float32)], axis=1)
        # print("shift right ", shift_right.shape)
        # print("shift left ", shift_left.shape)
        shift_right = tf.concat([shift_right, shift_left, x[:, :, :, :, 2 * x.shape[4] // self.fold_div:]], axis=4)
        return tf.reshape(shift_right, size)

        '''print("shift right__", shift_right.shape)
        print("shift right shape: ", shift_right)
        print("num splits: ", self.fold_div)
        splits = tf.split(x, self.fold_div, 4)
        split0 = splits[0]
        print("split 0 shape: ", split0.shape)
        split1 = splits[1]
        print("split 1 shape: ", split1.shape)

        split0_slices = tf.split(split0, num_or_size_splits=split0.shape[1], axis=1)
        print("split0 slices shape", split0_slices)
        zfill0 = tf.zeros_like(split0_slices[0])
        print("split0_slices[0]:", split0_slices[0])
        print("split1 slices", split0_slices[:-1])
        split0 = tf.concat([zfill0] + split0_slices[:-1], axis=1)
        print("split 0 shape; ", split0.shape)
        split1_slices = tf.split(split1, num_or_size_splits=split1.shape[1], axis=1)
        print("split 1 shape [1]: ", split1.shape[1])
        zfill1 = tf.zeros_like(split1_slices[0])
        split1 = tf.concat(split1_slices[1:] + [zfill1], axis=1)
        print("split1 : ", split1.shape)
        out = tf.reshape(tf.concat([split0, split1] + splits[2:], axis=4), size)
        print("out shape: ", out.shape)
        return out'''
        #return tf.reshape(tf.roll(x, 1, 1), size)
        #return tf.concat((tf.roll(x[:,:,:,:,:size[3]//4], 1, 1), 
        #                    x[:,:,:,:,size[3]//4::]),4)
        # return tf.reshape(tf.concat((tf.roll(x[:,:,:,:,:size[3]//4], 1, 1), 
        #                             x[:,:,:,:,size[3]//4::]),4), size)

if __name__ == '__main__':
    # test inplace shift v.s. vanilla shift
    print("test start")
    tsm1 = TemporalShift(tf.keras.Sequential(), n_segment=8, n_div=8, inplace=False)
    # tsm2 = TemporalShift(tf.keras.Sequential(), n_segment=8, n_div=8, inplace=True)

    x = tf.random.uniform([2 * 8, 3, 224, 224])
    # tf.stop_gradient(x)
    y1 = tsm1(x)


    print('Test passed.')

