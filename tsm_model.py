# ported to TF from https://github.com/midas-research/mobile-vsr/blob/master/model.py
import math
import numpy as np

import tensorflow as tf
#import keras
from depthwise import *
from tensorflow.keras.layers import InputLayer, Dense, ReLU, Flatten, Permute
from tensorflow.keras.layers import Conv1D, Conv2D, Conv3D, ZeroPadding3D
from tensorflow.keras.layers import BatchNormalization, AveragePooling2D, MaxPool1D
from tensorflow.keras.models import Model
from DepthwiseConv3D import DepthwiseConv3D
from tsm import TemporalShift

def LipRes(alpha=2, reduction=1, num_classes=256):
    block = lambda in_planes, planes, stride: \
        LipResBlock(in_planes, planes, stride, reduction=reduction)

    return ResNet(block, [alpha, alpha, alpha, alpha], reduction, num_classes) # TODO tunable alpha param + # alpha blocks

class ResNet(tf.keras.Model):
    def __init__(self, block, num_blocks, reduction=1, num_classes=256):
        super(ResNet, self).__init__()

        self.reduction = float(reduction) ** 0.5
        self.num_classes = num_classes
        self.in_planes = 64 #int(16 / self.reduction)

        self.layer1   = self._make_layer(block, self.in_planes, num_blocks[0], stride=1)
        self.tsm1     = TemporalShift(Sequential(InputLayer(input_shape=(22,22,64,))), 8,8,False)
        self.layer2   = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3   = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4   = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.tsm2     = TemporalShift(Sequential(InputLayer(input_shape=(22,22,256,))), 8,8,False)
        self.flatten  = Flatten()
        self.fc       = Dense(num_classes)
        self.bnfc     = BatchNormalization(momentum=0.1, epsilon=1e-5)
        self.avgpool = AveragePooling2D()

        # TODO: weight initialization port -> done in depthwise.py
        # for m in self.modules():
        #     if isinstance(m, Conv2d):
        #         torch.nn.init.kaiming_uniform_(m.weight)
        #     elif isinstance(m, BatchNormalization):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()



    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        planes = int(planes)
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return Sequential(layers)


    def call(self, x):
        x = self.layer1(x)
        x = self.tsm1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        #print(f'preTSM shape {x.shape}')
        #print(f'postTSM shape {x.shape}')
        x = self.layer4(x)

        x = self.avgpool(x)
        # 464 512 1 1
        x = self.flatten(x)
        # 464 512
        # x = self.inp(x)
        x = self.fc(x)
        x = self.bnfc(x)
        return x

class LipNext(tf.keras.Model):
    def __init__(self, inputDim=256, hiddenDim=512, nClasses=500, frameLen=29, alpha=2):
        super(LipNext, self).__init__()

#        initializer = tf.initializers.VarianceScaling(scale=2.0) # added for initialization
        initializer = 'glorot_uniform'

        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.nClasses = nClasses
        self.frameLen = frameLen
        self.nLayers = 2
        self.alpha = alpha

        # Input
        self.input_layer = InputLayer(input_shape=(29,88,88,1,))
        self.resnet34 = LipRes(self.alpha)
        # backend
        self.backend_conv1 = Sequential ( [  
                Conv1D(2*self.inputDim, kernel_size=5, strides=2, use_bias=False, kernel_initializer=initializer),
                BatchNormalization(momentum=0.1, epsilon=1e-5),
                ReLU(),
                MaxPool1D(2,2),
                Conv1D(4*self.inputDim, kernel_size=5, strides=2, use_bias=False, kernel_initializer=initializer),
                BatchNormalization(momentum=0.1, epsilon=1e-5),
                ReLU()
            ] )
        self.backend_conv2 = Sequential ( [
                Dense(self.inputDim, input_shape=(4 * self.inputDim, )),
                BatchNormalization(momentum=0.1, epsilon=1e-5),
                ReLU(),
                Dense(self.nClasses)
            ] )

        # now ignored due to initializer
        # self._initialize_weights()

    def call(self, x):
        x = self.input_layer(x)
        # Shape: None, 29, 88, 88, 1
        # rehshape & perm
        '''print(f'1{x.shape}')
        x = tf.reshape(x, [-1,  x.shape[2], x.shape[3], 1])
        x = self.permute1(x)
        # Front end TSM
        x = self.tsm(x)
        print(x.shape)
        x = self.permute2(x)
        print(f'TSM shape: {x.shape}')
        x = tf.reshape(x, [-1, self.frameLen, x.shape[1], x.shape[2], 1])
        # Shape: None, 29, 22, 22, 64'''
        
        print(f'input shape: {x.shape}')
        x = tf.reshape(x, [-1,  x.shape[2], x.shape[3], 1])
        # Shape: None, 88, 88,, 1
        
        x = self.resnet34(x)
        # Shape: None, 256
        
        x = tf.reshape(x, [-1, self.frameLen, self.inputDim])
        # Shape: None, 29, 256
        
        x = self.backend_conv1(x)
        # Shape: None, 1, 1024
        
        x = tf.math.reduce_mean(x, axis=1)
        # Shape: None, 1024
        
        x = self.backend_conv2(x)
        # Shape: None, nClasses

        return x

    def _initialize_weights(self):
        raise NotImplementedError

def lipnext(inputDim=256, hiddenDim=512, nClasses=500, frameLen=29, alpha=2):
    model = LipNext(inputDim=inputDim, hiddenDim=hiddenDim, nClasses=nClasses, frameLen=frameLen, alpha=alpha)
    return model
