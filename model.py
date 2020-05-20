# ported to TF from https://github.com/midas-research/mobile-vsr/blob/master/model.py
import math
import numpy as np

import keras
from depthwise import *
from keras.layers import Dense, ReLU
from keras.layers import Conv1D, Conv2D, Conv3D, ZeroPadding3D
from keras.layers import BatchNormalization, AveragePooling2D, MaxPool1D
from keras.models import Model

import tensorflow as tf

def LipRes(alpha=2, reduction=1, num_classes=256):
    block = lambda in_planes, planes, stride: \
        LipResBlock(in_planes, planes, stride, reduction=reduction)

    return ResNet(block, [alpha, alpha, alpha, alpha], reduction, num_classes) # TODO tunable alpha param + # alpha blocks

class ResNet(Model):
    def __init__(self, block, num_blocks, reduction=1, num_classes=256):
        super(ResNet, self).__init__()

        self.reduction = float(reduction) ** 0.5
        self.num_classes = num_classes
        self.in_planes = 64 #int(16 / self.reduction)

        self.layer1   = self._make_layer(block, self.in_planes, num_blocks[0], stride=1)
        self.layer2   = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3   = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4   = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc       = Dense(num_classes)
        self.bnfc     = BatchNormalization(momentum=0.1, epsilon=1e-5)
        self.avgpoool = AveragePooling2D()

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
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        # 464 512 1 1
        x = x.view(x.size(0), -1) # may need to change to flatten or reshape...
        # 464 512
        x = self.fc(x)
        x = self.bnfc(x)

class LipNext(Model):
    def __init__(self, inputDim=256, hiddenDim=512, nClasses=500, frameLen=29, alpha=2):
        super(LipNext, self).__init__()

        initializer = tf.initializers.VarianceScaling(scale=2.0) # added for initialization

        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.nClasses = nClasses
        self.frameLen = frameLen
        self.nLayers = 2
        self.alpha = alpha
        # frontend3D
        self.frontend3D = Sequential ( [
                ZeroPadding3D(padding=(1,1,1)), # double check channel placement
                Conv3D(64, kernel_size=(3,3,3), strides=(1,2,2), use_bias=False, kernel_initializer=initializer),
                BatchNormalization(momentum=.1, epsilon=1e-5), # should this be .9 instead?
                ReLU(), # check in place?
                # group convolution - TODO: THIS IS NOT RIGHT
                ZeroPadding3D(padding=(1,1,1)), # double check channel placement
                Conv3D(64, kernel_size=(3,3,3), strides=(1,2,2), use_bias=False, kernel_initializer=initializer),
                ZeroPadding3D(padding=(1,0,0)), # double check channel placement
                Conv3D(64, kernel_size=(3,1,1), strides=(1,1,1), use_bias=False, kernel_initializer=initializer)
            ] )
        # resnet
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
                Dense(self.inputDim),
                BatchNormalization(momentum=0.1, epsilon=1e-5),
                ReLU(),
                Dense(self.nClasses)
            ] )

        # now ignored due to initializer
        # self._initialize_weights()

    def call(self, x):
        # TODO: direct copy need to check all of this at runtime
        x = self.frontend3D(x)
        # 16, 64, 29, 22,22
        x = x.transpose(1, 2)
        # 16, 29, 64 , 22, 22
        x = x.contiguous()
        
        x = x.view(-1, 64, x.size(3), x.size(4))
        # 464, 64, 22, 22
        x = self.resnet34(x)
        # 464 256
        x = x.view(-1, self.frameLen, self.inputDim)
        # 16 29 256
        x = x.transpose(1, 2)
        # 16 256 29
        x = self.backend_conv1(x)
        x = torch.mean(x,2)
        # x = x.view(-1, 4, 16, 16)
        x = self.backend_conv2(x)
        # x = x.view(-1, self.nClasses)
        return x

    def _initialize_weights(self):
        raise NotImplementedError

def lipnext(inputDim=256, hiddenDim=512, nClasses=500, frameLen=29, alpha=2):
    model = LipNext(inputDim=inputDim, hiddenDim=hiddenDim, nClasses=nClasses, frameLen=frameLen, alpha=alpha)
    return model
