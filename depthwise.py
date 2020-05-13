# Ported to TF from https://github.com/midas-research/mobile-vsr/blob/master/depthwise.py
import keras
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, DepthwiseConv2D

class LipResBlock(keras.Model):
    def __init__ (self, in_planes, out_planes, stride=1, reduction=1):
        super(LipResBlock, self).__init__()

        initializer = tf.initializers.VarianceScaling(scale=2.0) # added for initialization

        self.expansion  = 1
        self.in_planes  = in_planes
        self.mid_planes = mid_planes = int(self.expansion * out_planes)
        self.out_planes = out_planes

        self.conv1 = Conv2D (mid_planes, kernel_size=1, use_bias=False, kernel_initializer=initializer)
        self.bn1   = BatchNormalization (momentum=0.1, epsilon=1e-5)

        self.depth = DepthwiseConv2D (kernel_size=3, use_bias=False, kernel_initializer=initializer)
        self.bn2   = BatchNormalization (momentum=0.1, epsilon=1e-5)

        self.conv3 = Conv2D (out_planes, kernel_size=1 use_bias=False, stride=(stride,stride), kernel_initializer=initializer)
        self.bn3   = BatchNormalization (momentum=0.1, epsilon=1e-5)

        self.shortcut = Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = Conv2d(out_planes, kernel_size=1, stride=stride, use_ bias=False, kernel_initializer=initializer)

    def call(self, x):
        out = keras.activations.relu(self.bn1(self.conv1(x)))

        self.int_nchw = out.size()

        out = self.bn2(self.depth(out))

        out = self.bn3(self.conv3(out))

        self.out_nchw = out.size

        out += self.shortcut(x)
        out = keras.activations.relu(out)
        return out