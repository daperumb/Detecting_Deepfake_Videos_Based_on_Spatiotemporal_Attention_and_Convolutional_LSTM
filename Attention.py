import tensorflow as tf
import numpy as np


class Spatial(tf.keras.Model):
    def __init__(self, c=3, r=3):
        super(Spatial, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(1, (1,1), (1,1), 'same', activation='sigmoid')
        self.reshape2 = tf.keras.layers.Reshape((224*224, c))
        self.reshape1 = tf.keras.layers.Reshape((1, 224*224))
        self.reshape3 = tf.keras.layers.Reshape((1,1,c))
        self.conv2 = tf.keras.layers.Conv2D(r, (1,1), (1,1), 'same')
        self.layerNorm = tf.keras.layers.LayerNormalization(axis=(1,2,3))
        self.relu = tf.keras.layers.ReLU()
        self.conv3 = tf.keras.layers.Conv2D(c, (1,1), (1,1), "same")

    def call(self, inputs, training=None, mask=None):
        x1 = self.conv1(inputs)
        x1 = self.reshape1(x1)
        x2 = self.reshape2(inputs)
        x3 = tf.matmul(x1, x2)
        x3 = self.reshape3(x3)
        x4 = self.conv2(x3)
        x5 = self.layerNorm(x4)
        x6 = self.relu(x5)
        x7 = self.conv3(x6)
        # x8 = tf.add(inputs, x7)
        return x7

    def compute_output_shape(self, input_shape):
        return input_shape[0], 1, 1, input_shape[-1]

class OnlySpatial(tf.keras.Model):
    def __init__(self, t=40, c=3):
        super(OnlySpatial, self).__init__()
        self.spatial = tf.keras.layers.TimeDistributed(Spatial(c=3, r=3))

    def call(self, inputs, training=None, mask=None):
        # input shape (NTHWC)
        w = self.spatial(inputs)
        x1 = tf.add(inputs, w)
        return x1

class Attention(tf.keras.Model):
    def __init__(self, t=30, c=3):
        super(Attention, self).__init__()
        self.spatial = tf.keras.layers.TimeDistributed(Spatial(c=3, r=3))
        self.q = tf.keras.layers.Conv2D(c, (1,1), (1,1), padding='same')
        self.k = tf.keras.layers.Conv2D(c, (1,1), (1,1), padding='same')
        self.v = tf.keras.layers.Conv2D(c, (1,1), (1,1), padding='same')
        self.qReshape = tf.keras.layers.Reshape((t,c))
        self.kReshape = tf.keras.layers.Reshape((c,t))
        self.vReshape = tf.keras.layers.Reshape((t,c))
        self.soft = tf.keras.layers.Softmax()
        self.reshape = tf.keras.layers.Reshape((t,1,1,c))
        self.conv1 = tf.keras.layers.Conv2D(1,(1,1),(1,1), padding='same')

    def call(self, inputs, training=None, mask=None):
        # input shape (NTHWC)
        w = self.spatial(inputs)
        x1 = tf.add(inputs, w)
        q = self.qReshape(self.q(w))
        k = self.kReshape(self.k(w))
        v = self.vReshape(self.v(w))
        weight = tf.matmul(q, k)
        weight = self.soft(weight)
        x2 = tf.matmul(weight, v)
        x2 = self.reshape(x2)
        w = w + self.conv1(x2)
        out = x1 * w + inputs
        return out

    def compute_output_shape(self, input_shape):
        return input_shape

if __name__ == '__main__':
    att = Attention()
    # att = Attention()
    inputs = np.ones([2, 30, 16, 16, 3])
    re = att(inputs)
    print(re)


