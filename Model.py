import os

import tensorflow.keras.applications.xception
import numpy as np
import tensorflow as tf
import tensorflow.keras.regularizers
import Attention

class CNN(tf.keras.Model):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn = tf.keras.applications.EfficientNetB0(include_top=False,
                                                        weights='imagenet',
                                                        input_tensor=None,
                                                        input_shape=(224, 224, 3),
                                                        pooling='avg',
                                                        classes=1280,
                                                        classifier_activation='softmax')
        self.cnn2 = tf.keras.applications.Xception(include_top=False,
                                                   weights='imagenet',
                                                   input_tensor=None,
                                                   input_shape=(224, 224, 3),
                                                   pooling='avg',
                                                   classes=1000,
                                                   classifier_activation='softmax')
    def call(self, inputs, training=True, mask=None):
        out = self.cnn2(inputs)
        return out

    def compute_output_shape(self, input_shape):
        return input_shape[0], 1280


class RNN(tf.keras.Model):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = tf.keras.layers.LSTM(2048, return_sequences=False)
        self.convlstm = tf.keras.layers.ConvLSTM2D(1280, (3,3), strides=(1,1), return_sequences=False, dropout=0.5)

    def call(self, inputs, training=True, mask=None, type="convlstm"):
        out = self.rnn(inputs)
        return out


class Detector(tf.keras.Model):
    def __init__(self):
        super(Detector, self).__init__()
        self.cnn = tf.keras.layers.TimeDistributed(CNN())
        self.rnn = RNN()
        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.drop1 = tf.keras.layers.Dropout(0.5)
        self.drop2 = tf.keras.layers.Dropout(0.5)
        self.fc1 = tf.keras.layers.Dense(512, activation='relu')
        self.fc2 = tf.keras.layers.Dense(2, activation='softmax')

    def call(self, inputs, training=True, mask=None):
        x = self.cnn(inputs)
        x = self.rnn(x)
        x = self.drop1(x)
        x = self.fc1(x)
        x = self.drop2(x)
        x = self.fc2(x)
        return x

def getModel(cnn='effi', rnn='convlstm', att="proposed", inputShape = (224, 224, 3), frames=30):
    if rnn == 'convlstm':
        pooling = None
    else:
        pooling = 'avg'
    if cnn=='effi':
        cnn = tf.keras.applications.efficientnet.EfficientNetB0(include_top=False,
                                                               weights='imagenet',
                                                               input_tensor=None,
                                                               input_shape=inputShape,
                                                               pooling=pooling,
                                                               classes=1000,
                                                               classifier_activation='softmax')
    else:
        cnn = tf.keras.applications.xception.Xception(include_top=False,
                                                        weights='imagenet',
                                                        input_tensor=None,
                                                        input_shape=inputShape,
                                                        pooling=pooling,
                                                        classes=1000,
                                                        classifier_activation='softmax')
    model = tf.keras.models.Sequential()
    if att == "proposed":
        attention = Attention.Attention(t=frames)
        model.add(attention)
    elif att == 'spatial':
        attention = Attention.OnlySpatial(t=frames)
        model.add(attention)
    model.add(tf.keras.layers.TimeDistributed(cnn))
    if rnn=='convlstm':
        model.add(tf.keras.layers.ConvLSTM2D(1024, (3,3), (1,1), return_sequences=False, dropout=0.5))
        model.add(tf.keras.layers.GlobalAvgPool2D())
    else:
        model.add(tf.keras.layers.LSTM(1024, return_sequences=False, dropout=0.5))  # 隐含层神经元个数20，输入步长19
    model.add(tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(2, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.build((None, frames, 224, 224, 3))
    model.summary()
    return model

if __name__ == '__main__':
    for i in range(10000):
        inputs = np.random.random_sample((1, 1, 224, 224, 3))
        model = tf.keras.layers.TimeDistributed(CNN())
        outputs = model(inputs)
        print(outputs.shape)
