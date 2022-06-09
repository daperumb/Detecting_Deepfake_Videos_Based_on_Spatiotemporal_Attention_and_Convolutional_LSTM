import argparse
import random
import shutil
import sys
import tensorflow as tf
import tensorflow.keras as keras
import os
import Data
import Model
import time
from tensorflow.keras import backend as K

def makeDataList(file):
    test = []
    with open(file, 'r') as f:
        data = f.readlines()
    test = data
    return test

def getPositive(file):
    test = []
    with open(file, 'r') as f:
        data = f.readlines()
    test = [d for d in data if d.strip()[-1] == "0"]
    return test

def getNegative(file):
    test = []
    with open(file, 'r') as f:
        data = f.readlines()
    test = [d for d in data if d.strip()[-1] == "1"]
    return test

def avg(*l):
    sum = 0
    for data in l:
        sum += data
    return sum / len(l)

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument("--augment", '-a', help="data augment", type=bool, default=True)
    parser.add_argument("--size", '-s', nargs=2, help="the size of frames", type=int, default=[224, 224])
    parser.add_argument("--message", '-m', help="commit", type=str, default="")
    parser.add_argument("--batchsize", '-b', help="batch size", type=int, default=1)
    parser.add_argument("--att", help="need att", type=str, default='proposed')
    parser.add_argument("--cnn", '-c', help="cnn archi", type=str, default='effi')
    parser.add_argument("--rnn", '-r', help="rnn archi", type=str, default='convlstm')
    parser.add_argument("--frames", '-f', help="the nums of frames in one video", type=int, default=30)
    parser.add_argument("--gpu", help="the num of gpu", type=str)

    args = parser.parse_args()

    gpu = args.gpu
    size: list = args.size
    batchSize = args.batchsize
    message = args.message
    cnn = args.cnn
    rnn = args.rnn
    frames = args.frames
    att = args.att
    archi = cnn + rnn + att
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    DF ="/exp/result/" + "xcepconvlstmproposedDeepfakes2021-07-14_23:30:43" + "/test_data_list"
    testNet = os.path.join("/exp/result/" + "xcepconvlstmproposedDeepfakes2021-07-14_23:30:43" + "/checkpoints")

    testNets = os.listdir(testNet)
    nets = [os.path.join(testNet, weight) for weight in testNets]

    DFNega = getNegative(DF)
    Real = getPositive(DF)

    DFSet = Data.MyData(DFNega + Real, "test", batchSize, shape=tuple(size), num=frames)

    inputShape = size.append(3)
    model = Model.getModel(cnn=cnn, rnn=rnn, att=att, inputShape=inputShape)
    optim = tf.keras.optimizers.Adam(learning_rate=0.000001,
                                     beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam')
    loss = tf.keras.losses.BinaryCrossentropy()

    auc = tf.keras.metrics.AUC()
    model.compile(optimizer=optim, loss="binary_crossentropy",
                  metrics=['acc', auc], run_eagerly=False)

    model.build((None, 30, 224, 224, 3))

    for net in nets:
        sys.stdout.flush()
        model.load_weights(net)

        DF = model.evaluate(
            x=DFSet,
            verbose=0)

        print("DF: ", DF)

        sys.stdout.flush()

