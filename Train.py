import argparse
import random
import shutil

import tensorflow as tf
import tensorflow.keras as keras
import os
import Data
import Model
import time
from tensorflow.keras import backend as K


def makeDataList(files):
    train = []
    val = []
    test = []
    for file in files:
        with open(file, 'r') as f:
            data = f.readlines()
        random.shuffle(data)
        train += data[:750]
        val += data[750:875]
        test += data[875:]
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    return train, val, test

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    # tf.keras.backend.set_floatx('float16')
    # os.environ['TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE'] = '1'
    # tf.keras.mixed_precision.set_global_policy('mixed_float16')
    parser = argparse.ArgumentParser()
    parser.add_argument("--augment", '-a', help="data augment", type=bool, default=True)
    parser.add_argument("--size", '-s', nargs=2, help="the size of frames", type=int, default=[224, 224])
    parser.add_argument("--message", '-m', help="commit", type=str, default="")
    parser.add_argument("--batchsize", '-b', help="batch size", type=int, default=1)
    parser.add_argument("--att", help="need att", type=str, default='proposed')
    parser.add_argument("--cnn", '-c', help="cnn archi", type=str, default='xcep')
    parser.add_argument("--rnn", '-r', help="rnn archi", type=str, default='convlstm')
    parser.add_argument("--frames", '-f', help="the nums of frames in one video", type=int, default=30)
    parser.add_argument("--fake", help="the fake image", type=str, default='df')
    parser.add_argument("--gpu", help="the num of gpu", type=str, default='0')
    args = parser.parse_args()

    gpu = args.gpu
    size: list = args.size
    batchSize = args.batchsize
    message = args.message
    cnn = args.cnn
    rnn = args.rnn
    frames = args.frames
    fakedata = args.fake
    att = args.att
    archi = cnn + rnn + att + "_frames"+str(frames)+"_"

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    if fakedata == 'df':
        fakedata = "Deepfakes"
    elif fakedata == 'fs':
        fakedata = "FaceSwap"
    elif fakedata == 'ff':
        fakedata = "Face2Face"
    elif fakedata == "nt":
        fakedata = "NeuralTextures"

    commit = message + archi + fakedata + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    if not os.path.exists("./result"):
        os.mkdir("./result")
    resultsFolder = "./result/" + commit
    if not os.path.exists(resultsFolder):
        os.mkdir(resultsFolder)
    checkpointsFolder = os.path.join(resultsFolder, "./checkpoints")
    if not os.path.exists(checkpointsFolder):
        os.mkdir(checkpointsFolder)
    logsFolder = os.path.join(resultsFolder, "logs")
    if not os.path.exists(logsFolder):
        os.mkdir(logsFolder)
    codeFolder = os.path.join(resultsFolder, "code")

    if not os.path.exists(codeFolder):
        os.mkdir(codeFolder)
    codeFiles = os.listdir("./")
    for codeFile in codeFiles:
        if codeFile.split(".")[-1] == "py":
            shutil.copy(codeFile, codeFolder)

    with open(resultsFolder+"/info", 'w') as f:
        f.writelines(str(args))

    dataFolder = "/data/ltm/keras/dataFile"
    # folders = ["Deepfakes0", "Face2Face0", "FaceSwap0", "NeuralTextures0",
    #      "youtube0", "youtube1", "youtube2", "youtube3"]
    folders = [fakedata+"0", "youtube0"]
    if fakedata == "all":
        folders = ["Deepfakes0", "Face2Face0", "FaceSwap0", "NeuralTextures0",
            "youtube0", "youtube1", "youtube2", "youtube3"]
    dataFolder = [os.path.join(dataFolder, folder) for folder in folders]
    train, val, test = makeDataList(dataFolder)
    with open(resultsFolder+"/test_data_list", 'w') as f:
        f.writelines(test)

    trainSet = Data.MyData(train, "train", batchSize, shape=tuple(size), num=frames)
    valSet = Data.MyData(val, "val", batchSize, shape=tuple(size), num=frames)

    inputShape = size.append(3)
    model = Model.getModel(cnn=cnn, rnn=rnn, att=att, inputShape=inputShape, frames=frames)

    optim = tf.keras.optimizers.Adam(learning_rate=0.00001,
                                     beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam')
    loss = tf.keras.losses.BinaryCrossentropy()
    model.compile(optimizer=optim, loss="binary_crossentropy",
                  metrics=['acc'], run_eagerly=False)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(checkpointsFolder, 'model.{epoch:02d}-{val_loss:.2f}.h5'), save_weights_only=True),
        tf.keras.callbacks.TensorBoard(log_dir=logsFolder),
    ]
    model.build((None, frames, 224, 224, 3))
    print(model.summary())
    model.fit(x=trainSet, callbacks=callbacks, epochs=200, validation_data=valSet,
              verbose=1, initial_epoch=0, validation_freq=1,
              max_queue_size=10, workers=4, use_multiprocessing=False)
