import tensorflow as tf
import tensorflow.keras as keras
import os
import numpy as np
import random
import cv2


class MyData(keras.utils.Sequence):

    def __init__(self, dataList, datatype ,batchSize=1, shape=(224,224), num=20):
        super(MyData, self).__init__()
        self.shape = shape
        self.dataList = [data.split(" ") for data in dataList]
        self.batchSize = batchSize
        self.num = num
        self.sampleCount = 10
        self.datatype = datatype

    def __len__(self):
        return len(self.dataList)

    def __getitem__(self, index):
        dataList = self.dataList[index:index+self.batchSize]
        videos = []
        labels = []
        for data in dataList:
            directory = data[0]
            videos.append(self.readFromFrameDir(directory, shape=self.shape, num=self.num))
            labels.append(int(data[1]))
        labels = keras.utils.to_categorical(labels, num_classes=2)
        return np.stack(videos), labels

    def setVideoPathList(self, dataList):
        self.dataList = dataList

    def setCuda(self, device):
        self.device = device

    def readDataFile(self, datafile):
        with open(datafile, mode="r") as datafile:
            dirList = datafile.readlines()
            return dirList

    def transform(self, video):
        imgGen = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=45,
            samplewise_center=True,
            samplewise_std_normalization=True,
            brightness_range=[0.5, 1.5],
            channel_shift_range=50.0,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        transPara = imgGen.get_random_transform(img_shape=self.shape, seed=None)
        video = [imgGen.apply_transform(frame, transPara) for frame in video]
        return video

    def readFromFrameDir(self, frameDir, shape=(224,224), num=20):
        frames = os.listdir(frameDir)[:num]
        video = []
        for frame in frames:
            frame = os.path.join(frameDir, frame)
            # img = skimage.io.imread(frame)
            # img = skimage.transform.resize(img, shape)
            img = cv2.imread(frame)
            img = cv2.resize(img, shape)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            video.append(img)
        if self.datatype == "train":
            video = self.transform(video)
        # self.saveVideo(video, "train_after_trans" + frameDir.split("/")[-1])
        video = np.stack(video, axis=0)
        return video

    def saveVideo(self, video, videoPath):
        if self.sampleCount > 0:
            self.sampleCount -= 1
        else:
            return
        videosFolder = "./train_samples"
        if not os.path.exists(videosFolder):
            os.mkdir(videosFolder)
        videoFolder = os.path.join(videosFolder, videoPath)
        if not os.path.exists(videoFolder):
            os.mkdir(videoFolder)

        num = 0
        for frame in video:
            savePath = os.path.join(videoFolder, "{:0>4d}.png".format(num))
            # print(savePath)
            cv2.imwrite(savePath, frame)
            num += 1


if __name__ == '__main__':
    def splitDataFile(fakeFile, trueFile):
        with open(fakeFile, 'r') as f:
            fakeData = f.readlines()
        with open(trueFile, 'r') as f:
            trueData = f.readlines()
        random.shuffle(fakeData)
        random.shuffle(trueData)
        train = fakeData[:800] + trueData[:800]
        val = fakeData[800:900] + trueData[800:900]
        test = fakeData[900:] + trueData[900:]
        return train, val, test


    def wrapDataFile(fakeFile, trueFile, compresson):
        train, val, test = splitDataFile(fakeFile, trueFile)
        parent_dir = "/data/ff++" + compresson
        train = [os.path.join(parent_dir, t) for t in train]
        val = [os.path.join(parent_dir, t) for t in val]
        test = [os.path.join(parent_dir, t) for t in test]
        return train, val, test

    train, val, test = wrapDataFile("/data/ltm/data/NT", "/data/ltm/data/Real", "c40")
    dataset = MyData(train, 4)
    t = dataset.__getitem__(0)
    img = t[0]
    label = t[1]
    print(img.shape)
    print(label.shape)
    print(label)
