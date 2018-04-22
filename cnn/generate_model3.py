# # Import
import glob
import ntpath
import os
import warnings
import csv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion

# warnings.filterwarnings('ignore')


numSamples = {}
topClasses = []
inputs = []
train_Y = []

print('Preprocessing Images')
csvfile = open('../data_reduced/train.csv', 'r')
csvreader = csv.reader(csvfile)
data = [line[:3] for line in csvreader]
data = data[1:]
i = 0

path = '../data_reduced/train'
for filename in os.listdir(path):
    entry = int(data[i][2])
    if entry > 450 or entry < 0:
        i = i + 1
        continue

    try:
        im = Image.open(os.path.join(path, filename)).convert('LA')
        arr = np.array(im)
        arr = arr[:, :, 0]
        # arr = arr.astype('float16')
        # arr = arr / 255.
        inputs.append(arr)
        train_Y.append(entry)
        i = i + 1

    except Exception as e:
        print('Can not identify file %s' % filename)
        i = i + 1

train_X = np.array(inputs)
del inputs


def split(array, size):
    arrays = []
    while len(array) > size:
        temp = array[:size]
        arrays.append(temp)
        array = array[size:]
    arrays.append(array)
    return arrays


stdSc = preprocessing.StandardScaler()
for arr in split(train_X, 500):
    stdSc.partial_fit(arr)
train_X = stdSc.transform(train_X)

train_X = train_X.reshape(-1, 256, 256, 1)

X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.2, random_state=13)


# # Convolutional Neural Network Model

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import to_categorical
from cnn.Setup import Setup

num_classes = 451

cnn = Sequential()
cnn.add(Conv2D(32, kernel_size=(8, 8), activation='linear', padding='same', input_shape=(256, 256, 1)))
cnn.add(LeakyReLU(alpha=0.1))
cnn.add(MaxPooling2D((8, 8), padding='same'))
cnn.add(Dropout(0.25))
cnn.add(Conv2D(64, (8, 8), activation='linear', padding='same'))
cnn.add(LeakyReLU(alpha=0.1))
cnn.add(MaxPooling2D(pool_size=(8, 8), padding='same'))
cnn.add(Dropout(0.25))
cnn.add(Conv2D(64, (8, 8), activation='linear', padding='same'))
cnn.add(LeakyReLU(alpha=0.1))
cnn.add(MaxPooling2D(pool_size=(8, 8), padding='same'))
cnn.add(Dropout(0.4))
cnn.add(Flatten())
cnn.add(Dense(256, activation='linear'))
cnn.add(LeakyReLU(alpha=0.1))
cnn.add(Dropout(0.3))
cnn.add(Dense(num_classes, activation='softmax'))

cnn.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.SGD(),
            metrics=['accuracy'])

X_train_cnn = X_train.reshape(-1, 256, 256, 1)
X_test_cnn = X_test.reshape(-1, 256, 256, 1)

y_train_one_hot = to_categorical(y_train, num_classes=num_classes)
y_test_one_hot = to_categorical(y_test, num_classes=num_classes)

# # Setup

cnn_setup = Setup('cnn_landmark_32-64-128-256_k88')
cnn_setup.setModel(cnn)
cnn_setup.setData(XTrain=X_train_cnn,
                  YTrain=y_train_one_hot,
                  XValidation=X_test_cnn,
                  YValidation=y_test_one_hot)
cnn_setup.save('setup')
