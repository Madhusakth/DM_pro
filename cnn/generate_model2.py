# # Import
import glob
import ntpath
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion

# warnings.filterwarnings('ignore')


# # Load Input Data
def read_image_array(image_path):
    im = Image.open(image_path).convert('L')
    im_data = np.asarray(im).ravel()
    im_data = im_data
    return im_data


def read_image_list(image_path):
    im = Image.open(image_path).convert('L')
    im_data = list(im.getdata())
    return im_data


def plot_image(image_array):
    plt.figure()
    plt.imshow(image_array.reshape(256, -1), cmap=plt.get_cmap('gray'))


pixels = 256 * 256
data_limit = 35000


def getImageTrainData(data_limit):
    label_train_data = pd.read_csv(os.path.join(os.getcwd(), '../data_reduced/train.csv'), index_col='id')
    image_train_file_list = glob.glob(os.path.join(os.getcwd(), '../data_reduced/train/' + "*.jpg"))
    image_train_name = []
    image_train_data = []
    image_train_label = []
    for index, image_path in enumerate(image_train_file_list):
        image_name = ntpath.basename(image_path)
        try:
            image_train_name.append(image_name)
            arr = np.array(read_image_list(image_path))
            arr.astype('float16')
            arr = arr / 255.
            image_train_data.append(arr)
            image_train_label.append(label_train_data.loc[image_name.split('.jpg')[0]]['landmark_id'])
        except KeyError as e:
            print('KeyError')

        if (data_limit > 0 or data_limit is not False) and index + 1 >= data_limit:
            break

    image_train_data = np.array(image_train_data)
    image_train_data.reshape(-1, 256, 256, 1)
    return image_train_name, image_train_data, image_train_label


def getImageTestData(data_limit):
    image_test_file_list = glob.glob(os.path.join(os.getcwd(), '../data_reduced/test/' + "*.jpg"))
    image_test_name = []
    image_test_data = []
    for index, image_path in enumerate(image_test_file_list):
        image_name = ntpath.basename(image_path)
        try:
            image_test_name.append(image_name)
            image_test_data.append(read_image_list(image_path))
        except KeyError as e:
            print('KeyError')

        if (data_limit > 0 or data_limit is not False) and index + 1 >= data_limit:
            break

    return image_test_name, image_test_data

def split(array, size):
    arrays = []
    while len(array) > size:
        temp = array[:size]
        arrays.append(temp)
        array = array[size:]
    arrays.append(array)
    return arrays

_, X, y = getImageTrainData(data_limit)

X_train, X_test = np.empty((0, 65536))
y_train, y_test = np.empty(0)

y_split = split(y, 100)
for index, x in enumerate(split(X, 100)):
    X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(x, y_split[index], test_size=0.2)
    X_train.append(X_train_split)
    X_test.append(X_test_split)
    y_train.append(y_train_split)
    y_test.append(y_test_split)

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
