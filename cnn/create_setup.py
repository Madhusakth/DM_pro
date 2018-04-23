# # Import
import csv
import os

import keras
import numpy as np
from PIL import Image
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from cnn.Setup import Setup

# warnings.filterwarnings('ignore')

# # ==========================================================================================
# # ==========================================================================================
# # ==========================================================================================

no_of_classes = 50

# # Convolutional Neural Network Model
print('Creating Model ...')
model_number = 207

first_layer = 32
second_layer = 64
third_layer = 128
forth_layer = 256

kernel_size = (3, 3)
pool_size = (3, 3)
strides = (1, 1)

cnn = Sequential()
cnn.add(Conv2D(first_layer, kernel_size=kernel_size, strides=strides, activation='linear', padding='same', input_shape=(256, 256, 1)))
cnn.add(LeakyReLU(alpha=0.1))
cnn.add(MaxPooling2D(pool_size=pool_size, padding='same'))
cnn.add(Dropout(0.25))
cnn.add(Conv2D(second_layer, kernel_size=kernel_size, strides=strides, activation='linear', padding='same'))
cnn.add(LeakyReLU(alpha=0.1))
cnn.add(MaxPooling2D(pool_size=pool_size, padding='same'))
cnn.add(Dropout(0.25))
cnn.add(Conv2D(third_layer, kernel_size=kernel_size, strides=strides, activation='linear', padding='same'))
cnn.add(LeakyReLU(alpha=0.1))
cnn.add(MaxPooling2D(pool_size=pool_size, padding='same'))
cnn.add(Dropout(0.4))
cnn.add(Flatten())
cnn.add(Dense(forth_layer, activation='linear'))
cnn.add(LeakyReLU(alpha=0.1))
cnn.add(Dropout(0.3))
cnn.add(Dense(no_of_classes + 1, activation='softmax'))

cnn.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.SGD(),
            metrics=['accuracy'])

cnn_setup = Setup('%s.cnn_landmark_%s-%s-%s-%s_k%s%s_p%s%s_s%s%s_reduced%d' % (model_number,
                                                                               first_layer,
                                                                               second_layer,
                                                                               third_layer,
                                                                               forth_layer,
                                                                               kernel_size[0], kernel_size[1],
                                                                               pool_size[0], pool_size[1],
                                                                               strides[0], strides[1],
                                                                               no_of_classes))

# # ==========================================================================================
# # ==========================================================================================
# # ==========================================================================================


datasets = {
    450: {
        'train_csv': '../data_reduced%s/train.csv' % '',
        'train_image_path': '../data_reduced%s/train' % '',
        'test_csv': '../data_reduced%s/test.csv' % '',
        'test_image_path': '../data_reduced%s/test' % '',
    },
    50: {
        'train_csv': '../data_reduced%s/train.csv' % '50',
        'train_image_path': '../data_reduced%s/train' % '50',
        'test_csv': '../data_reduced%s/test.csv' % '50',
        'test_image_path': '../data_reduced%s/test' % '50',
    },
    'full': {
        'train_csv': '../data/train.csv',
        'train_image_path': '../data/train',
        'test_csv': '../data/test.csv',
        'test_image_path': '../data/test',
    },
}

numSamples = {}
topClasses = []
inputs = []
train_Y = []

print('Creating Data ...')
csvfile = open(datasets[no_of_classes]['train_csv'], 'r')
csvreader = csv.reader(csvfile)
data = [line[:3] for line in csvreader]
data = data[1:]
i = 0

path = datasets[no_of_classes]['train_image_path']
for filename in os.listdir(path):
    entry = int(data[i][2])
    if entry > no_of_classes or entry < 0:
        i = i + 1
        continue

    try:
        im = Image.open(os.path.join(path, filename)).convert('LA')
        arr = np.array(im)
        arr = arr[:, :, 0]
        arr = arr.reshape(len(arr) * len(arr[0]))
        inputs.append(arr)
        train_Y.append(entry)
        i = i + 1

    except Exception as e:
        print('Can not identify file %s' % filename)
        i = i + 1

train_X = np.array(inputs)
del inputs

stdSc = preprocessing.StandardScaler()
stdSc.mean_ = np.mean(train_X)
stdSc.scale_ = np.std(train_X)

train_X = stdSc.transform(train_X, copy=False)

train_X = train_X.reshape(-1, 256, 256, 1)

X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.2, random_state=13)

X_train_cnn = X_train.reshape(-1, 256, 256, 1)
X_test_cnn = X_test.reshape(-1, 256, 256, 1)

y_train_one_hot = to_categorical(y_train, num_classes=(no_of_classes + 1))
y_test_one_hot = to_categorical(y_test, num_classes=(no_of_classes + 1))

# # Setup

print('Creating Setup ...')
cnn_setup.setModel(cnn)
cnn_setup.setData(XTrain=X_train_cnn,
                  YTrain=y_train_one_hot,
                  XValidation=X_test_cnn,
                  YValidation=y_test_one_hot)
cnn_setup.save('setup')
