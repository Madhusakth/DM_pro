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

no_of_classes = 15000

# # Convolutional Neural Network Model
print('Creating Model ...')
model_number = 302

first_layer = 32
second_layer = 64
third_layer = 128
forth_layer = 256

kernel_size = (3, 3)
pool_size = (3, 3)
strides = (1, 1)

cnn = Sequential()
cnn.add(Conv2D(first_layer, kernel_size=kernel_size, strides=strides, activation='linear', padding='same', input_shape=(2048, 1, 1)))
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
cnn.add(Dense(no_of_classes, activation='softmax'))

cnn.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adam(),
            metrics=['accuracy'])

model_name = '%s.cnn_landmark_%s-%s-%s-%s_k%s%s_p%s%s_s%s%s_reduced%d' % (model_number,
                                                                          first_layer,
                                                                          second_layer,
                                                                          third_layer,
                                                                          forth_layer,
                                                                          kernel_size[0], kernel_size[1],
                                                                          pool_size[0], pool_size[1],
                                                                          strides[0], strides[1],
                                                                          no_of_classes)
cnn_setup = Setup(model_name)

# # ==========================================================================================
# # ==========================================================================================
# # ==========================================================================================

# print('Preparing Data ...')
#
# filenames = [str(i) + '.npy' for i in range(2000, 1144000 + 1, 2000)] + ['1144636.npy']
#
# X = np.empty((0, 2048))
#
# for filename in filenames:
#     print('.', end='')
#     temp = np.load('../data/preprocessed/%s' % filename)
#     X = np.append(X, temp, axis=0)
# print('')
#
# y = np.load('../data/preprocessed/target.npy')
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)
# X = None
# y = None
#
# X_train_cnn = X_train.reshape(-1, 2048, 1, 1)
# X_train = None
# X_test_cnn = X_test.reshape(-1, 2048, 1, 1)
# X_test = None
#
# y_train_one_hot = to_categorical(y_train, num_classes=no_of_classes)
# y_train = None
# y_test_one_hot = to_categorical(y_test, num_classes=no_of_classes)
# y_test = None

# # Setup

print('Creating Setup ...')
cnn_setup.setModel(cnn)
# cnn_setup.setData(XTrain=X_train_cnn,
#                   YTrain=y_train_one_hot,
#                   XValidation=X_test_cnn,
#                   YValidation=y_test_one_hot)
cnn_setup.setDataDirectory(XTrain_directory=os.path.join('setup', model_name, 'XTrain'),
                           XValidation_directory=os.path.join('setup', model_name, 'XTrain'),
                           XTest_directory=os.path.join('setup', model_name, 'XTrain'),
                           YTrain_directory=os.path.join('setup', model_name, 'YTrain'),
                           YValidation_directory=os.path.join('setup', model_name, 'YTrain'),
                           YTest_directory=os.path.join('setup', model_name, 'YTrain'))
cnn_setup.save('setup')
