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

# # Neural Network Model
print('Creating Model ...')
model_number = 903

first_layer = 32
second_layer = 64
third_layer = 128
forth_layer = 256

kernel_size = (3, 3)
pool_size = (3, 3)
strides = (1, 1)

nn = Sequential()
nn.add(Dense(first_layer, input_dim=2048, activation='relu'))
nn.add(Dense(second_layer, activation='relu'))
nn.add(Dense(third_layer, activation='relu'))
nn.add(Dense(forth_layer, activation='relu'))
nn.add(Dense(no_of_classes, activation='softmax'))


nn.compile(loss=keras.losses.categorical_crossentropy,
           optimizer=keras.optimizers.Adagrad(),
           metrics=['accuracy'])

model_name = '%s.nn_landmark_%s-%s-%s-%s_reduced%d' % (model_number,
                                                       first_layer,
                                                       second_layer,
                                                       third_layer,
                                                       forth_layer,
                                                       no_of_classes)
nn_setup = Setup(model_name)

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
nn_setup.setModel(nn)
nn_setup.setDataDirectory(XTrain_directory=os.path.join('setup', model_name, 'XTrain'),
                          XValidation_directory=os.path.join('setup', model_name, 'XTrain'),
                          XTest_directory=os.path.join('setup', model_name, 'XTrain'),
                          YTrain_directory=os.path.join('setup', model_name, 'YTrain'),
                          YValidation_directory=os.path.join('setup', model_name, 'YTrain'),
                          YTest_directory=os.path.join('setup', model_name, 'YTrain'))
nn_setup.save('setup')
