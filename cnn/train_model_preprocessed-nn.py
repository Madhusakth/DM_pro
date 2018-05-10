from cnn.Setup import Setup
import sys
import keras.backend as K
import numpy as np
import os
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

rel_filepath = sys.argv[1]

continue_setup = Setup('')
continue_setup.load(rel_filepath=rel_filepath)

change_lr = None

if change_lr is not None:
    K.set_value(continue_setup.getModel().optimizer.lr, change_lr)
    print('Changing the model optimizer learning rate to = %f' % K.get_value(continue_setup.getModel().optimizer.lr))
else:
    print('Model optimizer learning rate = %f' % K.get_value(continue_setup.getModel().optimizer.lr))

XTrain_directory, YTrain_directory, XValidation_directory, YValidation_directory, XTest_directory, YTest_directory = continue_setup.getDataDirectory()

no_of_classes = 15000


def train_data_generator(XTrain_directory, YTrain_directory):
    filenames = [str(i) + '.npy' for i in range(2000, 1144000 + 1, 2000)] + ['1144636.npy']

    y_all = np.load(os.path.join(YTrain_directory, 'target.npy'))

    while True:
        for filename in filenames:
            X = np.load(os.path.join(XTrain_directory, filename))

            if filename == '1144636.npy':
                start = 1143999
                end = 1144636
            elif filename == '2000.npy':
                start = 0
                end = 1999
            else:
                end = int(filename.replace('.npy', '')) - 1
                start = end - 2000
            X = X.reshape(-1, 2048)
            y = to_categorical(y_all[start:end], no_of_classes)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)
            yield(X_train, y_train)


def val_data_generator(XValidation_directory, YValidation_directory):
    filenames = [str(i) + '.npy' for i in range(2000, 1144000 + 1, 2000)] + ['1144636.npy']

    y_all = np.load(os.path.join(YValidation_directory, 'target.npy'))

    while True:
        for filename in filenames:
            X = np.load(os.path.join(XValidation_directory, filename))

            if filename == '1144636.npy':
                start = 1143999
                end = 1144636
            elif filename == '2000.npy':
                start = 0
                end = 1999
            else:
                end = int(filename.replace('.npy', '')) - 1
                start = end - 2000
            X = X.reshape(-1, 2048)
            y = to_categorical(y_all[start:end], no_of_classes)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)
            yield(X_test, y_test)


for epoch in range(continue_setup.getEpoch() + 1, 10000):
    print('Training \'%s\': Epoch %d' % (continue_setup.getName(), epoch))
    # dropout = continue_setup.getModel().fit(X_train_cnn, y_train,
    #                                         batch_size=64, epochs=1, verbose=1,
    #                                         validation_data=(X_val_cnn, y_val))

    XTrain_directory, YTrain_directory, XValidation_directory, YValidation_directory, XTest_directory, YTest_directory = continue_setup.getDataDirectory()

    dropout = continue_setup.getModel().fit_generator(generator=train_data_generator(XTrain_directory, YTrain_directory),
                                                      steps_per_epoch=(1144000/2000 + 1), epochs=1, verbose=1,
                                                      validation_data=val_data_generator(XValidation_directory, YValidation_directory),
                                                      validation_steps=(1144000/2000 + 1))

    continue_setup.updateEpochs(add_epochs=1,
                                train_acc=dropout.history['acc'],
                                train_loss=dropout.history['loss'],
                                val_acc=dropout.history['val_acc'],
                                val_loss=dropout.history['val_loss'],
                                test_acc=[0],
                                test_loss=[0],
                                allow_modify=True)

    continue_setup.save('setup')
