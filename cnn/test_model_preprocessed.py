from cnn.Setup import Setup
import sys
import keras.backend as K
import numpy as np
import os
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

rel_filepath = sys.argv[1]
XTest_directory = sys.argv[2]

continue_setup = Setup('')
continue_setup.load(rel_filepath=rel_filepath)

no_of_classes = 15000


def test_data_generator(XTest_directory, YTest_directory):
    filenames = [str(i) + '.npy' for i in range(2000, 114000 + 1, 2000)] + ['115424.npy']

    while True:
        for filename in filenames:
            X_test = np.load(os.path.join(XTest_directory, filename))
            X_test = X_test.reshape(-1, 2048, 1, 1)
            yield(X_test)


y_pred = continue_setup.getModel().predict_generator(test_data_generator(XTest_directory, None), steps=(114000/2000 + 1))
y_pred = np.argmax(y_pred, axis=1)

np.save('test_result.npy', y_pred)
