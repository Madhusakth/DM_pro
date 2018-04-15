from unittest import TestCase

from cnn.Setup import Setup

import keras
from keras.datasets import fashion_mnist
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


class TestSetup(TestCase):
    @classmethod
    def setUpClass(cls):
        (cls.train_X, cls.train_Y), (cls.test_X, cls.test_Y) = fashion_mnist.load_data()
        (cls.train_X, cls.train_Y), (cls.test_X, cls.test_Y) = (cls.train_X[:100], cls.train_Y[:100]), (cls.test_X[:100], cls.test_Y[:100])

        cls.train_X = cls.train_X.reshape(-1, 28, 28, 1)
        cls.test_X = cls.test_X.reshape(-1, 28, 28, 1)

        cls.train_X = cls.train_X.astype('float32')
        cls.test_X = cls.test_X.astype('float32')
        cls.train_X = cls.train_X / 255.
        cls.test_X = cls.test_X / 255.

        # Change the labels from categorical to one-hot encoding
        train_Y_one_hot = to_categorical(cls.train_Y)
        test_Y_one_hot = to_categorical(cls.test_Y)

        cls.train_X, cls.valid_X, train_label, cls.valid_label = train_test_split(cls.train_X, train_Y_one_hot, test_size=0.2,
                                                                      random_state=13)

        batch_size = 64
        epochs = 2
        num_classes = 10

        cls.fashion_model = Sequential()
        cls.fashion_model.add(Conv2D(32, kernel_size=(3, 3), activation='linear', input_shape=(28, 28, 1), padding='same'))
        cls.fashion_model.add(LeakyReLU(alpha=0.1))
        cls.fashion_model.add(MaxPooling2D((2, 2), padding='same'))
        cls.fashion_model.add(Conv2D(64, (3, 3), activation='linear', padding='same'))
        cls.fashion_model.add(LeakyReLU(alpha=0.1))
        cls.fashion_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        cls.fashion_model.add(Conv2D(128, (3, 3), activation='linear', padding='same'))
        cls.fashion_model.add(LeakyReLU(alpha=0.1))
        cls.fashion_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        cls.fashion_model.add(Flatten())
        cls.fashion_model.add(Dense(128, activation='linear'))
        cls.fashion_model.add(LeakyReLU(alpha=0.1))
        cls.fashion_model.add(Dense(num_classes, activation='softmax'))

        cls.fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
                              metrics=['accuracy'])

        cls.fashion_train = cls.fashion_model.fit(cls.train_X, train_label, batch_size=batch_size, epochs=epochs, verbose=1,
                                          validation_data=(cls.valid_X, cls.valid_label))

        cls.test_eval = cls.fashion_model.evaluate(cls.test_X, test_Y_one_hot, verbose=0)

    def setUp(self):
        self.setup = Setup('fashion_model')

    def test_updateEpochs_correctList_noModify(self):
        mList = [1, 2, 3]
        self.setup.updateEpochs(3, mList, mList, mList, mList, mList, mList, allow_modify=False)
        self.assertEqual(self.setup._train_accuracy, [1, 2, 3])
        self.assertEqual(self.setup._train_loss, [1, 2, 3])
        self.assertEqual(self.setup._val_accuracy, [1, 2, 3])
        self.assertEqual(self.setup._val_loss, [1, 2, 3])
        self.assertEqual(self.setup._test_accuracy, [1, 2, 3])
        self.assertEqual(self.setup._test_loss, [1, 2, 3])

    def test_updateEpochs_smallerList_noModify(self):
        mList = [1, 2, 3]
        self.assertRaises(ValueError, self.setup.updateEpochs, 3, mList, mList, mList[:-1], mList, mList, mList, allow_modify=False)

    def test_updateEpochs_largerList_noModify(self):
        mList = [1, 2, 3]
        self.assertRaises(ValueError, self.setup.updateEpochs, 3, mList, mList, mList, mList, mList, mList + [4], allow_modify=False)

    def test_updateEpochs_correctList_modifyAllowed(self):
        mList = [1, 2, 3]
        self.setup.updateEpochs(3, mList, mList, mList, mList, mList, mList, allow_modify=True)
        self.assertEqual(self.setup._train_accuracy, [1, 2, 3])
        self.assertEqual(self.setup._train_loss, [1, 2, 3])
        self.assertEqual(self.setup._val_accuracy, [1, 2, 3])
        self.assertEqual(self.setup._val_loss, [1, 2, 3])
        self.assertEqual(self.setup._test_accuracy, [1, 2, 3])
        self.assertEqual(self.setup._test_loss, [1, 2, 3])

    def test_updateEpochs_smallerList_modifyAllowed(self):
        mList = [1, 2, 3]
        self.setup.updateEpochs(3, mList, mList, mList, mList, mList[:-1], mList, allow_modify=True)
        self.assertEqual(self.setup._train_accuracy, [1, 2, 3])
        self.assertEqual(self.setup._train_loss, [1, 2, 3])
        self.assertEqual(self.setup._val_accuracy, [1, 2, 3])
        self.assertEqual(self.setup._val_loss, [1, 2, 3])
        self.assertEqual(self.setup._test_accuracy, [1, 2, 2])
        self.assertEqual(self.setup._test_loss, [1, 2, 3])

    def test_updateEpochs_largerList_modifyAllowed(self):
        mList = [1, 2, 3]
        self.setup.updateEpochs(3, mList, mList, mList, mList, mList, mList + [4], allow_modify=True)
        self.assertEqual(self.setup._train_accuracy, [1, 2, 3])
        self.assertEqual(self.setup._train_loss, [1, 2, 3])
        self.assertEqual(self.setup._val_accuracy, [1, 2, 3])
        self.assertEqual(self.setup._val_loss, [1, 2, 3])
        self.assertEqual(self.setup._test_accuracy, [1, 2, 3])
        self.assertEqual(self.setup._test_loss, [1, 2, 3])

    def test_positive(self):
        self.setup.setModel(self.fashion_model)

        self.setup.setData(XTrain=self.train_X)
        self.setup.setData(XValidation=self.valid_X)
        self.setup.setData(XTest=self.test_X)
        self.setup.setData(YTrain=self.train_Y)
        self.setup.setData(YValidation=self.valid_label)
        self.setup.setData(YTest=self.test_Y)

        self.setup.save('\\setup\\')
