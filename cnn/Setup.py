import warnings

import numpy as np
import json
import os
import datetime
import glob
import re
import ntpath

from keras.models import load_model


class Setup(object):
    def __init__(self, name):
        self._name = name
        self._model = None
        self._emptyModel = None

        self._XTrain = None
        self._XValidation = None
        self._XTest = None
        self._YTrain = None
        self._YValidation = None
        self._YTest = None

        self._train_accuracy = []
        self._train_loss = []
        self._val_accuracy = []
        self._val_loss = []
        self._test_accuracy = []
        self._test_loss = []

        self._batch_size = None
        self._epochs = 0

        self._setup = {
            'name': self._name,

            'file': {
                'model': None,
                'model_arch_json': None,
                'model_arch_yaml': None,
                'model_weights': None,
                'XTrain': None,
                'XValidation': None,
                'XTest': None,
                'YTrain': None,
                'YValidation': None,
                'YTest': None,
            },

            'train_accuracy': self._train_accuracy,
            'train_loss': self._train_loss,
            'val_accuracy': self._val_accuracy,
            'val_loss': self._val_loss,
            'test_accuracy': self._test_accuracy,
            'test_loss': self._test_loss,
            'epochs': self._epochs,
        }

    def setName(self, name):
        self._name = name

    def getModel(self):
        return self._model

    def setModel(self, model):
        self._model = model
        # TODO: get the empty model and assign it to self._emptyModel

    def getData(self):
        return self._XTrain, self._YTrain, self._XValidation, self._YValidation, self._XTest, self._YTest

    def setData(self, XTrain=None, YTrain=None, XValidation=None, YValidation=None, XTest=None, YTest=None):
        self._XTrain = XTrain if XTrain is not None else self._XTrain
        self._XValidation = XValidation if XValidation is not None else self._XValidation
        self._XTest = XTest if XTest is not None else self._XTest
        self._YTrain = YTrain if YTrain is not None else self._YTrain
        self._YValidation = YValidation if YValidation is not None else self._YValidation
        self._YTest = YTest if YTest is not None else self._YTest

    def getEpoch(self):
        return self._epochs

    def updateEpochs(self, add_epochs, train_acc, train_loss, val_acc, val_loss, test_acc, test_loss,
                     allow_modify=True):
        # TODO: check

        def checkListLength(length, mList, listName, allowModify):
            modifiedList = mList

            if allow_modify:
                if len(modifiedList) < length:
                    modifiedList.extend([mList[-1] for i in range(length - len(modifiedList))])
                elif len(modifiedList) > length:
                    warnings.warn('%s list is longer than add_epochs. Trimmed list will be used.' % listName)
                    modifiedList = modifiedList[:length]
            else:
                if len(modifiedList) != length:
                    raise ValueError('%s list length is not equal to add_epochs' % listName)

            return modifiedList

        # Checking parameters
        # add_epochs
        if add_epochs is None or type(add_epochs) != int:
            raise TypeError('add_epochs should have type \'int\'')
        elif add_epochs < 0:
            raise ValueError('add_epochs should be > 0')

        if train_acc is None or type(train_acc) != list or \
                test_acc is None or type(test_acc) != list or \
                val_acc is None or type(val_acc) != list or \
                val_loss is None or type(val_loss) != list or \
                test_acc is None or type(test_acc) != list or \
                test_loss is None or type(test_loss) != list:
            raise TypeError('train_acc, test_acc, val_acc, val_loss, test_acc, test_loss should have type \'list\'')

        new_train_acc = checkListLength(add_epochs, train_acc, 'train_acc', allow_modify)
        new_train_loss = checkListLength(add_epochs, train_loss, 'train_loss', allow_modify)
        new_val_acc = checkListLength(add_epochs, val_acc, 'val_acc', allow_modify)
        new_val_loss = checkListLength(add_epochs, val_loss, 'val_loss', allow_modify)
        new_test_acc = checkListLength(add_epochs, test_acc, 'test_acc', allow_modify)
        new_test_loss = checkListLength(add_epochs, test_loss, 'test_loss', allow_modify)

        self._epochs += add_epochs
        self._train_accuracy.extend(new_train_acc)
        self._train_loss.extend(new_train_loss)
        self._val_accuracy.extend(new_val_acc)
        self._val_loss.extend(new_val_loss)
        self._test_accuracy.extend(new_test_acc)
        self._test_loss.extend(new_test_loss)

    def save(self, rel_path):
        # Save every information or object

        if rel_path is None:
            raise ValueError('rel_path should be None')
        else:
            pass

        self._setup = {
            'name': self._name,
            'time': str(datetime.datetime.now()),

            'file': {
                'model': os.path.join(rel_path, self._name, 'model.h5'),
                'model_arch_json': os.path.join(rel_path, self._name, 'model_architecture.json'),
                'model_arch_yaml': os.path.join(rel_path, self._name, 'model_architecture.yaml'),
                'model_weights': os.path.join(rel_path, self._name, 'model_weights.h5'),
                'XTrain': os.path.join(rel_path, self._name, 'XTrain.npy'),
                'XValidation': os.path.join(rel_path, self._name, 'XValidation.npy'),
                'XTest': os.path.join(rel_path, self._name, 'XTest.npy'),
                'YTrain': os.path.join(rel_path, self._name, 'YTrain.npy'),
                'YValidation': os.path.join(rel_path, self._name, 'YValidation.npy'),
                'YTest': os.path.join(rel_path, self._name, 'YTest.npy'),
                'setup': os.path.join(rel_path, self._name, 'setup.json'),
            },
            'train_accuracy': self._train_accuracy,
            'train_loss': self._train_loss,
            'val_accuracy': self._val_accuracy,
            'val_loss': self._val_loss,
            'test_accuracy': self._test_accuracy,
            'test_loss': self._test_loss,

            'epochs': self._epochs,
        }

        if not os.path.exists(os.getcwd() + rel_path):
            os.mkdir(os.getcwd() + rel_path)

        if not os.path.exists(os.path.join(os.getcwd(), rel_path, self._name)):
            os.mkdir(os.path.join(os.getcwd(), rel_path, self._name))

        if len(glob.glob(os.path.join(os.getcwd(), rel_path, self._name, '*.*'))) > 0:
            versions = []
            pattern = r'^.*version(?P<versionnumber>\d*)$'
            for dir in glob.glob(os.path.join(os.getcwd(), rel_path, self._name, 'version*')):
                regex = re.search(pattern, dir)
                versions.append(int(regex.group('versionnumber')))
            if len(versions) == 0:
                maxVer = 0
            else:
                maxVer = np.max(versions)

            newVerDirName = 'version%s' % (maxVer + 1)
            os.mkdir(os.path.join(os.getcwd(), rel_path, self._name, newVerDirName))
            self._backup_version(os.path.join(os.getcwd(), rel_path, self._name),
                                 os.path.join(os.getcwd(), rel_path, self._name, newVerDirName))

        # ==========================================
        # Save whole model
        self._model.save(os.path.join(os.getcwd(), self._setup['file']['model']))

        # ==========================================
        # Save model architecture
        json_model_arch = self._model.to_json()
        with open(os.path.join(os.getcwd(), self._setup['file']['model_arch_json']), 'w') as jsonfile:
            jsonfile.write(json_model_arch)

        yaml_model_arch = self._model.to_yaml()
        with open(os.path.join(os.getcwd(), self._setup['file']['model_arch_yaml']), 'w') as yamlfile:
            yamlfile.write(yaml_model_arch)

        # ==========================================
        # Save model weights
        self._model.save_weights(os.path.join(os.getcwd(), self._setup['file']['model_weights']))

        # ==========================================
        # Save data
        if self._XTrain is not None and type(self._XTrain) == np.ndarray:
            np.save(os.path.join(os.getcwd(), self._setup['file']['XTrain']), self._XTrain)
        if self._XValidation is not None and type(self._XValidation) == np.ndarray:
            np.save(os.path.join(os.getcwd(), self._setup['file']['XValidation']), self._XValidation)
        if self._XTest is not None and type(self._XTest) == np.ndarray:
            np.save(os.path.join(os.getcwd(), self._setup['file']['XTest']), self._XTest)
        if self._YTrain is not None and type(self._YTrain) == np.ndarray:
            np.save(os.path.join(os.getcwd(), self._setup['file']['YTrain']), self._YTrain)
        if self._YValidation is not None and type(self._YValidation) == np.ndarray:
            np.save(os.path.join(os.getcwd(), self._setup['file']['YValidation']), self._YValidation)
        if self._YTest is not None and type(self._YTest) == np.ndarray:
            np.save(os.path.join(os.getcwd(), self._setup['file']['YTest']), self._YTest)

        # ==========================================
        # Save setup
        with open(os.path.join(os.getcwd(), self._setup['file']['setup']), 'w') as setupfile:
            json.dump(self._setup, setupfile)

    def load(self, rel_filepath):
        cwd = os.getcwd()

        if rel_filepath is None:
            raise ValueError('rel_filepath should be None')
        else:
            pass

        # ==========================================
        # Load info
        with open(os.path.join(cwd, rel_filepath), 'r') as setupfile:
            self._setup = json.load(setupfile)

        # ==========================================
        # Load name
        self._name = self._setup['name']

        # ==========================================
        # Load whole model
        self._model = load_model(os.path.join(cwd, self._setup['file']['model']))

        # TODO: if loading from model h5 file fails, then load from model arch file and load weights
        # # ==========================================
        # # Load model architecture
        # with open(os.path.join(directory, self.setup['model_arch_json']), 'r') as jsonfile:
        #     self.emptyModel = model_from_json(jsonfile.read())
        #
        # with open(os.path.join(directory, self.setup['model_arch_yaml']), 'r') as yamlfile:
        #     self.emptyModel = model_from_yaml(yamlfile.read())

        # # ==========================================
        # # Load model weights
        # self.model.load_weights(os.path.join(directory, self.setup['model_weights']))

        # ==========================================
        # Load data
        self._XTrain = np.load(os.path.join(cwd, self._setup['file']['XTrain'])) \
            if os.path.exists(os.path.join(cwd, self._setup['file']['XTrain'])) else self._XTrain
        self._XValidation = np.load(os.path.join(cwd, self._setup['file']['XValidation'])) \
            if os.path.exists(os.path.join(cwd, self._setup['file']['XValidation'])) else self._XValidation
        self._XTest = np.load(os.path.join(cwd, self._setup['file']['XTest'])) \
            if os.path.exists(os.path.join(cwd, self._setup['file']['XTest'])) else self._XTest
        self._YTrain = np.load(os.path.join(cwd, self._setup['file']['YTrain'])) \
            if os.path.exists(os.path.join(cwd, self._setup['file']['YTrain'])) else self._YTrain
        self._YValidation = np.load(os.path.join(cwd, self._setup['file']['YValidation'])) \
            if os.path.exists(os.path.join(cwd, self._setup['file']['YValidation'])) else self._YValidation
        self._YTest = np.load(os.path.join(cwd, self._setup['file']['YTest'])) \
            if os.path.exists(os.path.join(cwd, self._setup['file']['YTest'])) else self._YTest

        # Load info
        self._train_accuracy = self._setup['train_accuracy']
        self._train_loss = self._setup['train_loss']
        self._val_accuracy = self._setup['val_accuracy']
        self._val_loss = self._setup['val_loss']
        self._test_accuracy = self._setup['test_accuracy']
        self._test_loss = self._setup['test_loss']
        self._epochs = self._setup['epochs']

    def _backup_version(self, source, destination):
        ignore_list = ['XTrain',
                       'XValidation',
                       'XTest',
                       'YTrain',
                       'YValidation',
                       'YTest']
        for file in glob.glob(os.path.join(source, '*.*')):
            ignore = False
            for ignored_filename in ignore_list:
                if ignored_filename in file:
                    ignore = True

            if ignore:
                continue
            else:
                os.rename(file, os.path.join(destination, ntpath.basename(file)))
