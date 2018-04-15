# coding: utf-8

# # Import

# In[2]:


import glob
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion

# In[3]:
warnings.filterwarnings('ignore')


# # Load Input Data

# In[4]:


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


# In[30]:


pixels = 256 * 256
data_limit = 5000


# In[31]:


label_train_data = pd.read_csv(os.getcwd() + '\\..\\data\\train.csv', index_col='id')
label_test_data = pd.read_csv(os.getcwd() + '\\..\\data\\test.csv', index_col='id')


# In[32]:


image_train_file_list = glob.glob(os.getcwd() + '\\..\\data\\train\\' + "*.jpg")
image_train_data = []
for index, image_path in enumerate(image_train_file_list):
    image_name = image_path.split('\\')[-1]
    data = [image_path.split('\\')[-1]] + read_image_list(image_path) + [label_train_data.loc[image_name.split('.')[0]]['landmark_id']]
    image_train_data.append(data)

    if (data_limit > 0 or data_limit is not False) and index + 1 >= data_limit:
        break

image_test_file_list = glob.glob(os.getcwd() + '\\..\\data\\test\\' + "*.jpg")
image_test_data = []
for index, image_path in enumerate(image_test_file_list):
    image_name = image_path.split('\\')[-1]
    data = [image_path.split('\\')[-1]] + read_image_list(image_path)
    image_test_data.append(data)

    if (data_limit > 0 or data_limit is not False) and index + 1 >= data_limit:
        break


# In[33]:


column_names = ['filename'] + ['X%s' % i for i in range(1, pixels + 1)] + ['landmark_id']

image_train_data = pd.DataFrame(data=image_train_data, columns=column_names)
image_train_data.set_index('filename')
image_test_data = pd.DataFrame(data=image_test_data, columns=column_names[:-1])
image_test_data.set_index('filename')

print(image_train_data.shape)
print(image_test_data.shape)


# In[34]:


X_train, X_test, y_train, y_test = train_test_split(image_train_data[['X%s' % i for i in range(1, pixels + 1)]].values,
                                                    image_train_data['landmark_id'].values, test_size=1/3)


# # Preprocessing

# In[35]:


numerical_pipeline = Pipeline([
    ('std_scaler', preprocessing.StandardScaler())
])

preprocess_machine = FeatureUnion(transformer_list=[
    ('numerical_pipeline', numerical_pipeline)
])


# In[36]:


X_train = preprocess_machine.fit_transform(X_train)
X_test = preprocess_machine.fit_transform(X_test)


# # Convolutional Neural Network

# In[40]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import to_categorical
from cnn.Setup import Setup


# In[41]:


num_classes = 15000


# In[42]:


cnn = Sequential()
cnn.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(256, 256, 1)))
cnn.add(LeakyReLU(alpha=0.1))
cnn.add(MaxPooling2D((2, 2),padding='same'))
cnn.add(Dropout(0.25))
cnn.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
cnn.add(LeakyReLU(alpha=0.1))
cnn.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
cnn.add(Dropout(0.25))
cnn.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
cnn.add(LeakyReLU(alpha=0.1))
cnn.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
cnn.add(Dropout(0.4))
cnn.add(Flatten())
cnn.add(Dense(256, activation='linear'))
cnn.add(LeakyReLU(alpha=0.1))
cnn.add(Dropout(0.3))
cnn.add(Dense(num_classes, activation='softmax'))


# In[43]:


cnn.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.SGD(),
            metrics=['accuracy'])


# In[45]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# In[44]:

X_train_cnn = X_train.reshape(-1, 256, 256, 1)
X_test_cnn = X_test.reshape(-1, 256, 256, 1)

y_train_one_hot = to_categorical(y_train, num_classes=num_classes)
y_test_one_hot = to_categorical(y_test, num_classes=num_classes)


# In[ ]:


print(X_train_cnn.shape)
print(X_test_cnn.shape)
print(y_train_one_hot.shape)
print(y_test_one_hot.shape)


# In[29]:

cnn_setup = Setup('cnn_landmark_32-64-128-256')
for epoch in range(0, 10000):
    print('Epoch %d' % epoch)
    dropout = cnn.fit(X_train_cnn, y_train_one_hot,
                      batch_size=64, epochs=1, verbose=1,
                      validation_data=(X_test_cnn, y_test_one_hot))

    cnn_setup.setModel(cnn)
    cnn_setup.setData(XTrain=X_train_cnn,
                      YTrain=y_train_one_hot,
                      XValidation=X_test_cnn,
                      YValidation=y_test_one_hot)
    cnn_setup.updateEpochs(add_epochs=1,
                           train_acc=dropout.history['acc'],
                           train_loss=dropout.history['loss'],
                           val_acc=dropout.history['val_acc'],
                           val_loss=dropout.history['val_loss'],
                           test_acc=[0],
                           test_loss=[0],
                           allow_modify=True)
    cnn_setup.save('\\setup\\')
