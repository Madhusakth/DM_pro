import csv
import numpy as np
from PIL import Image
import os, os.path
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

#from keras.datasets import fashion_mnist
#(train_X,train_Y), (test_X,test_Y) = fashion_mnist.load_data()
i = 0
skipped = []
inputs = []
train_Y = []

print('Preprocessing Images')
path = 'images/training'
for filename in os.listdir(path):
    try:
        im = Image.open(os.path.join(path, filename))
        arr = np.array(im)
        arr = arr[:,:,0]
        arr = arr.astype('float16')
        arr = arr / 255.
        inputs.append(arr)
        i = i + 1
    except Exception as e:
        print('Can not identify file %s' %(filename))
        skipped.append(i)
        i = i + 1

train_X = np.array(inputs)

print('Extracting Class Values')
csvfile = open('train.csv', 'r')
csvreader = csv.reader(csvfile)
data = [line[:3] for line in csvreader]
data = data[1:]
i = 0
for entry in data:
    if i not in skipped:
        train_Y.append(int(entry[2]))

    i = i + 1

train_X = train_X.reshape(-1, 256, 256, 1)

#train_X = train_X.astype('float32')
#test_X = test_X.astype('float32')
#train_X = train_X / 255. 
#test_X = test_X / 255.

train_Y_one_hot = to_categorical(train_Y[0:len(train_X)])
#test_Y_one_hot = to_categorical(test_Y)

train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y, test_size=0.2, random_state=13)

batch_size = 64000
epochs = 20
num_classes = len(train_Y_one_hot[0])

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(256,256,1)))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2),padding='same'))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))                  
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128, activation='linear'))
model.add(LeakyReLU(alpha=0.1))           
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
train_dropout = model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))

#test_eval = model.evaluate(test_X, test_Y_one_hot, verbose=0)
#print('Test loss:', test_eval[0])
#print('Test accuracy: ', test_eval[1])

accuracy = train_dropout.history['acc']
val_accuracy = train_dropout.history['val_acc']
loss = train_dropout.history['loss']
val_loss = train_dropout.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
