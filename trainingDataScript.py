import csv
import numpy as np
from PIL import Image
import os, os.path
from sklearn.model_selection import train_test_split

numSamples = {}
topClasses = []
inputs = []
train_Y = []

print('Preprocessing Images')
csvfile = open('train.csv', 'r')
csvreader = csv.reader(csvfile)
data = [line[:3] for line in csvreader]
data = data[1:]
i = 0

"""
for entry in data:
    classVal = int(entry[2])
    numSamples[classVal] = numSamples.get(classVal, 0) + 1

temp = sorted(numSamples, key=numSamples.get)

for kk in reversed(temp):
    topClasses.append(kk)
    i = i + 1
    if i > 150:
        break
i = 0
"""
path = '../images/training'
for entry in data:
    if int(entry[2]) > 450 or int(entry[2]) < 0:
        i = i + 1
        continue

    try:
        filename = entry[0] + '.jpg'
        im = Image.open(os.path.join(path, filename)).convert('LA')
        arr = np.array(im)
        arr = arr[:,:,0]
        arr = arr.astype('float16')
        arr = arr / 255.
        inputs.append(arr)
        train_Y.append(int(entry[2]))
        i = i + 1
    except Exception as e:
        print('Can not identify file %s' %(filename))
        i = i + 1

train_X = np.array(inputs)
train_X = train_X.reshape(-1, 256, 256, 1)

train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y, test_size=0.2, random_state=13)


